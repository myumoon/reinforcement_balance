"""SB3 PPO による BalancePole / CoinGame / Survivors 訓練スクリプト。

使い方:
  python train.py                              # BalancePole (UE5 接続)
  python train.py --game coin                  # CoinGame    (UE5 接続)
  python train.py --game survivors             # Survivors   (UE5 接続)
  python train.py --dry-run                    # BalancePole スタブ
  python train.py --game coin --dry-run        # CoinGame    スタブ
  python train.py --game survivors --dry-run   # Survivors   スタブ
  python train.py --resume runs/balance/v01/train/run-base
  python train.py --game coin --reward-fn eureka_results/my_run/best/reward_fn.py
  python train.py --game survivors --reward-fn eureka_results/my_run/best/reward_fn.py
  python train.py --help
"""

import argparse
import importlib
import importlib.util
import json
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Any

# リファクタリング前に保存されたモデル（entity_attention_extractor モジュールで pickle 化）を
# --resume でロードできるよう、旧モジュールパスを sys.modules に登録する
sys.modules.setdefault(
    "entity_attention_extractor",
    importlib.import_module("base.entity_attention_extractor"),
)

from stable_baselines3 import PPO
import yaml
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecNormalize
import torch

from base.base_ue5_env import UE5ConnectionError
from common.utils import _linear_schedule
from curriculum_callback import CurriculumCallback

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

try:
    from sb3_contrib import RecurrentPPO
    _SB3_CONTRIB_AVAILABLE = True
except ImportError:
    _SB3_CONTRIB_AVAILABLE = False

_GAME_DEFAULTS = {
    "balance":   {"port": 8765},
    "coin":      {"port": 8766},
    "survivors": {"port": 8767},
}


_PPO_KWARGS = dict(
    learning_rate=_linear_schedule(3e-4),
    n_steps=4096,
    batch_size=256,
    n_epochs=10,
    clip_range=0.1,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
)


def _log_device_status(requested_device: str) -> None:
    print(f"[INFO] requested device: {requested_device}")
    print(f"[INFO] torch={torch.__version__}, cuda_available={torch.cuda.is_available()}, "
          f"torch_cuda={torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"[INFO] cuda device[0]: {torch.cuda.get_device_name(0)}")


def _get_raw_env(env):
    """VecEnv ラッパーチェーンを辿って生の環境（DummyVecEnv の env[0]）を返す。

    DummyVecEnv: envs[0] に直接アクセスして返す。
    SubprocVecEnv: 直接アクセス不可のため None を返す（_get_obs_attrs を使うこと）。
    """
    inner = env
    while hasattr(inner, "venv"):
        inner = inner.venv
    if hasattr(inner, "envs"):
        inner = inner.envs[0]
        # gymnasium.Wrapper (Monitor等) は _ 始まりの属性をフォワードしないため unwrapped で剥がす
        if hasattr(inner, "unwrapped"):
            inner = inner.unwrapped
        return inner
    return None  # SubprocVecEnv


def _get_obs_attrs(env) -> tuple[dict, list]:
    """DummyVecEnv / SubprocVecEnv 両対応で _offsets と _obs_schema を取得する。

    entity_attention extractor の構築に必要。
    """
    inner = env
    while hasattr(inner, "venv"):
        inner = inner.venv
    if hasattr(inner, "envs"):
        raw = inner.envs[0]
        if hasattr(raw, "unwrapped"):
            raw = raw.unwrapped
        return getattr(raw, "_offsets", {}), getattr(raw, "_obs_schema", [])
    # SubprocVecEnv: env_method 経由で取得
    offsets = inner.env_method("get_offsets")[0]
    obs_schema = inner.env_method("get_obs_schema")[0]
    return offsets, obs_schema


def _find_vecnormalize(env):
    """VecEnv ラッパーチェーンから VecNormalize レイヤーを見つけて返す（無ければ None）。"""
    cur = env
    while cur is not None:
        if isinstance(cur, VecNormalize):
            return cur
        cur = getattr(cur, "venv", None)
    return None


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")




def _parse_step_shorthand(s: str) -> int:
    """200k/2M/2.5M/2_000k/200_000 などを int に変換する。

    アルゴリズム:
    1. アンダースコアを除去
    2. 末尾が k/K -> float(残部) * 1_000 を int に
    3. 末尾が m/M -> float(残部) * 1_000_000 を int に
    4. それ以外 -> int(s) で変換（小数はエラー）
    """
    s_clean = s.replace("_", "")
    if s_clean.lower().endswith("k"):
        return int(float(s_clean[:-1]) * 1_000)
    if s_clean.lower().endswith("m"):
        return int(float(s_clean[:-1]) * 1_000_000)
    return int(s_clean)


def _parse_resume_spec(spec: str) -> "tuple[Path, int | None]":
    """path@step または path を (Path, int | None) に分解する。

    Examples::

        _parse_resume_spec("runs/run-base@200k")   # (Path("runs/run-base"), 200_000)
        _parse_resume_spec("runs/run-base@2.5M")   # (Path("runs/run-base"), 2_500_000)
        _parse_resume_spec("runs/run-base")         # (Path("runs/run-base"), None)

    最後の @ を分割点とする（パス内に @ が含まれる場合に対応）。
    step が 0 以下の場合は ValueError。
    """
    at_idx = spec.rfind("@")
    if at_idx == -1:
        return Path(spec), None
    path_part = spec[:at_idx]
    step_part = spec[at_idx + 1:]
    step = _parse_step_shorthand(step_part)
    if step <= 0:
        raise ValueError(
            f"@step には正の整数を指定してください: @{step_part!r} -> {step}"
        )
    return Path(path_part), step


def _list_available_steps(source_dir: Path) -> list[int]:
    """source_dir/work/model_steps/model_*_steps.zip をglobしてステップ数の昇順リストを返す。"""
    model_steps_dir = source_dir / "work" / "model_steps"
    if not model_steps_dir.exists():
        return []
    steps = []
    for p in model_steps_dir.glob("model_*_steps.zip"):
        stem = p.stem
        try:
            step_val = int(stem.removeprefix("model_").removesuffix("_steps"))
            steps.append(step_val)
        except ValueError:
            pass
    return sorted(steps)


def _resolve_resume_from_run(
    source_dir: Path,
    step: "int | None",
) -> "tuple[Path, Path | None, dict]":
    """run ディレクトリとステップ指定から (model_zip, vecnorm_path, status_dict) を返す。

    step=None: result/model.zip -> なければ work/model_steps/ 最大ステップへ自動フォールバック。
    step=N:    work/model_steps/model_{N}_steps.zip を使用。
    """
    if not source_dir.exists():
        raise FileNotFoundError(
            f"[ERROR] resume 対象ディレクトリが見つかりません: {source_dir}\n"
            f"ヒント: 新記法は run ディレクトリを指定します。例: resume: runs/survivors/v06/train/run-base"
        )

    if step is None:
        result_model = source_dir / "result" / "model.zip"
        if result_model.exists():
            model_zip = result_model
            vecnorm_path = source_dir / "result" / "vecnormalize.pkl"
            if not vecnorm_path.exists():
                print(f"[WARN] VecNormalize 統計が見つかりません: {vecnorm_path}")
                vecnorm_path = None
            status_path = source_dir / "work" / "train_status.json"
            status_dict = _read_json(status_path) if status_path.exists() else {}
            return model_zip, vecnorm_path, status_dict

        available = _list_available_steps(source_dir)
        if not available:
            raise FileNotFoundError(
                f"[ERROR] resume 対象ディレクトリに再開可能なモデルがありません: {source_dir}\n"
                f"  result/model.zip も work/model_steps/ も見つかりませんでした。"
            )
        max_step = available[-1]
        model_zip = source_dir / "work" / "model_steps" / f"model_{max_step}_steps.zip"
        print(
            f"[WARN] result/model.zip が存在しません。最新チェックポイントを使用します: "
            f"work/model_steps/model_{max_step}_steps.zip"
        )
        vecnorm_path = source_dir / "work" / "vecnormalize" / f"vecnormalize_{max_step}_steps.pkl"
        if not vecnorm_path.exists():
            print(f"[WARN] VecNormalize 統計が見つかりません: {vecnorm_path}")
            vecnorm_path = None
        status_path = source_dir / "work" / "status" / f"train_status_{max_step}_steps.json"
        if not status_path.exists():
            status_path = source_dir / "work" / "train_status.json"
        status_dict = _read_json(status_path) if status_path.exists() else {}
        return model_zip, vecnorm_path, status_dict

    model_zip = source_dir / "work" / "model_steps" / f"model_{step}_steps.zip"
    if not model_zip.exists():
        available = _list_available_steps(source_dir)
        raise FileNotFoundError(
            f"[ERROR] ステップ {step} のモデルが見つかりません: {model_zip}\n"
            f"利用可能なステップ: {available}"
        )
    vecnorm_path = source_dir / "work" / "vecnormalize" / f"vecnormalize_{step}_steps.pkl"
    if not vecnorm_path.exists():
        print(f"[WARN] VecNormalize 統計が見つかりません: {vecnorm_path}")
        vecnorm_path = None
    status_path = source_dir / "work" / "status" / f"train_status_{step}_steps.json"
    status_dict = _read_json(status_path) if status_path.exists() else {}
    return model_zip, vecnorm_path, status_dict


def _model_zip_path(model_base: Path) -> Path:
    return model_base if model_base.suffix.lower() == ".zip" else model_base.with_suffix(".zip")


def _git_value(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _resolved_config(args: argparse.Namespace) -> dict:
    return {
        key: value
        for key, value in vars(args).items()
        if not key.startswith("_")
    }


def _config_hash(config: dict) -> str:
    encoded = json.dumps(config, default=_json_default, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _write_resolved_config(run_dir: Path, args: argparse.Namespace) -> str:
    config = _resolved_config(args)
    config_hash = _config_hash(config)
    payload = {key: (str(value) if isinstance(value, Path) else value) for key, value in config.items()}
    payload["config_hash"] = config_hash
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return config_hash


def _write_run_meta(run_dir: Path, args: argparse.Namespace, config_hash: str) -> None:
    _write_json(run_dir / "run_meta.json", {
        "run_name": args.run_name,
        "game": args.game,
        "config_path": str(args.config) if args.config else None,
        "config_hash": config_hash,
        "git_branch": _git_value(["branch", "--show-current"]),
        "git_commit": _git_value(["rev-parse", "HEAD"]),
    })


def _save_training_status(
    status_path: Path,
    args: argparse.Namespace,
    run_dir: Path,
    latest_model_path: Path,
    latest_vecnorm_path: Path | None,
    curriculum_cb,
    anneal_cb,
    config_hash: str,
    num_timesteps: int,
    exit_reason: str | None = None,
    exit_error: str | None = None,
    curriculum_completion: dict | None = None,
    mirror_paths: list[Path] | None = None,
    noveld_state_path: Path | None = None,
) -> None:
    # run_dir からの相対パスで記録することで新旧両構成に対応
    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(run_dir))
        except ValueError:
            return str(p)

    data = {
        "run_name": args.run_name,
        "game": args.game,
        "global_timestep": num_timesteps,
        "latest_model_path": _rel(latest_model_path),
        "latest_vecnormalize_path": _rel(latest_vecnorm_path) if latest_vecnorm_path else None,
        "config_hash": config_hash,
        "curriculum": curriculum_cb.export_state() if curriculum_cb is not None else None,
        "shaping_anneal": anneal_cb.export_state() if anneal_cb is not None else None,
        "curriculum_completion": curriculum_completion,
        "noveld_state_path": _rel(noveld_state_path) if noveld_state_path is not None else None,
    }
    if exit_reason is not None:
        data["last_exit_reason"] = exit_reason
    if exit_error is not None:
        data["last_exit_error"] = exit_error
    _write_json(status_path, data)
    for mirror_path in mirror_paths or []:
        mirror_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(mirror_path, data)


class _RunCheckpointCallback(BaseCallback):
    def __init__(
        self,
        model_steps_dir: Path,
        vecnorm_dir: Path,
        save_freq: int,
        vecnorm_getter,
        status_writer,
    ):
        super().__init__(verbose=0)
        self._model_steps_dir = model_steps_dir
        self._vecnorm_dir = vecnorm_dir
        self.save_freq = max(save_freq, 1)
        self._vecnorm_getter = vecnorm_getter
        self._status_writer = status_writer
        self._last_save = 0

    def _on_training_start(self) -> None:
        self._last_save = self.num_timesteps

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_save < self.save_freq:
            return True
        self._last_save = self.num_timesteps
        model_base = self._model_steps_dir / f"model_{self.num_timesteps}_steps"
        self.model.save(str(model_base))
        vecnorm_path = None
        vn = self._vecnorm_getter()
        if vn is not None:
            vecnorm_path = self._vecnorm_dir / f"vecnormalize_{self.num_timesteps}_steps.pkl"
            vn.save(str(vecnorm_path))
        self._status_writer(_model_zip_path(model_base), vecnorm_path, self.num_timesteps)
        print(f"[INFO] checkpoint saved: {_model_zip_path(model_base)}")
        return True


class _CurriculumCompletionCallback(BaseCallback):
    """カリキュラム完了条件を満たしたら SB3 の learn を停止する。"""

    def __init__(
        self,
        curriculum_cb,
        window: int,
        min_episodes: int,
        min_score_ratio: float,
        min_episode_len_ratio: float,
    ):
        super().__init__(verbose=0)
        self._curriculum_cb = curriculum_cb
        self.window = max(1, window)
        self.min_episodes = max(1, min_episodes)
        self.min_score_ratio = max(0.0, min_score_ratio)
        self.min_episode_len_ratio = max(0.0, min_episode_len_ratio)
        self.completed = False
        self.completed_timestep: int | None = None
        self.last_diagnostics: dict | None = None

    def _on_training_start(self) -> None:
        if hasattr(self._curriculum_cb, "configure_completion"):
            self._curriculum_cb.configure_completion(
                window=self.window,
                min_episodes=self.min_episodes,
                min_score_ratio=self.min_score_ratio,
                min_episode_len_ratio=self.min_episode_len_ratio,
            )

    def _on_step(self) -> bool:
        if not any(self.locals["dones"]):
            return True
        diagnostics = self._curriculum_cb.get_completion_diagnostics(
            window=self.window,
            min_episodes=self.min_episodes,
            min_score_ratio=self.min_score_ratio,
            min_episode_len_ratio=self.min_episode_len_ratio,
        )
        self.last_diagnostics = diagnostics
        if _WANDB_AVAILABLE and wandb.run:
            wandb.log({
                "curriculum/completion_ready": int(diagnostics["complete"]),
                "curriculum/is_final_phase": int(diagnostics["is_final_phase"]),
                "curriculum/completion_episode_count": diagnostics["episodes"],
                "curriculum/completion_score_mean": diagnostics["active_score_mean"],
                "curriculum/completion_ep_len_mean": diagnostics["episode_length_mean"],
                "global_step": self.num_timesteps,
            }, step=self.num_timesteps)

        if diagnostics["complete"]:
            self.completed = True
            self.completed_timestep = self.num_timesteps
            print(
                "\n[INFO] カリキュラム完了条件を満たしたため訓練を停止します "
                f"(step={self.num_timesteps}, phase={diagnostics['phase_name']}, "
                f"score_mean={diagnostics['active_score_mean']}, "
                f"ep_len_mean={diagnostics['episode_length_mean']})"
            )
            return False
        return True

    def export_state(self) -> dict:
        return {
            "enabled": True,
            "completed": self.completed,
            "completed_timestep": self.completed_timestep,
            "window": self.window,
            "min_episodes": self.min_episodes,
            "min_score_ratio": self.min_score_ratio,
            "min_episode_len_ratio": self.min_episode_len_ratio,
            "last_diagnostics": self.last_diagnostics,
        }


class _AnnealingShapingCallback(BaseCallback):
    """shaped_reward を線形アニーリングするコールバック。

    |shaped_reward_mean| / base_reward_mean の比率を check_freq ステップごとに計算し、
    比率が anneal_threshold を下回ったら anneal_steps かけて shaping_weight を 1.0→min_weight に減衰する。
    min_weight > 0 のとき shaping は永続的に維持される（推論時は不要）。

    n_envs > 1 対応: shaping_weight / reward_fn の操作は env_method で全インスタンスに適用する。
    """

    def __init__(
        self,
        anneal_threshold: float,
        anneal_steps: int,
        check_freq: int,
        min_steps: int,
        min_weight: float = 0.0,
        curriculum_cb=None,
    ):
        super().__init__(verbose=0)
        self.anneal_threshold = anneal_threshold
        self.anneal_steps = anneal_steps
        self.check_freq = check_freq
        self.min_steps = min_steps
        self.min_weight = min_weight
        self._curriculum_cb = curriculum_cb

        self._ep_base_per_env: list[float] = []    # env ごとのエピソード内累積
        self._ep_shaped_per_env: list[float] = []
        self._pending_shaping_weight: float | None = None  # import_state → _on_training_start で適用
        self._sum_base = 0.0
        self._sum_shaped = 0.0
        self._ep_count = 0
        self._last_check = 0
        self._anneal_start_step: int | None = None
        self._final_phase_start_step: int | None = None
        self._reward_fn_cleared = False

    def _on_training_start(self) -> None:
        n = self.training_env.num_envs
        self._ep_base_per_env = [0.0] * n
        self._ep_shaped_per_env = [0.0] * n
        if self._pending_shaping_weight is not None:
            self._set_shaping_weight(self._pending_shaping_weight)
            self._pending_shaping_weight = None

    def _set_shaping_weight(self, weight: float) -> None:
        """全 env に shaping_weight を設定する。"""
        self.training_env.env_method("set_shaping_weight", weight)

    def _clear_reward_fn(self) -> None:
        """全 env の reward_fn を無効化する。"""
        self.training_env.env_method("clear_reward_fn")
        self._reward_fn_cleared = True

    def _get_shaping_weight(self) -> float:
        """env[0] から shaping_weight を取得する（export_state 用）。"""
        if self.training_env is None:
            return self._pending_shaping_weight if self._pending_shaping_weight is not None else 1.0
        return self.training_env.env_method("get_shaping_weight")[0]

    def export_state(self) -> dict:
        return {
            "ep_base": self._ep_base_per_env[0] if self._ep_base_per_env else 0.0,
            "ep_shaped": self._ep_shaped_per_env[0] if self._ep_shaped_per_env else 0.0,
            "sum_base": self._sum_base,
            "sum_shaped": self._sum_shaped,
            "ep_count": self._ep_count,
            "last_check": self._last_check,
            "anneal_start_step": self._anneal_start_step,
            "final_phase_start_step": self._final_phase_start_step,
            "shaping_weight": self._get_shaping_weight(),
        }

    def import_state(self, state: dict) -> None:
        ep_base = float(state.get("ep_base", 0.0))
        ep_shaped = float(state.get("ep_shaped", 0.0))
        n = len(self._ep_base_per_env) if self._ep_base_per_env else 1
        self._ep_base_per_env = [ep_base] + [0.0] * (n - 1)
        self._ep_shaped_per_env = [ep_shaped] + [0.0] * (n - 1)
        self._sum_base = float(state.get("sum_base", 0.0))
        self._sum_shaped = float(state.get("sum_shaped", 0.0))
        self._ep_count = int(state.get("ep_count", 0))
        self._last_check = int(state.get("last_check", 0))
        self._anneal_start_step = state.get("anneal_start_step")
        self._final_phase_start_step = state.get("final_phase_start_step")
        shaping_weight = state.get("shaping_weight")
        if shaping_weight is not None:
            # training_env は _on_training_start まで未設定のため pending に保存
            self._pending_shaping_weight = float(shaping_weight)

    def _anneal_reference_ready(self) -> tuple[bool, int, str]:
        if self._curriculum_cb is None:
            return self.num_timesteps >= self.min_steps, self.num_timesteps, "global"
        if self._final_phase_start_step is None and self._curriculum_cb.is_final_phase:
            self._final_phase_start_step = self.num_timesteps
            print(f"[INFO] 最終フェーズ到達。シェーピングアニーリング基準step={self._final_phase_start_step}")
        if self._final_phase_start_step is None:
            return False, 0, "last_phase"
        elapsed = self.num_timesteps - self._final_phase_start_step
        return elapsed >= self.min_steps, elapsed, "last_phase"

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        dones = self.locals["dones"]
        n_envs = len(infos)

        # _ep_*_per_env が未初期化（resume 直後など）の場合は補完
        if len(self._ep_base_per_env) != n_envs:
            self._ep_base_per_env = [0.0] * n_envs
            self._ep_shaped_per_env = [0.0] * n_envs

        for i, (info, done) in enumerate(zip(infos, dones)):
            self._ep_base_per_env[i]   += info.get("base_reward",   0.0)
            self._ep_shaped_per_env[i] += info.get("shaped_reward", 0.0)

            if done:
                self._sum_base   += self._ep_base_per_env[i]
                self._sum_shaped += abs(self._ep_shaped_per_env[i])
                self._ep_count   += 1
                self._ep_base_per_env[i] = 0.0
                self._ep_shaped_per_env[i] = 0.0

        # check_freq ステップごとに比率を計算してアニーリングをトリガー
        reference_ready, reference_steps, reference_name = self._anneal_reference_ready()
        if (reference_ready
                and self.num_timesteps - self._last_check >= self.check_freq
                and self._ep_count > 0):
            self._last_check = self.num_timesteps
            mean_base   = self._sum_base   / self._ep_count
            mean_shaped = self._sum_shaped / self._ep_count
            ratio = mean_shaped / max(mean_base, 1e-8)
            self._sum_base = self._sum_shaped = 0.0
            self._ep_count = 0
            print(f"[INFO] シェーピング比率: {ratio:.3f} "
                  f"(|shaped|={mean_shaped:.4f} / base={mean_base:.4f}, "
                  f"anneal_reference={reference_name}, reference_steps={reference_steps})")

            if _WANDB_AVAILABLE and wandb.run:
                wandb.log({
                    "shaping/ratio": ratio,
                    "shaping/mean_shaped": mean_shaped,
                    "shaping/mean_base": mean_base,
                    "shaping/anneal_reference_steps": reference_steps,
                    "global_step": self.num_timesteps,
                }, step=self.num_timesteps)

            if ratio < self.anneal_threshold and self._anneal_start_step is None:
                self._anneal_start_step = self.num_timesteps
                print(f"[INFO] shaped_reward アニーリング開始 "
                      f"(ratio={ratio:.3f} < {self.anneal_threshold})")

        # アニーリング中: shaping_weight を 1.0 → min_weight に線形減衰（全 env に適用）
        if self._anneal_start_step is not None:
            progress = (self.num_timesteps - self._anneal_start_step) / self.anneal_steps
            weight = self.min_weight + (1.0 - self.min_weight) * max(0.0, 1.0 - progress)
            self._set_shaping_weight(weight)
            if weight <= self.min_weight and self.min_weight == 0.0 and not self._reward_fn_cleared:
                self._clear_reward_fn()
                print("[INFO] shaped_reward アニーリング完了 → reward_fn を無効化")

        return True


def _load_reward_fn(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"--reward-fn ファイルが見つかりません: {path}")
    spec = importlib.util.spec_from_file_location("_reward_fn_mod", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.reward_shaping


def _strip_zip(path: Path) -> Path:
    """SB3 が .zip を自動付加するため、ユーザーが指定した .zip 拡張子を除去する。"""
    return path.with_suffix("") if path.suffix.lower() == ".zip" else path


def _create_model(args, env, algo_class, default_policy: str, ppo_kwargs: dict):
    """現在の args/env から PPO モデルを新規作成する。

    --init-model と通常新規作成の両方で使う共通パス。
    entity_attention / recurrent の分岐をここに集約する。
    """
    if args.entity_attention and args.game in ("coin", "survivors"):
        offsets, obs_schema = _get_obs_attrs(env)
        if args.game == "survivors":
            from games.survivors.survivors_entity_attention_extractor import SurvivorsEntityAttentionExtractor
            extractor_class = SurvivorsEntityAttentionExtractor
        else:
            from games.coin.coin_entity_attention_extractor import CoinEntityAttentionExtractor
            extractor_class = CoinEntityAttentionExtractor
        policy_kwargs = dict(
            features_extractor_class=extractor_class,
            features_extractor_kwargs=dict(features_dim=128, offsets=offsets, obs_schema=obs_schema),
            net_arch=[64, 64],
        )
        if args.recurrent:
            policy_kwargs["lstm_hidden_size"] = args.lstm_hidden_size
            policy_kwargs["n_lstm_layers"] = args.n_lstm_layers
        print(f"[INFO] {extractor_class.__name__} を使用します (game={args.game}, policy={default_policy})")
        return algo_class(default_policy, env, policy_kwargs=policy_kwargs, **ppo_kwargs)
    elif args.entity_attention:
        print(f"[WARN] --entity-attention はコイン/サバイバーズゲーム専用です。{default_policy} を使用します。")
        return algo_class(default_policy, env, **ppo_kwargs)
    elif args.recurrent:
        policy_kwargs = dict(
            lstm_hidden_size=args.lstm_hidden_size,
            n_lstm_layers=args.n_lstm_layers,
        )
        print(
            f"[INFO] RecurrentPPO (MlpLstmPolicy) を使用します "
            f"(lstm_hidden_size={args.lstm_hidden_size}, n_lstm_layers={args.n_lstm_layers})"
        )
        return algo_class(default_policy, env, policy_kwargs=policy_kwargs, **ppo_kwargs)
    else:
        return algo_class(default_policy, env, **ppo_kwargs)


def _load_policy_weights_from_model(
    model, init_model_path: Path, algo_class, device: str
) -> None:
    """BC済みモデルから policy の state_dict だけを現在モデルへ移植する。

    optimizer state / clip_range / learning_rate / ent_coef / n_steps 等の
    PPO ハイパーパラメータは現在モデル（現在の config）の値を維持する。

    --init-model: 事前学習済み policy を初期値として使う（PPO 設定は引き継がない）
    --resume:     同じ訓練の完全再開（optimizer state / timesteps も含めて復元）
    """
    source = algo_class.load(str(_strip_zip(init_model_path)), device=device)

    src_obs = source.policy.observation_space
    dst_obs = model.policy.observation_space
    if src_obs != dst_obs:
        raise RuntimeError(
            f"[ERROR] --init-model: obs_space 不一致。アーキテクチャが一致するBC モデルを指定してください。\n"
            f"  BC zip:  {src_obs}\n"
            f"  current: {dst_obs}"
        )

    model.policy.load_state_dict(source.policy.state_dict(), strict=True)
    print("[INFO] --init-model: policy weights のみを移植しました (strict=True)")
    print("[INFO] PPO ハイパーパラメータは BC zip ではなく現在の config を使用します")


def parse_args() -> argparse.Namespace:
    # 事前パース: --config のみ抽出（他の引数は無視）
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre.parse_known_args()

    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=None,
                   help="YAML 設定ファイルのパス（CLI 引数で上書き可能）")
    p.add_argument("--game", choices=["balance", "coin", "survivors"], default="balance",
                   help="訓練対象のゲーム (default: balance)")
    p.add_argument("--dry-run", action="store_true", help="UE5 なしでスタブ環境を使用")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=None,
                   help="サーバーポート（未指定時はゲーム別デフォルト: balance=8765, coin=8766, survivors=8767）")
    p.add_argument("--n-envs", type=int, default=1,
                   help="並列 UE5 インスタンス数（--game survivors 専用, default: 1）")
    p.add_argument("--n-steps", type=int, default=4096,
                   help="1 rollout で各 env が収集するステップ数 (default: 4096)。"
                        "--n-envs > 1 では総 rollout = n_steps × n_envs になります")
    p.add_argument("--base-port", type=int, default=None,
                   help="並列時の起点ポート（--n-envs > 1 時は base_port+i を各インスタンスに割り当て。"
                        "未指定時は --port またはゲーム別デフォルトを使用）")
    p.add_argument("--eval-port", type=int, default=None,
                   help="eval 専用 UE5 インスタンスのポート（--eval-freq > 0 かつ survivors 専用）。"
                        "未指定時は base_port + n_envs を自動採用")
    p.add_argument("--total-steps", type=int, default=500_000)
    p.add_argument("--version-name", default=None,
                   help="runs/<game>/<version-name>/ 以下に成果物を保存するバージョン名（新規run時は必須）")
    p.add_argument("--run-name", default=None,
                   help="runs/<game>/<version-name>/train/<run-name>/ に成果物を保存する実行名")
    p.add_argument("--resume", type=str, default=None,
                   help="再開する run_dir または既存モデルのパス（.zip 拡張子は省略・付加どちらでも可）")
    p.add_argument("--checkpoint-freq", type=int, default=10_000,
                   help="チェックポイント保存間隔 (ステップ数, デフォルト: 10000)")
    p.add_argument("--entity-attention", action="store_true",
                   help="エンティティアテンション特徴抽出器を使用 (--game coin/survivors 専用, --resume 時は無視)")
    p.add_argument("--dist-alpha", type=float, default=1.0,
                   help="距離バイアスの強さ (--entity-attention 専用, default: 1.0)")
    p.add_argument("--reward-fn", type=Path, default=None,
                   help="報酬シェーピング関数のパス (例: eureka_results/my_run/best/reward_fn.py, --game coin/survivors 専用)")
    p.add_argument("--no-vec-normalize", action="store_true",
                   help="VecNormalize による観測・報酬の正規化を無効化する")
    p.add_argument("--anneal-threshold", type=float, default=0.1,
                   help="|shaped|/base の比率がこれを下回ったらアニーリング開始 (default: 0.1, --reward-fn 専用)")
    p.add_argument("--anneal-steps", type=int, default=50_000,
                   help="アニーリングにかけるステップ数 (default: 50000)")
    p.add_argument("--anneal-check-freq", type=int, default=5_000,
                   help="比率のチェック間隔・ステップ数 (default: 5000)")
    p.add_argument("--anneal-min-steps", type=int, default=50_000,
                   help="アニーリングチェックを開始する最小ステップ数 (default: 50000)")
    p.add_argument("--anneal-min-weight", type=float, default=0.0,
                   help="shaping_weight の下限値 (default: 0.0=完全除去, 例: 0.05 で5%%維持)")
    p.add_argument("--noveld", action="store_true",
                   help="NovelD 内発的動機を有効化（Survivors のみ）")
    p.add_argument("--noveld-beta", type=float, default=0.3,
                   help="NovelD 内発的報酬スケール係数 β（default: 0.3）")
    p.add_argument("--noveld-alpha", type=float, default=0.5,
                   help="NovelD 差分係数 α（default: 0.5）")
    p.add_argument("--ent-coef", type=float, default=0.01,
                   help="PPO エントロピー係数 (default: 0.01)")
    p.add_argument("--n-epochs", type=int, default=None,
                   help="PPO の勾配更新エポック数（default: 10）")
    p.add_argument("--clip-range", type=float, default=None,
                   help="PPO のクリップ範囲（default: 0.1）")
    p.add_argument("--learning-rate", type=float, default=None,
                   help="PPO の初期学習率・線形スケジュール（default: 3e-4）")
    p.add_argument("--batch-size", type=int, default=None,
                   help="PPO のミニバッチサイズ（default: 256）")
    p.add_argument("--device", default="auto",
                   help="SB3/PyTorch device (auto, cpu, cuda, cuda:0 など。default: auto)")
    p.add_argument("--frame-skip", type=int, default=1,
                   help="フレームスキップ数 N: 1 RL ステップで N 物理ステップ実行（default: 1）")
    p.add_argument("--frame-stack", type=int, default=1,
                   help="観測フレーム数 N: 直近 N ステップの観測を結合する (default: 1=スタックなし)")
    p.add_argument("--recurrent", action="store_true",
                   help="RecurrentPPO (LSTM) ポリシーを使用 (sb3-contrib が必要)")
    p.add_argument("--lstm-hidden-size", type=int, default=256,
                   help="LSTM の隠れ層サイズ (default: 256, --recurrent 専用)")
    p.add_argument("--n-lstm-layers", type=int, default=1,
                   help="LSTM のレイヤー数 (default: 1, --recurrent 専用)")
    p.add_argument("--curriculum", action="store_true",
                   help="カリキュラム学習を有効化（survivors 専用）")
    p.add_argument("--curriculum-window", type=int, default=20,
                   help="active_score を平均するエピソード数 (default: 20)")
    p.add_argument("--curriculum-threshold", type=float, default=5.0,
                   help="Stage 昇格の active_score 閾値 (default: 5.0, 目安: 撃破数×2.0+収集数×3.0)")
    p.add_argument("--curriculum-alive-reward", type=float, default=0.001,
                   help="生存ボーナスの1物理ステップあたりの値 (default: 0.001, UE5側と合わせる)")
    p.add_argument("--until-curriculum-complete", action="store_true",
                   help="survivors カリキュラムが最終Phaseで安定するまで継続し、完了時に自動停止する")
    p.add_argument("--curriculum-complete-window", type=int, default=20,
                   help="カリキュラム完了判定に使う直近エピソード数 (default: 20)")
    p.add_argument("--curriculum-complete-min-episodes", type=int, default=20,
                   help="カリキュラム完了判定に必要な最小エピソード数 (default: 20)")
    p.add_argument("--curriculum-complete-min-score-ratio", type=float, default=1.0,
                   help="最終Phase完了に必要なスコア倍率。直前Phase閾値×curriculum-thresholdに乗算 (default: 1.0)")
    p.add_argument("--curriculum-complete-min-episode-len-ratio", type=float, default=1.0,
                   help="最終Phase完了に必要な episode_length / min_episode_steps の比率 (default: 1.0)")
    p.add_argument("--spalf", action="store_true",
                   help="SPALF 連続パラメータカリキュラム学習を有効化（survivors 専用、--curriculum と排他）")
    p.add_argument("--spalf-r-b", type=float, default=0.1,
                   help="SPALF モード切替閾値（正規化スコアの移動平均がこれを下回ると SPALF モード、default: 0.1）")
    p.add_argument("--spalf-alpha", type=float, default=0.2,
                   help="SPALF f_α の非線形増幅係数（default: 0.2）")
    p.add_argument("--spalf-buffer-size", type=int, default=200,
                   help="SPALF 報酬履歴・ALP バッファサイズ（default: 200）")
    p.add_argument("--spalf-warmup-episodes", type=int, default=50,
                   help="SPALF 開始前のウォームアップエピソード数（default: 50）")
    p.add_argument("--spalf-max-score", type=float, default=2250.0,
                   help="スコア正規化の最大値（default: 2250.0）")
    p.add_argument("--eval-freq", type=int, default=50_000,
                   help="deterministic 評価の間隔 (timesteps, 0=無効, survivors のみ有効, default: 50000)")
    p.add_argument("--eval-episodes", type=int, default=5,
                   help="評価エピソード数 (default: 5, --eval-freq > 0 のとき有効)")
    p.add_argument("--init-model", type=Path, default=None,
                   help="BC 済みモデルを新規 PPO の初期重みとしてロード（--resume とは排他）")
    p.add_argument("--init-vecnormalize", type=Path, default=None,
                   help="--init-model と組み合わせて BC 時の VecNormalize 統計をロード")
    p.add_argument("--wandb", action="store_true", help="W&B ログを有効にする")
    p.add_argument("--wandb-project", default="rl-balance", help="W&B プロジェクト名")
    p.add_argument("--wandb-run-name", default=None, help="W&B ラン名（未指定時は自動生成）")

    # YAML があればデフォルトを差し込む（CLI が常に優先）
    if pre_args.config:
        from common.config import load_yaml_config, apply_yaml_defaults
        apply_yaml_defaults(p, load_yaml_config(pre_args.config))

    return p.parse_args()


def main() -> None:
    _use_wandb = False
    args = parse_args()

    if args.recurrent and not _SB3_CONTRIB_AVAILABLE:
        raise ImportError(
            "--recurrent には sb3-contrib が必要です。"
            "requirements.txt の sb3-contrib>=2.3.0 をインストールしてください。"
        )
    if args.init_model and args.resume:
        raise ValueError("--init-model と --resume は同時に使用できません")
    if args.init_vecnormalize and not args.init_model:
        raise ValueError("--init-vecnormalize は --init-model と組み合わせて使用してください")
    if args.curriculum and args.spalf:
        raise ValueError("--curriculum と --spalf は同時に使用できません")
    if args.recurrent and args.frame_stack > 1:
        print(f"[WARN] --recurrent と --frame-stack={args.frame_stack} を併用しています。"
              " 部分観測対応が二重になるため意図的でなければ片方のみ使用してください。")
    if args.n_envs > 1:
        total_rollout = args.n_steps * args.n_envs
        print(
            f"[INFO] --n-envs={args.n_envs}: 総rollout = {args.n_steps} steps × {args.n_envs} envs"
            f" = {total_rollout} steps/iteration"
        )

    _log_device_status(args.device)
    ppo_kwargs = {**_PPO_KWARGS, "ent_coef": args.ent_coef, "device": args.device, "n_steps": args.n_steps}
    if args.n_epochs is not None:
        ppo_kwargs["n_epochs"] = args.n_epochs
    if args.clip_range is not None:
        ppo_kwargs["clip_range"] = args.clip_range
    if args.learning_rate is not None:
        ppo_kwargs["learning_rate"] = _linear_schedule(args.learning_rate)
    if args.batch_size is not None:
        ppo_kwargs["batch_size"] = args.batch_size

    defaults = _GAME_DEFAULTS[args.game]
    port = args.port if args.port is not None else defaults["port"]
    base_port = args.base_port if args.base_port is not None else port

    if args.n_envs > 1 and args.game != "survivors":
        raise ValueError("--n-envs > 1 は survivors ゲームのみ対応しています")
    if args.n_envs > 1 and args.dry_run:
        raise ValueError("--n-envs > 1 は --dry-run と同時に使用できません")

    # eval port 解決（survivors + non-dry-run + eval_freq > 0 + n_envs > 1 の場合のみ eval 専用 env を作る）
    # n_envs == 1 では既存互換: eval 専用 env なし、training_env を評価に使用
    if args.game == "survivors" and not args.dry_run and args.eval_freq > 0 and args.n_envs > 1:
        _train_ports = set(range(base_port, base_port + args.n_envs))
        if args.eval_port is None:
            args.eval_port = base_port + args.n_envs
        if args.eval_port in _train_ports:
            raise ValueError(
                f"--eval-port が train ports と重複しています: "
                f"eval_port={args.eval_port}, train_ports={sorted(_train_ports)}"
            )

    resume_status = {}
    model_zip_to_load = None
    source_dir = None
    resume_step = None
    is_branch = False
    if args.resume:
        source_dir, resume_step = _parse_resume_spec(args.resume)
        model_zip_to_load, vecnorm_from_run, resume_status = _resolve_resume_from_run(source_dir, resume_step)
        if args.run_name is None:
            if args.config is not None:
                # config-inferred mode: config ファイルの親フォルダを run_dir とする
                # 想定構造: run_dir/config/train_config.yaml → run_dir = config.parent.parent
                run_dir = Path(args.config).resolve().parent.parent
                args.run_name = run_dir.name
                print(f"[INFO] --run-name 未指定: config から run_dir を推論しました → {run_dir}")
            else:
                # continue mode: 同一 run に追記（--config なし時のフォールバック）
                args.run_name = source_dir.name
                run_dir = source_dir
        elif args.version_name:
            # branch mode: 新規 run_dir を生成
            run_dir = Path("runs") / args.game / args.version_name / "train" / args.run_name
        else:
            # --run-name 指定あり + --version-name 未指定は branch mode の設定ミス
            raise ValueError("--version-name は必須です（branch resume 時: --run-name を指定した場合）")
        is_branch = run_dir.resolve() != source_dir.resolve()
        # continue mode で @step 指定は不可（ロールバックは branch mode で行うこと）
        if not is_branch and resume_step is not None:
            raise ValueError(
                "同一 run への resume では @step 指定はできません。\n"
                "最終結果からの継続には resume: {run_path}（@なし）を使用してください。\n"
                "チェックポイントへのロールバックは branch mode（--run-name 指定）を使用してください。"
            )
    else:
        if not args.run_name:
            raise ValueError("--run-name は必須です（--resume <run_dir> 時は省略可能）")
        if not args.version_name:
            raise ValueError("--version-name は必須です（新規 run 時）")
        run_dir = Path("runs") / args.game / args.version_name / "train" / args.run_name

    # サブディレクトリ定義
    work_dir        = run_dir / "work"
    log_dir         = run_dir / "log"
    result_dir      = run_dir / "result"
    model_steps_dir = work_dir / "model_steps"
    vecnorm_dir     = work_dir / "vecnormalize"
    status_dir      = work_dir / "status"
    noveld_dir      = work_dir / "noveld"

    for d in [work_dir, log_dir, result_dir, model_steps_dir, vecnorm_dir, status_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if not args.resume:
        existing_models = [
            *model_steps_dir.glob("model_*_steps.zip"),
            result_dir / "model.zip",
        ]
        existing_models = [p for p in existing_models if p.exists()]
        if existing_models:
            raise FileExistsError(
                f"run-name フォルダに既存モデルがあります。resume を使うか別 run-name を指定してください: "
                f"{run_dir}"
            )

    model_base_path         = result_dir / "model"
    vecnorm_path            = result_dir / "vecnormalize.pkl"
    status_path             = log_dir / "train_status.json"
    work_status_path        = work_dir / "train_status.json"
    curriculum_status_path  = log_dir / "curriculum_status.json"
    spalf_status_path       = log_dir / "spalf_state.json"
    config_hash = _write_resolved_config(log_dir, args)
    _write_run_meta(log_dir, args, config_hash)
    print(f"[INFO] run_dir: {run_dir}")
    print(f"[INFO] config_path: {args.config if args.config else '(none)'}")
    print(f"[INFO] config_hash: {config_hash}")

    _use_wandb = args.wandb and _WANDB_AVAILABLE

    # W&B 初期化:
    #   continue mode (is_branch=False): 同一 run_id で継続（run_dir/work/wandb_run_id.txt を読む）
    #   branch mode  (is_branch=True):  新規 run、元 run を group で紐付け
    wandb_run_id: str | None = None
    _wandb_id_path = work_dir / "wandb_run_id.txt"
    if args.spalf and args.resume and not is_branch:
        # SPALF resume: spalf_state.json から wandb_run_id を読む（wandb_run_id.txt より優先）
        _spalf_wid_path = source_dir / "log" / "spalf_state.json"
        if _spalf_wid_path.exists():
            try:
                _spalf_wid_state = json.loads(_spalf_wid_path.read_text(encoding="utf-8"))
                _spalf_wid = _spalf_wid_state.get("wandb_run_id")
                if _spalf_wid:
                    wandb_run_id = _spalf_wid
                    print(f"[INFO] W&B run_id を spalf_state.json から再利用: {wandb_run_id}")
            except Exception as _e:
                print(f"[WARN] spalf_state.json から wandb_run_id 読み込み失敗: {_e}")
    elif args.resume and not is_branch and _wandb_id_path.exists():
        wandb_run_id = _wandb_id_path.read_text().strip() or None
        if wandb_run_id:
            print(f"[INFO] W&B run_id を再利用します (continue mode): {wandb_run_id}")

    if _use_wandb:
        _wandb_init_kwargs: dict = dict(
            project=args.wandb_project,
            name=args.wandb_run_name or args.run_name,
            sync_tensorboard=True,
            config={
                "game": args.game,
                "run_name": args.run_name,
                "run_dir": str(run_dir),
                "total_steps": args.total_steps,
                "frame_skip": args.frame_skip,
                "frame_stack": args.frame_stack,
                "recurrent": args.recurrent,
                "lstm_hidden_size": args.lstm_hidden_size if args.recurrent else None,
                "n_lstm_layers": args.n_lstm_layers if args.recurrent else None,
                "ent_coef": args.ent_coef,
                "device": args.device,
                "curriculum": args.curriculum,
                "until_curriculum_complete": args.until_curriculum_complete,
                "curriculum_complete_window": args.curriculum_complete_window,
                "curriculum_complete_min_episodes": args.curriculum_complete_min_episodes,
                "curriculum_complete_min_score_ratio": args.curriculum_complete_min_score_ratio,
                "curriculum_complete_min_episode_len_ratio": args.curriculum_complete_min_episode_len_ratio,
                "n_envs": args.n_envs,
                "base_port": base_port,
                "eval_port": args.eval_port,
                "reward_fn": str(args.reward_fn) if args.reward_fn else None,
                "noveld": args.noveld,
                "noveld_beta": args.noveld_beta,
                "noveld_alpha": args.noveld_alpha,
                "spalf": args.spalf,
                "spalf_r_b": args.spalf_r_b,
                "spalf_alpha": args.spalf_alpha,
                "spalf_buffer_size": args.spalf_buffer_size,
                "spalf_warmup_episodes": args.spalf_warmup_episodes,
                "spalf_max_score": args.spalf_max_score,
                "config_hash": config_hash,
                "tensorboard_log": str(log_dir / "tensorboard"),
                **ppo_kwargs,
            },
        )
        if is_branch:
            # branch mode: 元 run を group で紐付け（新規 run として開始）
            _wandb_init_kwargs["group"] = source_dir.name
        else:
            # continue mode または新規: run_id があれば resume="must" で継続
            _wandb_init_kwargs["id"] = wandb_run_id
            _wandb_init_kwargs["resume"] = "must" if wandb_run_id else None
        wandb.init(**_wandb_init_kwargs)
        # カスタム wandb.log メトリクスの x 軸を global_step に統一する。
        # sync_tensorboard=True の TensorBoard メトリクスは自動設定されるが、
        # 手動 wandb.log で記録するメトリクスは define_metric が必要。
        for _prefix in ("survivors/", "spalf/", "episode/", "behavior/",
                         "curriculum/", "expl/", "eval/"):
            wandb.define_metric(f"{_prefix}*", step_metric="global_step")
        # branch mode または新規 run のとき run_id を保存（continue mode は上書きしない）
        if is_branch or not wandb_run_id:
            _wandb_id_path.write_text(wandb.run.id)
        ppo_kwargs["tensorboard_log"] = str(log_dir / "tensorboard")

    # --reward-fn の事前チェック
    reward_fn = None
    if args.reward_fn:
        if args.game not in ("coin", "survivors"):
            print("[WARN] --reward-fn はコイン/サバイバーズゲーム専用です。無視します。")
        elif args.dry_run:
            print("[WARN] --reward-fn は --dry-run 時は無視されます。")
        else:
            reward_fn = _load_reward_fn(args.reward_fn)
            print(f"[INFO] 報酬シェーピング関数をロード: {args.reward_fn}")

    eval_env = None

    if args.dry_run:
        if args.game == "coin":
            from games.coin.coin_env_stub import DummyCoinEnv
            env = make_vec_env(DummyCoinEnv, n_envs=4)
        elif args.game == "survivors":
            from games.survivors.survivors_env_stub import DummySurvivorsEnv
            env = make_vec_env(DummySurvivorsEnv, n_envs=4)
        else:
            from games.balance.balance_env_stub import DummyBalanceEnv
            env = make_vec_env(DummyBalanceEnv, n_envs=4)
        print(f"[dry-run] game={args.game} スタブ環境で実行")
    else:
        if args.game == "coin":
            from games.coin.coin_env import CoinEnv
            def _make_coin_env():
                e = CoinEnv(host=args.host, port=port, frame_skip=args.frame_skip)
                e._reward_fn = reward_fn
                return e
            env = make_vec_env(_make_coin_env, n_envs=1)
        elif args.game == "survivors":
            from games.survivors.survivors_env import SurvivorsEnv
            from stable_baselines3.common.monitor import Monitor
            from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

            class _SurvivorsMonitor(Monitor):
                """Monitor + SurvivorsEnv 固有メソッドの明示フォワード。
                gymnasium の __getattr__ 経由のフォワードは v1.0 で廃止されるため
                set_params / set_shaping_weight を明示定義して非推奨警告を抑制する。"""
                def set_params(self, **kwargs) -> bool:
                    return self.env.set_params(**kwargs)

                def get_params(self) -> dict:
                    return self.env.get_params()

                def set_shaping_weight(self, weight: float) -> None:
                    self.env.set_shaping_weight(weight)

                def clear_reward_fn(self) -> None:
                    self.env.clear_reward_fn()

            def _make_survivors_fn(p: int):
                def _init():
                    e = SurvivorsEnv(host=args.host, port=p, frame_skip=args.frame_skip)
                    e._reward_fn = reward_fn
                    return _SurvivorsMonitor(e)
                return _init

            if args.n_envs > 1:
                env_fns = [_make_survivors_fn(base_port + i) for i in range(args.n_envs)]
                env = DummyVecEnv(env_fns)
                print(f"[INFO] survivors マルチ env: n_envs={args.n_envs}, "
                      f"ports={list(range(base_port, base_port + args.n_envs))}")
                # 全 env の obs_schema_hash が一致しているか確認
                hashes = env.env_method("get_obs_schema_hash")
                if len(set(hashes)) != 1:
                    raise RuntimeError(
                        f"[ERROR] 各 UE5 インスタンスの obs_schema_hash が一致しません: {hashes}\n"
                        "全インスタンスが同じレベル・設定で起動されているか確認してください。"
                    )
                print(f"[INFO] obs_schema_hash 一致確認: {hashes[0]}")
            else:
                env = DummyVecEnv([_make_survivors_fn(base_port)])
            # eval 専用 env 作成（n_envs > 1 かつ eval_freq > 0 の場合のみ）
            # n_envs == 1 は training_env を評価に転用する旧来の動作を維持する
            if args.n_envs > 1 and args.eval_freq > 0:
                eval_env = DummyVecEnv([_make_survivors_fn(args.eval_port)])
                eval_hash = eval_env.env_method("get_obs_schema_hash")[0]
                train_hash = env.env_method("get_obs_schema_hash")[0]
                if eval_hash != train_hash:
                    raise RuntimeError(
                        f"[ERROR] eval env の obs_schema_hash が train env と一致しません。\n"
                        f"  train: {train_hash}\n"
                        f"  eval : {eval_hash}"
                    )
                print(f"[INFO] train ports: {list(range(base_port, base_port + args.n_envs))}")
                print(f"[INFO] eval port  : {args.eval_port}")
                print(f"[INFO] eval env は訓練 env から独立しています")
        else:
            from games.balance.balance_env import BalanceEnv
            env = make_vec_env(
                lambda: BalanceEnv(host=args.host, port=port),
                n_envs=1,
            )
        if args.game == "survivors" and args.n_envs > 1:
            print(f"[INFO] game={args.game}  UE5 サーバー {args.host}:{base_port}〜{base_port + args.n_envs - 1} に接続...")
        else:
            print(f"[INFO] game={args.game}  UE5 サーバー {args.host}:{port} に接続...")

    # VecNormalize 適用
    # - 新規訓練: 正規化統計を初期化
    # - --init-vecnormalize: BC 時の統計をロードして引き継ぐ
    # - 再開: run status またはモデル同階層から既存の統計ファイルをロード
    if not args.no_vec_normalize:
        if args.init_vecnormalize:
            env = VecNormalize.load(str(args.init_vecnormalize), env)
            env.training = True
            print(f"[INFO] VecNormalize 統計をロード (--init-vecnormalize): {args.init_vecnormalize}")
        elif args.resume:
            # _resolve_resume_from_run が返す vecnorm_from_run を使用
            if vecnorm_from_run is not None:
                env = VecNormalize.load(str(vecnorm_from_run), env)
                env.training = True
                print(f"[INFO] VecNormalize 統計をロード: {vecnorm_from_run}")
            else:
                print("[WARN] VecNormalize 統計が見つかりません。VecNormalize を無効化します。")
        else:
            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
            print("[INFO] VecNormalize を有効化しました (norm_obs=True, norm_reward=True)")

    # eval env の VecNormalize（訓練 env の stats を評価直前にコピーするため空の統計で初期化）
    if eval_env is not None and not args.no_vec_normalize:
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)
        print("[INFO] eval env に VecNormalize を適用しました (training=False, norm_reward=False)")

    # フレームスタック: 正規化済み観測を直近 N ステップ分結合する。
    # VecNormalize の running stats は単一フレームに対するため、VecFrameStack は最外側に配置する。
    if args.frame_stack > 1:
        env = VecFrameStack(env, n_stack=args.frame_stack)
        print(f"[INFO] VecFrameStack を有効化しました (n_stack={args.frame_stack})")
    # eval env にも訓練 env と同じ順序で VecFrameStack を適用する
    if eval_env is not None and args.frame_stack > 1:
        eval_env = VecFrameStack(eval_env, n_stack=args.frame_stack)
        print(f"[INFO] eval env に VecFrameStack を適用しました (n_stack={args.frame_stack})")

    algo_class = RecurrentPPO if args.recurrent else PPO
    default_policy = "MlpLstmPolicy" if args.recurrent else "MlpPolicy"

    if args.resume:
        # model_zip_to_load は _resolve_resume_from_run が返す Path（.zip 付き）
        resume_path = str(_strip_zip(model_zip_to_load))
        print(f"[INFO] {model_zip_to_load} から再開")
        if args.entity_attention:
            print("[INFO] --entity-attention は --resume 時は無視されます（保存済みモデルのアーキテクチャを使用）")
        if args.recurrent:
            print("[INFO] --recurrent は --resume 時は保存済みモデルのアーキテクチャに従います")
        load_kwargs = {"tensorboard_log": ppo_kwargs["tensorboard_log"]} if _use_wandb else {}
        model = algo_class.load(resume_path, env=env, device=args.device, **load_kwargs)
    else:
        # --init-model の有無にかかわらず、現在の config でモデルを新規作成する
        model = _create_model(args, env, algo_class, default_policy, ppo_kwargs)
        if args.init_model:
            # policy weights のみを移植する（PPO ハイパーパラメータは現在 config を維持）
            print(f"[INFO] --init-model: {args.init_model} から policy weights を移植します")
            print("[INFO] --resume との違い: --init-model は weights のみ移植。optimizer/PPO設定は現在 config を使用")
            _load_policy_weights_from_model(model, args.init_model, algo_class, args.device)
            model.num_timesteps = 0
            lr_val = ppo_kwargs.get("learning_rate", "N/A")
            cr_val = ppo_kwargs.get("clip_range", "N/A")
            ec_val = ppo_kwargs.get("ent_coef", args.ent_coef)
            ns_val = ppo_kwargs.get("n_steps", "N/A")
            bs_val = ppo_kwargs.get("batch_size", "N/A")
            print(
                f"[INFO] 現在 config の PPO 設定: "
                f"clip_range={cr_val}, learning_rate={lr_val}, "
                f"ent_coef={ec_val}, n_steps={ns_val}, batch_size={bs_val}"
            )
    print(f"[INFO] SB3 model device: {model.device}")

    callbacks = []
    curriculum_cb = None
    curriculum_completion_cb = None
    resume_episode_count = 0
    anneal_cb = None
    noveld_cb = None
    survivors_metrics_callback = None
    survivors_curriculum_metrics_callback = None
    if args.game == "survivors" and not args.dry_run:
        from games.survivors.survivors_training_metrics import (
            ActionDistributionCallback,
            SurvivorsCurriculumProgressMetricsCallback,
            SurvivorsMetricsCallback,
        )
        survivors_metrics_callback = SurvivorsMetricsCallback
        survivors_curriculum_metrics_callback = SurvivorsCurriculumProgressMetricsCallback
        callbacks.append(survivors_metrics_callback(log_freq=5_000, frame_skip=args.frame_skip))
        callbacks.append(ActionDistributionCallback(n_actions=9, log_freq=5_000))
        if args.eval_freq > 0:
            from games.survivors.survivors_eval_callback import SurvivorsEvalCallback
            callbacks.append(SurvivorsEvalCallback(
                eval_env=eval_env,  # None なら n_envs=1 旧互換（training_env を使用）
                eval_freq=args.eval_freq,
                n_eval_episodes=args.eval_episodes,
                frame_skip=args.frame_skip,
                alive_reward=args.curriculum_alive_reward,
            ))
            if eval_env is not None:
                print(f"[INFO] SurvivorsEvalCallback 有効 (eval_freq={args.eval_freq:,}, "
                      f"n_eval_episodes={args.eval_episodes}, eval_port={args.eval_port}, "
                      f"eval_env=isolated)")
            else:
                print(f"[INFO] SurvivorsEvalCallback 有効 (eval_freq={args.eval_freq:,}, "
                      f"n_eval_episodes={args.eval_episodes}, eval_env=training_env[n_envs=1互換])")

    if args.curriculum and args.game == "survivors" and not args.dry_run:
        curriculum_cb = CurriculumCallback(
            raw_env=_get_raw_env(env),
            frame_skip=args.frame_skip,
            window=args.curriculum_window,
            threshold_mult=args.curriculum_threshold,
            alive_reward=args.curriculum_alive_reward,
            status_path=str(curriculum_status_path),
        )
        curriculum_state = resume_status.get("curriculum") if resume_status else None
        if curriculum_state:
            curriculum_cb.import_state(curriculum_state)
            print(f"[INFO] curriculum state を復元: phase={curriculum_state.get('phase_name')}")
        resume_episode_count = len((curriculum_state or {}).get("episode_scores", []))
        callbacks.append(curriculum_cb)
        if _use_wandb and survivors_curriculum_metrics_callback is not None:
            callbacks.append(survivors_curriculum_metrics_callback(curriculum_cb, log_freq=5_000))
        print(f"[INFO] CurriculumCallback 有効 "
              f"(window={args.curriculum_window}, threshold_mult={args.curriculum_threshold}, "
              f"frame_skip={args.frame_skip}, alive_reward={args.curriculum_alive_reward})")
        print(f"[INFO] curriculum_status.json → {curriculum_status_path}")
    elif args.curriculum:
        print("[WARN] --curriculum は survivors ゲームかつ非 dry-run 時のみ有効です。無視します。")

    spalf_cb = None
    if args.spalf and args.game == "survivors" and not args.dry_run:
        from games.survivors.spalf_callback import SpalfCallback
        spalf_cb = SpalfCallback(
            raw_env=_get_raw_env(env),
            frame_skip=args.frame_skip,
            alive_reward=args.curriculum_alive_reward,
            r_b=args.spalf_r_b,
            alpha=args.spalf_alpha,
            max_score=args.spalf_max_score,
            buffer_size=args.spalf_buffer_size,
            warmup_episodes=args.spalf_warmup_episodes,
            status_path=str(spalf_status_path),
        )
        if args.resume:
            _spalf_state_file = source_dir / "log" / "spalf_state.json"
            if _spalf_state_file.exists():
                _spalf_state = json.loads(_spalf_state_file.read_text(encoding="utf-8"))
                spalf_cb.import_state(_spalf_state)
                print(f"[INFO] SPALF state を復元: total_episodes={_spalf_state.get('total_episodes', 0)}")
            else:
                print("[WARN] --spalf resume: spalf_state.json が見つかりません。新規開始します。")
        callbacks.append(spalf_cb)
        print(
            f"[INFO] SpalfCallback 有効 "
            f"(r_b={args.spalf_r_b}, alpha={args.spalf_alpha}, "
            f"buffer_size={args.spalf_buffer_size}, warmup_episodes={args.spalf_warmup_episodes}, "
            f"max_score={args.spalf_max_score})"
        )
        print(f"[INFO] spalf_state.json → {spalf_status_path}")
    elif args.spalf:
        print("[WARN] --spalf は survivors ゲームかつ非 dry-run 時のみ有効です。無視します。")

    if args.until_curriculum_complete:
        if curriculum_cb is None:
            raise ValueError("--until-curriculum-complete には survivors の --curriculum が必要です")
        curriculum_completion_cb = _CurriculumCompletionCallback(
            curriculum_cb=curriculum_cb,
            window=args.curriculum_complete_window,
            min_episodes=args.curriculum_complete_min_episodes,
            min_score_ratio=args.curriculum_complete_min_score_ratio,
            min_episode_len_ratio=args.curriculum_complete_min_episode_len_ratio,
        )
        callbacks.append(curriculum_completion_cb)
        print(
            "[INFO] カリキュラム完了まで継続する停止条件を有効化 "
            f"(max_total_steps={args.total_steps:,}, window={args.curriculum_complete_window}, "
            f"min_episodes={args.curriculum_complete_min_episodes}, "
            f"score_ratio={args.curriculum_complete_min_score_ratio}, "
            f"ep_len_ratio={args.curriculum_complete_min_episode_len_ratio})"
        )

    if args.game == "survivors" and args.noveld and not args.dry_run:
        from games.survivors.noveld_callback import NovelDCallback
        noveld_cb = NovelDCallback(
            beta=args.noveld_beta,
            alpha=args.noveld_alpha,
            verbose=1,
        )
        callbacks.append(noveld_cb)
        print(f"[INFO] NovelDCallback 有効 (beta={args.noveld_beta}, alpha={args.noveld_alpha})")
        noveld_state_rel = resume_status.get("noveld_state_path") if resume_status else None
        if noveld_state_rel and source_dir is not None:
            noveld_cb.load_from_file(source_dir / noveld_state_rel)
            print("[INFO] NovelD state を resume から復元しました。")
        elif noveld_state_rel:
            print("[WARN] NovelD state パスが見つかりましたが source_dir が不明のためスキップします。")
    elif args.noveld and args.dry_run:
        print("[WARN] --noveld は --dry-run 時は無視されます。")

    if reward_fn is not None and args.game in ("coin", "survivors"):
        anneal_cb = _AnnealingShapingCallback(
            anneal_threshold=args.anneal_threshold,
            anneal_steps=args.anneal_steps,
            check_freq=args.anneal_check_freq,
            min_steps=args.anneal_min_steps,
            min_weight=args.anneal_min_weight,
            curriculum_cb=curriculum_cb if args.curriculum else None,
        )
        anneal_state = resume_status.get("shaping_anneal") if resume_status else None
        if anneal_state:
            anneal_cb.import_state(anneal_state)
            print("[INFO] shaping anneal state を復元")
        callbacks.append(anneal_cb)
        reference = "last_phase" if curriculum_cb is not None else "global"
        print(f"[INFO] シェーピングアニーリング有効 "
              f"(threshold={args.anneal_threshold}, anneal_steps={args.anneal_steps:,}, "
              f"min_steps={args.anneal_min_steps:,}, min_weight={args.anneal_min_weight}, "
              f"reference={reference})")

    def _write_status_for_model(
        model_zip: Path,
        vecnorm_file: Path | None,
        timestep: int,
        exit_reason: str | None = None,
        exit_error: str | None = None,
    ) -> None:
        completion_state = (
            curriculum_completion_cb.export_state()
            if curriculum_completion_cb is not None
            else (
                {"enabled": False, "last_diagnostics": curriculum_cb.get_completion_diagnostics()}
                if curriculum_cb is not None and hasattr(curriculum_cb, "get_completion_diagnostics")
                else None
            )
        )
        noveld_pt_path = None
        if noveld_cb is not None:
            noveld_pt_path = noveld_dir / f"noveld_{timestep}_steps.pt"
            noveld_cb.save_to_file(noveld_pt_path)
        _save_training_status(
            status_path=status_path,
            args=args,
            run_dir=run_dir,
            latest_model_path=model_zip,
            latest_vecnorm_path=vecnorm_file,
            curriculum_cb=curriculum_cb,
            anneal_cb=anneal_cb,
            config_hash=config_hash,
            num_timesteps=timestep,
            exit_reason=exit_reason,
            exit_error=exit_error,
            curriculum_completion=completion_state,
            mirror_paths=[
                work_status_path,
                status_dir / f"train_status_{timestep}_steps.json",
            ],
            noveld_state_path=noveld_pt_path,
        )

    checkpoint_cb = _RunCheckpointCallback(
        model_steps_dir=model_steps_dir,
        vecnorm_dir=vecnorm_dir,
        save_freq=max(args.checkpoint_freq, 1),
        vecnorm_getter=lambda: _find_vecnormalize(env),
        status_writer=_write_status_for_model,
    )
    callbacks.append(checkpoint_cb)

    exit_reason = "completed"
    exit_error = None

    try:
        # --resume: 継続 (False), --init-model / 新規: 0から開始 (True)
        reset_timesteps = args.resume is None
        model.learn(total_timesteps=args.total_steps, callback=callbacks,
                    reset_num_timesteps=reset_timesteps)
        if curriculum_completion_cb is not None and curriculum_completion_cb.completed:
            exit_reason = "curriculum_complete"
    except KeyboardInterrupt:
        exit_reason = "keyboard_interrupt"
        print("\n[INFO] 訓練を中断しました。モデルを保存します...")
    except UE5ConnectionError as e:
        exit_reason = "ue5_connection_lost"
        exit_error = str(e)
        print("\n[WARN] UE5 HTTP 接続が復旧できませんでした。モデルを保存して終了します。")
        print(f"[WARN] {exit_error}")
    finally:
        model.save(str(model_base_path))
        final_model_zip = _model_zip_path(model_base_path)
        print(f"[INFO] Model saved to {final_model_zip}")
        vn = _find_vecnormalize(env)
        final_vecnorm_path = None
        if vn is not None:
            vn.save(str(vecnorm_path))
            final_vecnorm_path = vecnorm_path
            print(f"[INFO] VecNormalize 統計を保存: {vecnorm_path}")
        _write_status_for_model(
            final_model_zip,
            final_vecnorm_path,
            model.num_timesteps,
            exit_reason=exit_reason,
            exit_error=exit_error,
        )
        if spalf_cb is not None:
            spalf_cb._save_status()
            print(f"[INFO] SPALF state を保存: {spalf_status_path}")
        env.close()
        if eval_env is not None:
            eval_env.close()
        if args.curriculum and curriculum_cb is not None and args.total_steps >= 20_000:
            current_episode_count = len(curriculum_cb.export_state().get("episode_scores", []))
            if current_episode_count <= resume_episode_count:
                print(
                    "[WARN] curriculum episode count が増えていません "
                    f"(resume時={resume_episode_count}, 終了時={current_episode_count})。"
                    " Monitor / info['episode'] を確認してください。"
                )
        if _use_wandb:
            try:
                if wandb.run:
                    wandb.run.summary["run/exit_reason"] = exit_reason
                    wandb.log({
                        "run/ended_by_ue5_connection_lost": (
                            1 if exit_reason == "ue5_connection_lost" else 0
                        ),
                        "global_step": model.num_timesteps,
                    }, step=model.num_timesteps)
                wandb.finish()
            except Exception as e:
                print(f"[WARN] W&B finish/log failed: {e}")


if __name__ == "__main__":
    main()
