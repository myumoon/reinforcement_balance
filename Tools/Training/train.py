"""SB3 PPO による BalancePole / CoinGame / Survivors 訓練スクリプト。

使い方:
  python train.py                              # BalancePole (UE5 接続)
  python train.py --game coin                  # CoinGame    (UE5 接続)
  python train.py --game survivors             # Survivors   (UE5 接続)
  python train.py --dry-run                    # BalancePole スタブ
  python train.py --game coin --dry-run        # CoinGame    スタブ
  python train.py --game survivors --dry-run   # Survivors   スタブ
  python train.py --resume models/balance_model
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
    """VecEnv ラッパーチェーンを辿って生の環境（CoinEnv）を返す。"""
    inner = env
    while hasattr(inner, "venv"):
        inner = inner.venv
    if hasattr(inner, "envs"):
        inner = inner.envs[0]
    # gymnasium.Wrapper (Monitor等) は _ 始まりの属性をフォワードしないため unwrapped で剥がす
    if hasattr(inner, "unwrapped"):
        inner = inner.unwrapped
    return inner


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


def _model_zip_path(model_base: Path) -> Path:
    return model_base if model_base.suffix.lower() == ".zip" else model_base.with_suffix(".zip")


def _latest_checkpoint(run_dir: Path) -> Path | None:
    candidates = list(run_dir.glob("model_*_steps.zip"))
    if not candidates:
        final_model = run_dir / "model.zip"
        return final_model if final_model.exists() else None

    def _step(path: Path) -> int:
        stem = path.stem
        try:
            return int(stem.removeprefix("model_").removesuffix("_steps"))
        except ValueError:
            return -1

    return max(candidates, key=_step)


def _resolve_resume_path(resume: Path) -> tuple[Path, Path, dict]:
    if resume.is_dir():
        run_dir = resume
        status = _read_json(run_dir / "train_status.json")
        model_path = status.get("latest_model_path")
        if model_path:
            model_zip = Path(model_path)
            if not model_zip.is_absolute():
                model_zip = run_dir / model_zip
        else:
            model_zip = _latest_checkpoint(run_dir)
        if model_zip is None or not model_zip.exists():
            raise FileNotFoundError(f"--resume run_dir に再開可能なモデルがありません: {run_dir}")
        return run_dir, _strip_zip(model_zip), status

    model_zip = _model_zip_path(resume)
    if not model_zip.exists():
        raise FileNotFoundError(f"--resume モデルが見つかりません: {model_zip}")
    run_dir = model_zip.parent
    return run_dir, _strip_zip(model_zip), _read_json(run_dir / "train_status.json")


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
) -> None:
    data = {
        "run_name": args.run_name,
        "game": args.game,
        "global_timestep": num_timesteps,
        "latest_model_path": latest_model_path.name,
        "latest_vecnormalize_path": latest_vecnorm_path.name if latest_vecnorm_path else None,
        "config_hash": config_hash,
        "curriculum": curriculum_cb.export_state() if curriculum_cb is not None else None,
        "shaping_anneal": anneal_cb.export_state() if anneal_cb is not None else None,
        "curriculum_completion": curriculum_completion,
    }
    if exit_reason is not None:
        data["last_exit_reason"] = exit_reason
    if exit_error is not None:
        data["last_exit_error"] = exit_error
    _write_json(status_path, data)


class _RunCheckpointCallback(BaseCallback):
    def __init__(
        self,
        run_dir: Path,
        save_freq: int,
        vecnorm_getter,
        status_writer,
    ):
        super().__init__(verbose=0)
        self.run_dir = run_dir
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
        model_base = self.run_dir / f"model_{self.num_timesteps}_steps"
        self.model.save(str(model_base))
        vecnorm_path = None
        vn = self._vecnorm_getter()
        if vn is not None:
            vecnorm_path = self.run_dir / f"vecnormalize_{self.num_timesteps}_steps.pkl"
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
        if not self.locals["dones"][0]:
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
    """

    def __init__(
        self,
        raw_env,
        anneal_threshold: float,
        anneal_steps: int,
        check_freq: int,
        min_steps: int,
        min_weight: float = 0.0,
        curriculum_cb=None,
    ):
        super().__init__(verbose=0)
        self._raw_env = raw_env
        self.anneal_threshold = anneal_threshold
        self.anneal_steps = anneal_steps
        self.check_freq = check_freq
        self.min_steps = min_steps
        self.min_weight = min_weight
        self._curriculum_cb = curriculum_cb

        self._ep_base = 0.0
        self._ep_shaped = 0.0
        self._sum_base = 0.0
        self._sum_shaped = 0.0
        self._ep_count = 0
        self._last_check = 0
        self._anneal_start_step: int | None = None
        self._final_phase_start_step: int | None = None

    def export_state(self) -> dict:
        return {
            "ep_base": self._ep_base,
            "ep_shaped": self._ep_shaped,
            "sum_base": self._sum_base,
            "sum_shaped": self._sum_shaped,
            "ep_count": self._ep_count,
            "last_check": self._last_check,
            "anneal_start_step": self._anneal_start_step,
            "final_phase_start_step": self._final_phase_start_step,
            "shaping_weight": getattr(self._raw_env, "shaping_weight", None),
        }

    def import_state(self, state: dict) -> None:
        self._ep_base = float(state.get("ep_base", 0.0))
        self._ep_shaped = float(state.get("ep_shaped", 0.0))
        self._sum_base = float(state.get("sum_base", 0.0))
        self._sum_shaped = float(state.get("sum_shaped", 0.0))
        self._ep_count = int(state.get("ep_count", 0))
        self._last_check = int(state.get("last_check", 0))
        self._anneal_start_step = state.get("anneal_start_step")
        self._final_phase_start_step = state.get("final_phase_start_step")
        shaping_weight = state.get("shaping_weight")
        if shaping_weight is not None:
            self._raw_env.shaping_weight = float(shaping_weight)

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
        info = self.locals["infos"][0]
        self._ep_base   += info.get("base_reward",   0.0)
        self._ep_shaped += info.get("shaped_reward", 0.0)

        if self.locals["dones"][0]:
            self._sum_base   += self._ep_base
            self._sum_shaped += abs(self._ep_shaped)
            self._ep_count   += 1
            self._ep_base = self._ep_shaped = 0.0

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
                }, step=self.num_timesteps)

            if ratio < self.anneal_threshold and self._anneal_start_step is None:
                self._anneal_start_step = self.num_timesteps
                print(f"[INFO] shaped_reward アニーリング開始 "
                      f"(ratio={ratio:.3f} < {self.anneal_threshold})")

        # アニーリング中: shaping_weight を 1.0 → min_weight に線形減衰
        if self._anneal_start_step is not None:
            progress = (self.num_timesteps - self._anneal_start_step) / self.anneal_steps
            weight = self.min_weight + (1.0 - self.min_weight) * max(0.0, 1.0 - progress)
            self._raw_env.shaping_weight = weight
            if weight <= self.min_weight and self.min_weight == 0.0 \
                    and self._raw_env._reward_fn is not None:
                self._raw_env._reward_fn = None
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
    p.add_argument("--total-steps", type=int, default=500_000)
    p.add_argument("--run-name", default=None,
                   help="results/<game>/<run-name>/ に成果物を保存する実行名")
    p.add_argument("--resume", type=Path, default=None,
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
    p.add_argument("--ent-coef", type=float, default=0.01,
                   help="PPO エントロピー係数 (default: 0.01)")
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
    if args.recurrent and args.frame_stack > 1:
        print(f"[WARN] --recurrent と --frame-stack={args.frame_stack} を併用しています。"
              " 部分観測対応が二重になるため意図的でなければ片方のみ使用してください。")

    _log_device_status(args.device)
    ppo_kwargs = {**_PPO_KWARGS, "ent_coef": args.ent_coef, "device": args.device}

    defaults = _GAME_DEFAULTS[args.game]
    port = args.port if args.port is not None else defaults["port"]

    resume_status = {}
    resume_model_base = None
    resume_source_dir = None
    if args.resume:
        resume_source_dir, resume_model_base, resume_status = _resolve_resume_path(args.resume)
        if args.run_name is None:
            if args.resume.is_dir() or (resume_source_dir / "train_status.json").exists():
                args.run_name = resume_source_dir.name
                run_dir = resume_source_dir
            else:
                args.run_name = resume_model_base.name
                run_dir = Path("results") / args.game / args.run_name
        else:
            run_dir = Path("results") / args.game / args.run_name
    else:
        if not args.run_name:
            raise ValueError("--run-name は必須です（--resume <run_dir|model.zip> 時は省略可能）")
        run_dir = Path("results") / args.game / args.run_name

    run_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        existing_models = [run_dir / "model.zip", *run_dir.glob("model_*_steps.zip")]
        existing_models = [path for path in existing_models if path.exists()]
        if existing_models:
            raise FileExistsError(
                f"run-name フォルダに既存モデルがあります。resume を使うか別 run-name を指定してください: "
                f"{run_dir}"
            )

    model_base_path = run_dir / "model"
    vecnorm_path = run_dir / "vecnormalize.pkl"
    status_path = run_dir / "train_status.json"
    curriculum_status_path = run_dir / "curriculum_status.json"
    config_hash = _write_resolved_config(run_dir, args)
    _write_run_meta(run_dir, args, config_hash)
    print(f"[INFO] run_dir: {run_dir}")
    print(f"[INFO] config_path: {args.config if args.config else '(none)'}")
    print(f"[INFO] config_hash: {config_hash}")

    _use_wandb = args.wandb and _WANDB_AVAILABLE
    if _use_wandb:
        wandb.init(
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
                "reward_fn": str(args.reward_fn) if args.reward_fn else None,
                "config_hash": config_hash,
                "tensorboard_log": str(run_dir / "tensorboard"),
                **_PPO_KWARGS,
            },
        )
        ppo_kwargs["tensorboard_log"] = str(run_dir / "tensorboard")

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
            def _make_survivors_env():
                e = SurvivorsEnv(host=args.host, port=port, frame_skip=args.frame_skip)
                e._reward_fn = reward_fn
                return e
            env = make_vec_env(_make_survivors_env, n_envs=1)
        else:
            from games.balance.balance_env import BalanceEnv
            env = make_vec_env(
                lambda: BalanceEnv(host=args.host, port=port),
                n_envs=1,
            )
        print(f"[INFO] game={args.game}  UE5 サーバー {args.host}:{port} に接続...")

    # VecNormalize 適用
    # - 新規訓練: 正規化統計を初期化
    # - 再開: run status またはモデル同階層から既存の統計ファイルをロード
    if not args.no_vec_normalize:
        if args.resume:
            status_vecnorm = resume_status.get("latest_vecnormalize_path")
            load_path = Path(status_vecnorm) if status_vecnorm else None
            if load_path is not None and not load_path.is_absolute():
                load_path = resume_source_dir / load_path
            if load_path is None or not load_path.exists():
                resume_vecnorm = resume_source_dir / f"{resume_model_base.name}_vecnorm.pkl"
                final_vecnorm = resume_source_dir / "vecnormalize.pkl"
                checkpoint_vecnorm = resume_source_dir / f"vecnormalize_{resume_model_base.name.removeprefix('model_').removesuffix('_steps')}_steps.pkl"
                old_prefix = resume_model_base.name
                parts = old_prefix.rsplit("_", 2)
                old_prefix_vecnorm = None
                if len(parts) == 3 and parts[1].isdigit() and parts[2] == "steps":
                    old_prefix_vecnorm = resume_source_dir / f"{parts[0]}_vecnorm.pkl"
                load_path = next(
                    (
                        path
                        for path in (checkpoint_vecnorm, final_vecnorm, resume_vecnorm, old_prefix_vecnorm)
                        if path is not None and path.exists()
                    ),
                    None,
                )
            if load_path is None:
                print("[WARN] VecNormalize 統計ファイルが見つかりません。VecNormalize を無効化します。"
                      f" (resume_source_dir: {resume_source_dir})")
            else:
                env = VecNormalize.load(str(load_path), env)
                env.training = True
                print(f"[INFO] VecNormalize 統計をロード: {load_path}")
        else:
            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
            print("[INFO] VecNormalize を有効化しました (norm_obs=True, norm_reward=True)")

    # フレームスタック: 正規化済み観測を直近 N ステップ分結合する。
    # VecNormalize の running stats は単一フレームに対するため、VecFrameStack は最外側に配置する。
    if args.frame_stack > 1:
        env = VecFrameStack(env, n_stack=args.frame_stack)
        print(f"[INFO] VecFrameStack を有効化しました (n_stack={args.frame_stack})")

    algo_class = RecurrentPPO if args.recurrent else PPO
    default_policy = "MlpLstmPolicy" if args.recurrent else "MlpPolicy"

    if args.resume:
        resume_path = str(resume_model_base)
        print(f"[INFO] {resume_path} から再開")
        if args.entity_attention:
            print("[INFO] --entity-attention は --resume 時は無視されます（保存済みモデルのアーキテクチャを使用）")
        if args.recurrent:
            print("[INFO] --recurrent は --resume 時は保存済みモデルのアーキテクチャに従います")
        load_kwargs = {"tensorboard_log": ppo_kwargs["tensorboard_log"]} if _use_wandb else {}
        model = algo_class.load(resume_path, env=env, device=args.device, **load_kwargs)
    elif args.entity_attention:
        if args.game not in ("coin", "survivors"):
            print(f"[WARN] --entity-attention はコイン/サバイバーズゲーム専用です。{default_policy} を使用します。")
            model = algo_class(default_policy, env, **ppo_kwargs)
        else:
            offsets = getattr(_get_raw_env(env), "_offsets", {})
            if args.game == "survivors":
                from games.survivors.survivors_entity_attention_extractor import SurvivorsEntityAttentionExtractor
                extractor_class = SurvivorsEntityAttentionExtractor
            else:  # coin
                from games.coin.coin_entity_attention_extractor import CoinEntityAttentionExtractor
                extractor_class = CoinEntityAttentionExtractor
            policy_kwargs = dict(
                features_extractor_class=extractor_class,
                features_extractor_kwargs=dict(features_dim=128, offsets=offsets),
                net_arch=[64, 64],
            )
            if args.recurrent:
                policy_kwargs["lstm_hidden_size"] = args.lstm_hidden_size
                policy_kwargs["n_lstm_layers"] = args.n_lstm_layers
            print(f"[INFO] {extractor_class.__name__} を使用します (game={args.game}, policy={default_policy})")
            model = algo_class(default_policy, env, policy_kwargs=policy_kwargs, **ppo_kwargs)
    elif args.recurrent:
        policy_kwargs = dict(
            lstm_hidden_size=args.lstm_hidden_size,
            n_lstm_layers=args.n_lstm_layers,
        )
        print(f"[INFO] RecurrentPPO (MlpLstmPolicy) を使用します "
              f"(lstm_hidden_size={args.lstm_hidden_size}, n_lstm_layers={args.n_lstm_layers})")
        model = algo_class(default_policy, env, policy_kwargs=policy_kwargs, **ppo_kwargs)
    else:
        model = algo_class(default_policy, env, **ppo_kwargs)
    print(f"[INFO] SB3 model device: {model.device}")

    callbacks = []
    curriculum_cb = None
    curriculum_completion_cb = None
    anneal_cb = None
    survivors_metrics_callback = None
    survivors_curriculum_metrics_callback = None
    if args.game == "survivors" and not args.dry_run:
        from games.survivors.survivors_training_metrics import (
            SurvivorsCurriculumProgressMetricsCallback,
            SurvivorsMetricsCallback,
        )
        survivors_metrics_callback = SurvivorsMetricsCallback
        survivors_curriculum_metrics_callback = SurvivorsCurriculumProgressMetricsCallback
        callbacks.append(survivors_metrics_callback(log_freq=5_000))

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
        callbacks.append(curriculum_cb)
        if _use_wandb and survivors_curriculum_metrics_callback is not None:
            callbacks.append(survivors_curriculum_metrics_callback(curriculum_cb, log_freq=5_000))
        print(f"[INFO] CurriculumCallback 有効 "
              f"(window={args.curriculum_window}, threshold_mult={args.curriculum_threshold}, "
              f"frame_skip={args.frame_skip}, alive_reward={args.curriculum_alive_reward})")
        print(f"[INFO] curriculum_status.json → {curriculum_status_path}")
    elif args.curriculum:
        print("[WARN] --curriculum は survivors ゲームかつ非 dry-run 時のみ有効です。無視します。")

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

    if reward_fn is not None and args.game in ("coin", "survivors"):
        anneal_cb = _AnnealingShapingCallback(
            raw_env=_get_raw_env(env),
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
        )

    checkpoint_cb = _RunCheckpointCallback(
        run_dir=run_dir,
        save_freq=max(args.checkpoint_freq // (env.num_envs or 1), 1),
        vecnorm_getter=lambda: _find_vecnormalize(env),
        status_writer=_write_status_for_model,
    )
    callbacks.append(checkpoint_cb)

    exit_reason = "completed"
    exit_error = None

    try:
        model.learn(total_timesteps=args.total_steps, callback=callbacks,
                    reset_num_timesteps=args.resume is None)
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
        env.close()
        if _use_wandb:
            try:
                if wandb.run:
                    wandb.run.summary["run/exit_reason"] = exit_reason
                    wandb.log({
                        "run/ended_by_ue5_connection_lost": (
                            1 if exit_reason == "ue5_connection_lost" else 0
                        ),
                    }, step=model.num_timesteps)
                wandb.finish()
            except Exception as e:
                print(f"[WARN] W&B finish/log failed: {e}")


if __name__ == "__main__":
    main()
