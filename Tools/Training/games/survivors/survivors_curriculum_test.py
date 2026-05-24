"""フェーズ型カリキュラムの昇格チェックスクリプト（推論専用）。

使用例:
  python survivors_curriculum_test.py 09_goldilocks_phase12
  python survivors_curriculum_test.py 09_goldilocks_phase12 --start-phase 12 --advance-patience-steps 300000
  python survivors_curriculum_test.py 09_goldilocks --config runs/survivors/v06/train/09_goldilocks/config/train_config_resume.yaml
"""

import sys
from pathlib import Path

_TRAINING_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

import argparse
import json

import torch
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor


class _MockModel:
    """CurriculumCallback が参照する num_timesteps の追跡用シム。"""

    def __init__(self, start_timesteps: int = 0):
        self.num_timesteps = start_timesteps


def parse_args() -> argparse.Namespace:
    # 事前パース: --config のみ抽出（他の引数は無視）
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(description="フェーズ型カリキュラム昇格チェック（推論専用）")
    p.add_argument("run", type=str, help="run 名（例: 09_goldilocks_phase12）")
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML 設定ファイルのパス。既存 run の config/train_config_resume.yaml を指定すると各設定を自動引き継ぎ",
    )
    p.add_argument(
        "--version-name",
        default="v06",
        help="runs/survivors/<version>/train/<run> (default: v06)",
    )
    p.add_argument(
        "--start-phase",
        type=int,
        default=None,
        help="開始フェーズ index（省略時は Phase 0 から）",
    )
    p.add_argument("--n-envs", type=int, default=1, help="並列環境数 (default: 1)")
    p.add_argument(
        "--port",
        type=int,
        default=8767,
        help="UE5 接続ポート（ベースポート, default: 8767）",
    )
    p.add_argument(
        "--frame-skip",
        type=int,
        default=4,
        help="訓練時の frame-skip と合わせる (default: 4)",
    )
    p.add_argument(
        "--curriculum-threshold",
        type=float,
        default=2.0,
        help="threshold_mult。訓練時の --curriculum-threshold と合わせること (default: 2.0)",
    )
    p.add_argument(
        "--curriculum-window",
        type=int,
        default=20,
        help="昇格判定ウィンドウ幅 (default: 20)",
    )
    p.add_argument(
        "--curriculum-alive-reward",
        type=float,
        default=0.001,
        help="alive_reward 係数 (default: 0.001)",
    )
    p.add_argument(
        "--advance-patience-steps",
        type=int,
        default=500_000,
        help="直近の phase advance から何 step 昇格なければ終了 (default: 500000)",
    )
    p.add_argument(
        "--wandb-project",
        default="rl-balance",
        help="W&B プロジェクト名 (default: rl-balance)",
    )
    p.add_argument(
        "--wandb-run-name",
        default=None,
        help="W&B run 名（省略時: ctest_{run}）",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="UE5 なしのスタブ環境で動作確認",
    )

    if pre_args.config:
        from common.config import load_yaml_config, apply_yaml_defaults

        apply_yaml_defaults(p, load_yaml_config(pre_args.config))

    return p.parse_args()


def _load_run(run_dir: Path) -> tuple[PPO, Path, dict]:
    """train_status.json を読んで最新モデルと VecNormalize パスを返す。"""
    status_path = run_dir / "log" / "train_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8", errors="replace"))

    def _resolve(rel: str) -> Path:
        # Windows バックスラッシュ対応
        return run_dir / Path(rel.replace("\\", "/"))

    model_path = _resolve(status["latest_model_path"])
    vecnorm_path = _resolve(status["latest_vecnormalize_path"])

    model = PPO.load(model_path, device="cuda" if torch.cuda.is_available() else "cpu")
    return model, vecnorm_path, status


def _make_env(port: int, frame_skip: int, dry_run: bool):
    """(raw_env, Monitor(raw_env)) を返す。raw_env は CurriculumCallback.set_params 用、Monitor は SB3 統計収集用。"""
    if dry_run:
        from games.survivors.survivors_env_stub import SurvivorsEnvStub

        raw_env = SurvivorsEnvStub()
    else:
        from games.survivors.survivors_env import SurvivorsUE5Env

        raw_env = SurvivorsUE5Env(port=port, frame_skip=frame_skip)
    return raw_env, Monitor(raw_env)


def main() -> None:
    args = parse_args()

    # run ディレクトリを解決（_TRAINING_ROOT 基準の絶対パス）
    run_dir = _TRAINING_ROOT / "runs" / "survivors" / args.version_name / "train" / args.run

    if not run_dir.exists():
        print(f"[ERROR] run_dir が見つかりません: {run_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] run_dir: {run_dir}")
    model, vecnorm_path, status = _load_run(run_dir)
    print(f"[INFO] model loaded from: {status.get('latest_model_path', '?')}")

    n_envs = args.n_envs
    frame_skip = args.frame_skip
    curriculum_threshold = args.curriculum_threshold
    curriculum_window = args.curriculum_window
    curriculum_alive_reward = args.curriculum_alive_reward

    # 環境作成
    raw_envs: list = []
    monitored_envs: list = []
    for i in range(n_envs):
        raw_env, mon_env = _make_env(args.port + i, frame_skip, args.dry_run)
        raw_envs.append(raw_env)
        monitored_envs.append(mon_env)

    vec_env = DummyVecEnv([lambda e=e: e for e in monitored_envs])
    vec_env = VecNormalize.load(vecnorm_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    # W&B 初期化（CurriculumCallback._on_training_start() より前に行う）
    wandb_run_name = args.wandb_run_name or f"ctest_{args.run}"
    wandb.init(
        project=args.wandb_project,
        name=wandb_run_name,
        config={
            "source_run": args.run,
            "start_phase": args.start_phase,
            "n_envs": n_envs,
            "frame_skip": frame_skip,
            "curriculum_threshold": curriculum_threshold,
            "curriculum_window": curriculum_window,
            "advance_patience_steps": args.advance_patience_steps,
        },
    )

    # CurriculumCallback 初期化
    from games.survivors.survivors_curriculum import CurriculumCallback, PHASES

    curriculum_cb = CurriculumCallback(
        raw_env=raw_envs[0],
        frame_skip=frame_skip,
        window=curriculum_window,
        threshold_mult=curriculum_threshold,
        alive_reward=curriculum_alive_reward,
        status_path=None,  # テスト中はファイル書き出しなし
    )

    if args.start_phase is not None:
        curriculum_cb.import_state(
            {
                "phase_idx": args.start_phase,
                "phase_name": PHASES[args.start_phase].name,
            }
        )

    mock_model = _MockModel(start_timesteps=0)
    curriculum_cb.model = mock_model
    curriculum_cb.training_env = vec_env
    curriculum_cb._on_training_start()

    # 評価ループ
    obs = vec_env.reset()
    last_advance_step = 0
    total_steps = 0
    prev_phase_idx = curriculum_cb._phase_idx

    print(
        f"[START] 開始フェーズ: {PHASES[curriculum_cb._phase_idx].name}"
        f" (idx={curriculum_cb._phase_idx})"
    )
    print(f"        advance_patience_steps={args.advance_patience_steps:,}")

    try:
        while True:
            # 終了判定 1: 最終フェーズに到達（ループ先頭でチェック）
            if curriculum_cb.is_final_phase:
                print(
                    f"[DONE] 最終フェーズ（{PHASES[curriculum_cb._phase_idx].name}）に到達 → 終了"
                )
                break

            # 終了判定 2: advance_patience_steps を超過
            if total_steps > 0 and (total_steps - last_advance_step) > args.advance_patience_steps:
                print(
                    f"[DONE] 直近の昇格から {args.advance_patience_steps:,} step 経過しても昇格なし → 終了"
                )
                break

            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = vec_env.step(action)

            total_steps += n_envs
            mock_model.num_timesteps = total_steps

            curriculum_cb.locals = {"infos": infos, "dones": dones}
            curriculum_cb._on_step()

            # phase advance/rollback 検出 → patience タイマーリセット
            if curriculum_cb._phase_idx != prev_phase_idx:
                last_advance_step = total_steps
                prev_phase_idx = curriculum_cb._phase_idx

            if any(dones):
                metrics = curriculum_cb.get_wandb_progress_metrics()
                wandb.log({**metrics, "global_step": total_steps}, step=total_steps)

        # 終了サマリー
        print("=== カリキュラムテスト完了 ===")
        print(f"  総ステップ数  : {total_steps:,}")
        print(
            f"  最終フェーズ  : {PHASES[curriculum_cb._phase_idx].name}"
            f" (idx={curriculum_cb._phase_idx})"
        )
        print(f"  Phase Events  : {len(curriculum_cb._phase_events)} 件")
        for ev in curriculum_cb._phase_events:
            print(
                f"    {ev['timestep']:>10,}: {ev['from_phase_name']} → {ev['to_phase_name']}"
                f" [{ev['event']}] score={ev['active_score_mean']:.1f}"
            )
    finally:
        vec_env.close()
        wandb.finish()


if __name__ == "__main__":
    main()
