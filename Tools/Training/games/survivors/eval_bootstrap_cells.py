"""Bootstrap cell を deterministic policy で固定評価する standalone スクリプト。

KingBible solo phase2 / Garlic maintenance phase2 などの bootstrap cell を
argmax(deterministic) policy で固定評価し、訓練中の stochastic episode p10 ではなく
「実際にデプロイする policy の実力」を JSON artifact として出力する。

訓練ループには組み込まない。UE5 HTTP env に接続して eval を実行する。

使用例:
    python -m games.survivors.eval_bootstrap_cells \
        --run-dir runs/survivors/v12/train/run07 \
        --episodes 40 \
        --cells king_bible:solo_bootstrap:2 garlic:maintenance:2 \
        --deterministic --port 8767

出力（既定）: <run-dir>/eval/bootstrap_deterministic_<timestamp>.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from games.survivors.bootstrap_eval import (
    build_eval_params,
    parse_cell_spec,
    summarize_eval_results,
)


def _strip_zip(path: Path) -> Path:
    """SB3 の load は拡張子なしパスを期待するため .zip を除去する。"""
    return path.with_suffix("") if path.suffix == ".zip" else path


def resolve_model_path(run_dir: Path, model_path: str | None) -> Path:
    """run-dir から eval 対象モデルの zip パスを解決する。

    優先順位: 明示 --model-path > <run>/result/best_model.zip >
              <run>/result/model.zip > work/model_steps/ 最大ステップ。
    """
    if model_path is not None:
        p = Path(model_path)
        if not p.exists() and p.with_suffix(".zip").exists():
            p = p.with_suffix(".zip")
        if not p.exists():
            raise FileNotFoundError(f"model_path が見つかりません: {model_path}")
        return p

    candidates = [
        run_dir / "result" / "best_model.zip",
        run_dir / "result" / "model.zip",
        run_dir / "best_model.zip",
        run_dir / "model.zip",
    ]
    for c in candidates:
        if c.exists():
            return c

    # work/model_steps/ 最大ステップへフォールバック
    steps_dir = run_dir / "work" / "model_steps"
    if steps_dir.exists():
        step_models = sorted(
            steps_dir.glob("model_*_steps.zip"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        if step_models:
            return step_models[-1]

    raise FileNotFoundError(
        f"eval 対象モデルが見つかりません: {run_dir}\n"
        f"探索した候補: {[str(c) for c in candidates]}"
    )


def resolve_vecnormalize_path(run_dir: Path, model_path: Path) -> Path | None:
    """vecnormalize.pkl のパスを解決する（存在しなければ None）。"""
    candidates = [
        model_path.parent / "vecnormalize.pkl",
        run_dir / "result" / "vecnormalize.pkl",
        run_dir / "vecnormalize.pkl",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _is_recurrent_zip(model_zip: Path) -> bool:
    """保存済み zip の policy_class から RecurrentPPO かどうかを判定する。"""
    try:
        from stable_baselines3.common.save_util import load_from_zip_file
        data, _params, _pytorch = load_from_zip_file(str(_strip_zip(model_zip)), device="cpu")
        policy_class = data.get("policy_class")
        name = getattr(policy_class, "__name__", str(policy_class))
        return "Recurrent" in name or "Lstm" in name or "LSTM" in name
    except Exception:
        return False


def load_model(model_zip: Path, env, device: str, recurrent: bool | None):
    """PPO / RecurrentPPO を自動判別してロードする。"""
    from stable_baselines3 import PPO

    if recurrent is None:
        recurrent = _is_recurrent_zip(model_zip)

    if recurrent:
        from sb3_contrib import RecurrentPPO
        return RecurrentPPO.load(str(_strip_zip(model_zip)), env=env, device=device)
    return PPO.load(str(_strip_zip(model_zip)), env=env, device=device)


def build_eval_env(host: str, port: int, frame_skip: int, vecnorm_path: Path | None):
    """UE5 HTTP env（DummyVecEnv + optional VecNormalize eval mode）を構築する。"""
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize

    from games.survivors.survivors_env import SurvivorsEnv

    env = make_vec_env(
        lambda: SurvivorsEnv(host=host, port=port, frame_skip=frame_skip),
        n_envs=1,
    )
    if vecnorm_path is not None:
        env = VecNormalize.load(str(vecnorm_path), env)
        env.training = False
        env.norm_reward = False
    return env


def evaluate_cells(
    *,
    run_dir: Path,
    model_path: str | None,
    episodes: int,
    cells: list[str],
    deterministic: bool,
    seed: int | None,
    host: str,
    port: int,
    frame_skip: int,
    alive_reward: float,
    device: str,
    recurrent: bool | None,
) -> dict:
    """全 cell を eval して artifact dict を返す。"""
    from games.survivors.survivors_eval_callback import run_survivors_eval_episodes

    model_zip = resolve_model_path(run_dir, model_path)
    vecnorm_path = resolve_vecnormalize_path(run_dir, model_zip)

    # cell spec を先に parse して不正な spec で env を立てる前に落とす
    parsed_cells = [parse_cell_spec(spec) for spec in cells]

    env = build_eval_env(host, port, frame_skip, vecnorm_path)
    if seed is not None:
        env.seed(seed)

    model = load_model(model_zip, env, device, recurrent)
    global_timestep = int(getattr(model, "num_timesteps", 0) or 0)

    cell_summaries: list[dict] = []
    try:
        for parsed in parsed_cells:
            ue_params = build_eval_params(parsed)
            # ParamApplier と二重変換しないよう UE key 済み full params を直接送る
            env.env_method("set_params", **ue_params)

            episode_results, _metrics, _last_obs = run_survivors_eval_episodes(
                model=model,
                env=env,
                n_eval_episodes=episodes,
                frame_skip=frame_skip,
                alive_reward=alive_reward,
                deterministic=deterministic,
            )
            summary = summarize_eval_results(
                parsed.spec,
                episode_results,
                deterministic=deterministic,
                model_path=str(model_zip),
                global_timestep=global_timestep,
            )
            cell_summaries.append(summary)
            print(
                f"[bootstrap_eval] {parsed.spec}: "
                f"p10={summary['active_score_p10']:.1f} "
                f"p50={summary['active_score_p50']:.1f} "
                f"mean={summary['active_score_mean']:.1f} "
                f"ep_len_mean={summary['episode_length_mean']:.0f} "
                f"short_rate={summary['short_episode_rate']:.3f}"
            )
    finally:
        env.close()

    return {
        "run_dir": str(run_dir),
        "model_path": str(model_zip),
        "vecnormalize_path": str(vecnorm_path) if vecnorm_path else None,
        "global_timestep": global_timestep,
        "episodes": episodes,
        "deterministic": deterministic,
        "seed": seed,
        "port": port,
        "cells": cell_summaries,
    }


def default_output_path(run_dir: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return run_dir / "eval" / f"bootstrap_deterministic_{ts}.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="run ディレクトリ")
    parser.add_argument("--model-path", default=None, help="明示的なモデル zip パス（省略時は run-dir から解決）")
    parser.add_argument("--episodes", type=int, default=40, help="cell ごとの eval episode 数")
    parser.add_argument(
        "--cells", nargs="+", required=True,
        help="eval する cell spec（例: king_bible:solo_bootstrap:2 garlic:maintenance:2）",
    )
    parser.add_argument("--deterministic", action="store_true", default=True, help="deterministic policy を使う（既定 True）")
    parser.add_argument("--stochastic", dest="deterministic", action="store_false", help="stochastic policy を使う")
    parser.add_argument("--seed", type=int, default=None, help="eval env の seed")
    parser.add_argument("--host", default="127.0.0.1", help="UE5 HTTP host")
    parser.add_argument("--port", type=int, default=8767, help="UE5 HTTP port")
    parser.add_argument("--frame-skip", type=int, default=4, help="frame skip")
    parser.add_argument("--alive-reward", type=float, default=0.001, help="alive_reward（active_score 計算用）")
    parser.add_argument("--device", default="auto", help="torch device")
    parser.add_argument(
        "--recurrent", dest="recurrent", action="store_true", default=None,
        help="RecurrentPPO を強制。省略時は zip から自動判定",
    )
    parser.add_argument("--no-recurrent", dest="recurrent", action="store_false", help="PPO を強制")
    parser.add_argument("--output", default=None, help="出力 JSON パス（省略時は run-dir/eval/ 配下）")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    artifact = evaluate_cells(
        run_dir=run_dir,
        model_path=args.model_path,
        episodes=args.episodes,
        cells=args.cells,
        deterministic=args.deterministic,
        seed=args.seed,
        host=args.host,
        port=args.port,
        frame_skip=args.frame_skip,
        alive_reward=args.alive_reward,
        device=args.device,
        recurrent=args.recurrent,
    )

    out_path = Path(args.output) if args.output else default_output_path(run_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[bootstrap_eval] artifact を書き出しました: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
