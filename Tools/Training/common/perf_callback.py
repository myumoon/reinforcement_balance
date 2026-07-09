"""PerfLoggingCallback: 訓練スループット（FPS）を計測して W&B と JSONL に記録するコールバック。

計測タイミング:
    _on_rollout_start   [t0] ← ロールアウト開始
      ...collect (n_steps × n_envs steps)...
    _on_rollout_end     [t1] ← ロールアウト終了 / PPO 更新開始
      ...PPO training...
    _on_rollout_start   [t2] ← ロールアウト2開始 = PPO 更新終了

    rollout_time = t1 - t0
    train_time   = t2 - t1
    iter_time    = t2 - t0

    t2 のタイミング（次の _on_rollout_start）で前イテレーションのメトリクスをログ。
"""
from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Callable

try:
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError as _e:  # pragma: no cover
    raise ImportError("stable_baselines3 が必要です") from _e

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


class PerfLoggingCallback(BaseCallback):
    """ロールアウト収集フェーズと PPO 更新フェーズの時間を分けて計測するコールバック。

    W&B に記録するメトリクス:
        perf/iter_fps        : 1 イテレーション全体(収集+更新)の steps/秒（移動平均）
        perf/rollout_fps     : ロールアウト収集フェーズの steps/秒
        perf/train_fps       : PPO 更新フェーズの steps/秒
        perf/rollout_time_s  : ロールアウト収集の所要時間(秒)
        perf/train_time_s    : PPO 更新の所要時間(秒)
        perf/session_fps     : セッション開始からの累積 steps/秒（resume 時の前回分を除外）

    Args:
        log_dir:     JSONL 出力先ディレクトリ。None の場合 JSONL は書き込まない。
        window_size: iter_fps 移動平均のウィンドウサイズ（デフォルト 5）。
        verbose:     1 でコンソール出力（デフォルト 0）。
    """

    def __init__(
        self,
        log_dir: Path | None = None,
        window_size: int = 5,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self._log_dir: Path | None = Path(log_dir) if log_dir is not None else None
        self._window_size: int = max(1, window_size)

        # タイミング記録
        self._rollout_t0: float | None = None   # ロールアウト開始 t0
        self._rollout_end_t: float | None = None  # ロールアウト終了 t1

        # セッション基準値（resume 時のオフセット除外）
        self._session_start_steps: int = 0
        self._session_start_t: float = 0.0

        # 移動平均用キュー
        self._iter_fps_deque: deque[float] = deque(maxlen=self._window_size)

        # W&B ログメソッド（テストでモック差し替え可能）
        self._wandb_log: Callable | None = None

    # ------------------------------------------------------------------
    # SB3 フック
    # ------------------------------------------------------------------

    def _on_training_start(self) -> None:
        """セッション開始時の基準ステップ数・時刻を記録する。"""
        self._session_start_steps = self.model.num_timesteps
        self._session_start_t = time.perf_counter()

        # W&B ログメソッドを初期化
        if _WANDB_AVAILABLE and _wandb.run is not None:
            self._wandb_log = _make_wandb_log_fn()

        # JSONL 出力先の準備
        if self._log_dir is not None:
            self._log_dir.mkdir(parents=True, exist_ok=True)

    def _on_rollout_start(self) -> None:
        """ロールアウト開始。前イテレーションの train_time も確定する。"""
        t2 = time.perf_counter()

        if self._rollout_t0 is not None and self._rollout_end_t is not None:
            # 前イテレーション分を集計してログ
            t0 = self._rollout_t0
            t1 = self._rollout_end_t
            self._flush_iter_metrics(t0=t0, t1=t1, t2=t2)

        # 新しいイテレーションの t0 を記録
        self._rollout_t0 = t2
        self._rollout_end_t = None

    def _on_rollout_end(self) -> None:
        """ロールアウト終了（= PPO 更新開始）。t1 を記録するだけ。"""
        self._rollout_end_t = time.perf_counter()

    def _on_step(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # 内部メソッド
    # ------------------------------------------------------------------

    def _flush_iter_metrics(self, t0: float, t1: float, t2: float) -> None:
        """1 イテレーション分のメトリクスを W&B と JSONL に書き込む。"""
        rollout_time = t1 - t0
        train_time = t2 - t1
        iter_time = t2 - t0

        num_timesteps = self.model.num_timesteps
        rollout_size = self._rollout_size()

        # FPS 計算（0 除算ガード付き）
        rollout_fps = rollout_size / rollout_time if rollout_time > 0 else 0.0
        train_fps = rollout_size / train_time if train_time > 0 else 0.0
        iter_fps_raw = rollout_size / iter_time if iter_time > 0 else 0.0

        # iter_fps の移動平均
        self._iter_fps_deque.append(iter_fps_raw)
        iter_fps_avg = sum(self._iter_fps_deque) / len(self._iter_fps_deque)

        # session_fps: resume 前のステップを除外
        elapsed_session = t2 - self._session_start_t
        session_steps = num_timesteps - self._session_start_steps
        session_fps = session_steps / elapsed_session if elapsed_session > 0 else 0.0

        metrics = {
            "perf/iter_fps": iter_fps_avg,
            "perf/rollout_fps": rollout_fps,
            "perf/train_fps": train_fps,
            "perf/rollout_time_s": rollout_time,
            "perf/train_time_s": train_time,
            "perf/session_fps": session_fps,
        }

        if self.verbose >= 1:
            print(
                f"[PerfLogging] step={num_timesteps:,}  "
                f"iter_fps={iter_fps_avg:.1f}  session_fps={session_fps:.1f}"
            )

        # W&B へ書き込み
        self._write_wandb(metrics, step=num_timesteps)

        # JSONL へ書き込み
        self._write_jsonl(metrics, timestep=num_timesteps)

    def _rollout_size(self) -> int:
        """1 ロールアウトのステップ数（n_steps × n_envs）を返す。"""
        try:
            return int(self.model.n_steps) * int(self.model.n_envs)
        except AttributeError:
            return 0

    def _write_wandb(self, metrics: dict[str, float], step: int) -> None:
        """W&B へメトリクスを書き込む。wandb 未インストール・未 init 時はスキップ。"""
        # テストでモック差し替えされた場合
        if self._wandb_log is not None and callable(self._wandb_log):
            self._wandb_log(metrics, step)
            return

        if not _WANDB_AVAILABLE:
            return
        try:
            if _wandb.run is not None:
                _wandb.log({**metrics, "global_step": step})
        except Exception:
            pass  # W&B エラーで訓練を止めない

    def _write_jsonl(self, metrics: dict[str, float], timestep: int) -> None:
        """JSONL ファイルへ追記する。log_dir が None の場合はスキップ。"""
        if self._log_dir is None:
            return
        record = {"timestep": timestep, **metrics}
        jsonl_path = self._log_dir / "perf_log.jsonl"
        try:
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass  # I/O エラーで訓練を止めない


# ---------------------------------------------------------------------------
# モジュールレベルのヘルパー
# ---------------------------------------------------------------------------

def _make_wandb_log_fn():
    """wandb.log をラップした関数を返す（インポート時エラー回避）。"""
    def _log(metrics: dict[str, float], step: int) -> None:
        try:
            if _wandb.run is not None:
                _wandb.log({**metrics, "global_step": step})
        except Exception:
            pass

    return _log
