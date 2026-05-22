"""NovelD 内発的動機コールバック（Survivors 専用）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


class _RNDModule(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()
        self.target = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        for p in self.target.parameters():
            p.requires_grad = False
        self.predictor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def novelty(self, x: torch.Tensor) -> torch.Tensor:
        """novelty スコアを返す。y_t は no_grad、y_p は grad あり。shape: (batch,)"""
        with torch.no_grad():
            y_t = self.target(x)
        y_p = self.predictor(x)
        return ((y_p - y_t) ** 2).mean(dim=-1)

    def predictor_loss(self, x: torch.Tensor) -> torch.Tensor:
        return self.novelty(x).mean()


class NovelDCallback(BaseCallback):
    """NovelD 内発的動機コールバック（Survivors 専用）。

    rollout 中に VecNormalize 済み obs から RND novelty を計算し、
    NovelD（差分）形式の intrinsic reward を外部報酬に加算する。
    predictor は rollout 終了後に batch 更新する。

    Args:
        beta:       intrinsic reward のスケール係数（default: 0.3）
        alpha:      NovelD 差分係数（default: 0.5）
        hidden_dim: RND ネットの隠れ層次元（default: 128）
        lr:         predictor の学習率（default: 1e-4）
        verbose:    1 で rollout ごとにコンソール出力
    """

    def __init__(self, beta: float = 0.3, alpha: float = 0.5,
                 hidden_dim: int = 128, lr: float = 1e-4, verbose: int = 0):
        super().__init__(verbose)
        self.beta = beta
        self.alpha = alpha
        self.hidden_dim = hidden_dim
        self.lr = lr
        self._rnd: _RNDModule | None = None
        self._optimizer = None
        self._prev_novelty: np.ndarray | None = None
        self._obs_buffer: list[np.ndarray] = []
        self._int_reward_buffer: list[np.ndarray] = []
        # Welford online variance（バッチ対応版）
        self._nov_mean: float = 0.0
        self._nov_m2: float = 0.0
        self._nov_count: int = 0
        self._warmup_steps: int = 0
        # resume 時に _on_training_start で適用する pending state ファイルパス
        self._pending_import_path: Path | None = None

    def _on_training_start(self) -> None:
        obs_space = self.training_env.observation_space
        obs_dim = int(np.prod(obs_space.shape))
        device = self.model.device
        self._rnd = _RNDModule(obs_dim, self.hidden_dim).to(device)
        self._optimizer = torch.optim.Adam(
            self._rnd.predictor.parameters(), lr=self.lr
        )
        n_envs = self.training_env.num_envs
        self._prev_novelty = np.zeros(n_envs, dtype=np.float32)
        # warmup: 最初の1ロールアウト分は r_int=0 にして running stats を安定させる
        self._warmup_steps = self.model.n_steps * n_envs

        # resume からの状態復元（_rnd / _optimizer 初期化後に適用）
        if self._pending_import_path is not None:
            self._restore_state(self._pending_import_path, obs_dim)
            self._pending_import_path = None

        print(
            f"[NovelD] 初期化完了: obs_dim={obs_dim}, beta={self.beta}, "
            f"alpha={self.alpha}, warmup={self._warmup_steps} steps"
        )

    def _on_step(self) -> bool:
        new_obs = self.locals["new_obs"]
        dones   = self.locals["dones"]

        with torch.no_grad():
            obs_t = torch.as_tensor(new_obs, dtype=torch.float32, device=self.model.device)
            novelty = self._rnd.novelty(obs_t).cpu().numpy()

        # Welford running stats 更新（n_envs 分を逐次更新）
        for nov in novelty:
            self._nov_count += 1
            delta = float(nov) - self._nov_mean
            self._nov_mean += delta / self._nov_count
            self._nov_m2 += delta * (float(nov) - self._nov_mean)

        nov_std = float(np.sqrt(max(self._nov_m2 / max(self._nov_count - 1, 1), 1e-8)))
        nov_norm = (novelty - self._nov_mean) / (nov_std + 1e-8)

        r_int = np.maximum(0.0, nov_norm - self.alpha * self._prev_novelty)
        # resume 後の最初の step は _prev_novelty=0（エピソード境界相当）として扱う
        self._prev_novelty = np.where(dones, 0.0, nov_norm).astype(np.float32)

        # warmup 中は r_int を加算しない
        if self.num_timesteps > self._warmup_steps:
            self.locals["rewards"] = self.locals["rewards"] + self.beta * r_int.astype(
                self.locals["rewards"].dtype
            )

        self._obs_buffer.append(new_obs.copy())
        self._int_reward_buffer.append(r_int.copy())
        return True

    def _on_rollout_end(self) -> None:
        if not self._obs_buffer:
            return

        all_obs = np.concatenate(self._obs_buffer, axis=0)
        obs_t = torch.as_tensor(all_obs, dtype=torch.float32, device=self.model.device)
        self._optimizer.zero_grad()
        loss = self._rnd.predictor_loss(obs_t)
        loss.backward()
        self._optimizer.step()

        all_r_int = np.concatenate(self._int_reward_buffer, axis=0)
        stats = {
            "expl/noveld_r_int_mean": float(np.mean(all_r_int)),
            "expl/noveld_r_int_max":  float(np.max(all_r_int)),
            "expl/predictor_loss":    float(loss.item()),
            "expl/novelty_mean":      self._nov_mean,
        }
        for k, v in stats.items():
            self.logger.record(k, v)
        if _WANDB_AVAILABLE and wandb.run:
            wandb.log({**stats, "global_step": self.num_timesteps}, step=self.num_timesteps)
        if self.verbose >= 1:
            print(
                f"[NovelD] r_int_mean={stats['expl/noveld_r_int_mean']:.4f} "
                f"predictor_loss={stats['expl/predictor_loss']:.4f}"
            )
        self._obs_buffer.clear()
        self._int_reward_buffer.clear()

    # ── 状態の保存・復元 ─────────────────────────────────────────────────────

    def save_to_file(self, path: Path) -> None:
        """target / predictor / optimizer state と Welford stats を .pt ファイルに保存する。"""
        if self._rnd is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        obs_dim = self._rnd.target[0].in_features
        payload = {
            "meta": {
                "obs_dim": obs_dim,
                "hidden_dim": self.hidden_dim,
                "beta": self.beta,
                "alpha": self.alpha,
                "lr": self.lr,
            },
            "target":    self._rnd.target.state_dict(),
            "predictor": self._rnd.predictor.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "nov_mean":  self._nov_mean,
            "nov_m2":    self._nov_m2,
            "nov_count": self._nov_count,
        }
        torch.save(payload, path)

    def load_from_file(self, path: Path) -> None:
        """resume 時に呼び出す。_on_training_start まで pending に保持する。"""
        if not path.exists():
            print(f"[NovelD] WARN: state ファイルが見つかりません: {path} → 初期状態で起動します。")
            return
        self._pending_import_path = path
        print(f"[NovelD] resume state を pending に登録: {path}")

    def _restore_state(self, path: Path, obs_dim: int) -> None:
        """_on_training_start 内で _rnd / _optimizer 初期化後に呼ばれる。"""
        payload = torch.load(path, map_location=self.model.device, weights_only=True)
        meta = payload["meta"]

        # 構造不一致はエラー（state_dict のロード自体が失敗するため早期検出）
        if meta["obs_dim"] != obs_dim:
            raise ValueError(
                f"[NovelD] obs_dim 不一致: saved={meta['obs_dim']}, current={obs_dim}"
            )
        if meta["hidden_dim"] != self.hidden_dim:
            raise ValueError(
                f"[NovelD] hidden_dim 不一致: saved={meta['hidden_dim']}, current={self.hidden_dim}"
            )

        # スケール変更は警告のみ（意図的な変更を許容）
        for key in ("beta", "alpha", "lr"):
            saved_val = meta[key]
            current_val = getattr(self, key)
            if saved_val != current_val:
                print(
                    f"[NovelD] WARN: {key} 不一致 saved={saved_val}, current={current_val} "
                    f"（現在の設定値を使用）"
                )

        self._rnd.target.load_state_dict(payload["target"])
        self._rnd.predictor.load_state_dict(payload["predictor"])
        self._optimizer.load_state_dict(payload["optimizer"])
        self._nov_mean  = float(payload["nov_mean"])
        self._nov_m2    = float(payload["nov_m2"])
        self._nov_count = int(payload["nov_count"])

        # resume 後は warmup 不要（predictor は継続学習済み）
        self._warmup_steps = 0

        print(
            f"[NovelD] resume から predictor・target・optimizer・running stats を復元しました。"
            f"（obs_dim={obs_dim}, nov_count={self._nov_count}）"
        )
