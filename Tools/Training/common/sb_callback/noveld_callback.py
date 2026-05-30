"""NovelD 内発的動機コールバック（ゲーム非依存版）。

RND ネットワークはゲーム固有ロジックを持たないため、
common/sb_callback/ に配置して任意のゲームで再利用できるようにする。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback

class _RNDModule(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()
        self.target = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        for p_ in self.target.parameters():
            p_.requires_grad = False
        self.predictor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def novelty(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            y_t = self.target(x)
        y_p = self.predictor(x)
        return ((y_p - y_t) ** 2).mean(dim=-1)

    def predictor_loss(self, x: torch.Tensor) -> torch.Tensor:
        return self.novelty(x).mean()

class NovelDCallback(BaseCallback):
    """NovelD 内発的動機コールバック（ゲーム非依存）。

    Args:
        beta:           intrinsic reward のスケール係数（default: 0.3）
        alpha:          NovelD 差分係数（default: 0.5）
        hidden_dim:     RND ネットの隠れ層次元（default: 128）
        lr:             predictor の学習率（default: 1e-4）
        wandb_logger:   WandbLogger インスタンス（None の場合は W&B ログを行わない）
        verbose:        1 で rollout ごとにコンソール出力
    """

    def __init__(self, beta: float = 0.3, alpha: float = 0.5,
                 hidden_dim: int = 128, lr: float = 1e-4,
                 wandb_logger=None, verbose: int = 0):
        super().__init__(verbose)
        self.beta = beta
        self.alpha = alpha
        self.hidden_dim = hidden_dim
        self.lr = lr
        self._wandb_logger = wandb_logger
        if wandb_logger:
            wandb_logger.add_metric_prefix("expl/")
        self._rnd: _RNDModule | None = None
        self._optimizer = None
        self._prev_novelty: np.ndarray | None = None
        self._obs_buffer: list[np.ndarray] = []
        self._int_reward_buffer: list[np.ndarray] = []
        self._nov_mean: float = 0.0
        self._nov_m2: float = 0.0
        self._nov_count: int = 0
        self._warmup_steps: int = 0
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
        self._warmup_steps = self.model.n_steps * n_envs
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
        for nov in novelty:
            self._nov_count += 1
            delta = float(nov) - self._nov_mean
            self._nov_mean += delta / self._nov_count
            self._nov_m2 += delta * (float(nov) - self._nov_mean)
        nov_std = float(np.sqrt(max(self._nov_m2 / max(self._nov_count - 1, 1), 1e-8)))
        nov_norm = (novelty - self._nov_mean) / (nov_std + 1e-8)
        r_int = np.maximum(0.0, nov_norm - self.alpha * self._prev_novelty)
        self._prev_novelty = np.where(dones, 0.0, nov_norm).astype(np.float32)
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
        if self._wandb_logger:
            self._wandb_logger.log(stats, step=self.num_timesteps)
        if self.verbose >= 1:
            r_int_mean = stats["expl/noveld_r_int_mean"]
            pred_loss = stats["expl/predictor_loss"]
            print(f"[NovelD] r_int_mean={r_int_mean:.4f} predictor_loss={pred_loss:.4f}")
        self._obs_buffer.clear()
        self._int_reward_buffer.clear()

    def save_to_file(self, path: Path) -> None:
        if self._rnd is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        obs_dim = self._rnd.target[0].in_features
        payload = {
            "meta": {
                "obs_dim": obs_dim, "hidden_dim": self.hidden_dim,
                "beta": self.beta, "alpha": self.alpha, "lr": self.lr,
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
        if not path.exists():
            print(f"[NovelD] WARN: state ファイルが見つかりません: {path} → 初期状態で起動します。")
            return
        self._pending_import_path = path
        print(f"[NovelD] resume state を pending に登録: {path}")

    def _restore_state(self, path: Path, obs_dim: int) -> None:
        payload = torch.load(path, map_location=self.model.device, weights_only=True)
        meta = payload["meta"]
        if meta["obs_dim"] != obs_dim:
            saved_od=meta['obs_dim'];raise ValueError(f'[NovelD] obs_dim 不一致: saved={saved_od}, current={obs_dim}')
        if meta["hidden_dim"] != self.hidden_dim:
            saved_hd=meta['hidden_dim'];raise ValueError(f'[NovelD] hidden_dim 不一致: saved={saved_hd}, current={self.hidden_dim}')
        for key in ("beta", "alpha", "lr"):
            saved_val = meta[key]
            current_val = getattr(self, key)
            if saved_val != current_val:
                print(f"[NovelD] WARN: {key} 不一致 saved={saved_val}, current={current_val} （現在の設定値を使用）")
        self._rnd.target.load_state_dict(payload["target"])
        self._rnd.predictor.load_state_dict(payload["predictor"])
        self._optimizer.load_state_dict(payload["optimizer"])
        self._nov_mean  = float(payload["nov_mean"])
        self._nov_m2    = float(payload["nov_m2"])
        self._nov_count = int(payload["nov_count"])
        self._warmup_steps = 0
        print(f"[NovelD] resume から predictor・target・optimizer・running stats を復元しました。（obs_dim={obs_dim}, nov_count={self._nov_count}）")
