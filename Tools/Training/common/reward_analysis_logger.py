"""RewardAnalysisLogger: shaped_reward の分布・時系列・obs 相関を収集して Markdown レポートを生成する。"""
from __future__ import annotations
import json
import datetime
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


SURVIVORS_OBS_SCHEMA: dict[int, str] = {
    0: "player_pos_x", 1: "player_pos_y",
    2: "player_vel_x", 3: "player_vel_y",
    12: "player_hp_norm",
    13: "shield_active",
    53: "enemy_count_norm",
    54: "elapsed_time_norm",
    55: "xp_progress",
    56: "player_level_norm",
}


class RewardAnalysisLogger:
    _MAX_STEPS = 500_000
    _OBS_SAMPLE_INTERVAL = 10

    def __init__(
        self,
        obs_schema: dict[int, str] | None = None,
        game: str = "",
        version: str = "",
    ):
        self.obs_schema = obs_schema or {}
        self.game = game
        self.version = version

        self._shaped_buf: list[float] = []
        self._base_buf: list[float] = []
        self._obs_buf: list[np.ndarray] = []
        self._obs_shaped_buf: list[float] = []

        self._ep_shaped: list[float] = []
        self._ep_base: list[float] = []
        self.ep_shaped_totals: list[float] = []
        self.ep_base_totals: list[float] = []
        self.ep_lengths: list[int] = []
        self._sampled_ep_seqs: list[list[float]] = []

        self._step_count = 0
        self._saved = False

    def on_step(self, shaped: float, base: float, obs: np.ndarray | None = None) -> None:
        if len(self._shaped_buf) < self._MAX_STEPS:
            self._shaped_buf.append(shaped)
            self._base_buf.append(base)
        self._ep_shaped.append(shaped)
        self._ep_base.append(base)
        self._step_count += 1
        if obs is not None and self._step_count % self._OBS_SAMPLE_INTERVAL == 0:
            self._obs_buf.append(obs.copy())
            self._obs_shaped_buf.append(shaped)

    def on_episode_end(self) -> None:
        self.ep_shaped_totals.append(sum(self._ep_shaped))
        self.ep_base_totals.append(sum(self._ep_base))
        self.ep_lengths.append(len(self._ep_shaped))
        if len(self._sampled_ep_seqs) < 50:
            self._sampled_ep_seqs.append(self._ep_shaped[:])
        self._ep_shaped = []
        self._ep_base = []

    def to_markdown(self, metadata: dict | None = None) -> str:
        shaped = np.array(self._shaped_buf) if self._shaped_buf else np.array([])
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        n_ep = len(self.ep_shaped_totals)
        n_steps = len(shaped)

        lines: list[str] = []
        lines.append("# 報酬解析レポート")
        meta_str = f"生成: {now}"
        if self.game:
            meta_str += f" | ゲーム: {self.game}"
            if self.version:
                meta_str += f" {self.version}"
        if metadata:
            for k, v in metadata.items():
                meta_str += f" | {k}: {v}"
        meta_str += f" | ステップ数: {n_steps:,} | エピソード数: {n_ep}"
        lines.append(meta_str)
        lines.append("")

        lines.append("## shaped_reward 分布")
        if n_steps > 0:
            zero_rate = float(np.mean(shaped == 0.0))
            pos_mask = shaped > 0
            neg_mask = shaped < 0
            pos_rate = float(np.mean(pos_mask))
            neg_rate = float(np.mean(neg_mask))
            lines.append("| 統計 | 値 |")
            lines.append("|------|-----|")
            lines.append(f"| ゼロ率 | {zero_rate:.1%} |")
            lines.append(f"| 正の報酬率 | {pos_rate:.1%} |")
            lines.append(f"| 負の報酬率 | {neg_rate:.1%} |")
            if pos_rate > 0:
                lines.append(f"| 正の報酬の平均 | {float(np.mean(shaped[pos_mask])):.4f} |")
            if neg_rate > 0:
                lines.append(f"| 負の報酬の平均 | {float(np.mean(shaped[neg_mask])):.4f} |")
            lines.append(f"| 全体平均 | {float(np.mean(shaped)):.4f} |")
            lines.append(f"| 標準偏差 | {float(np.std(shaped)):.4f} |")
            for p in [50, 75, 90, 95, 99]:
                lines.append(f"| p{p} | {float(np.percentile(shaped, p)):.4f} |")
            lines.append(f"| min | {float(np.min(shaped)):.4f} |")
            lines.append(f"| max | {float(np.max(shaped)):.4f} |")
        else:
            lines.append("（データなし）")
        lines.append("")

        lines.append("## エピソード内時系列（平均）")
        if self._sampled_ep_seqs:
            early_vals, mid_vals, late_vals = [], [], []
            for seq in self._sampled_ep_seqs:
                n = len(seq)
                if n < 3:
                    continue
                t1, t2 = n // 3, 2 * n // 3
                early_vals.append(float(np.mean(seq[:t1])))
                mid_vals.append(float(np.mean(seq[t1:t2])))
                late_vals.append(float(np.mean(seq[t2:])))
            if early_vals:
                lines.append("| 区間 | shaped_mean |")
                lines.append("|------|------------|")
                lines.append(f"| 前半 (0-33%) | {float(np.mean(early_vals)):.4f} |")
                lines.append(f"| 中盤 (33-66%) | {float(np.mean(mid_vals)):.4f} |")
                lines.append(f"| 後半 (66-100%) | {float(np.mean(late_vals)):.4f} |")
                early_m = float(np.mean(early_vals))
                late_m = float(np.mean(late_vals))
                if abs(late_m) > 1e-6 and abs(early_m / (late_m + 1e-9)) < 0.1:
                    lines.append("")
                    lines.append("→ 報酬がエピソード後半に偏っています。序盤の学習シグナルが不足している可能性があります。")
                elif abs(early_m) > abs(late_m) * 1.5:
                    lines.append("")
                    lines.append("→ 報酬がエピソード序盤に偏っています。エージェントが序盤に得やすい行動に特化している可能性があります。")
        else:
            lines.append("（データなし）")
        lines.append("")

        lines.append("## base_reward との相関（エピソード単位）")
        if len(self.ep_shaped_totals) >= 2:
            corr_val = float(np.corrcoef(self.ep_base_totals, self.ep_shaped_totals)[0, 1])
            if not np.isnan(corr_val):
                lines.append(f"r = {corr_val:.3f}")
                if corr_val < -0.2:
                    lines.append("")
                    lines.append("⚠️ 負の相関: shaped_reward を稼ぐほどタスク達成が悪化しています（報酬ハッキングの兆候）。")
                elif corr_val < 0.1:
                    lines.append("")
                    lines.append("→ shaped_reward とタスク達成がほぼ無相関。シェーピングがタスク改善に寄与していない可能性があります。")
                else:
                    lines.append("")
                    lines.append("→ 正の相関: shaped_reward はタスク達成と概ね整合しています。")
        else:
            lines.append("（エピソード数不足）")
        lines.append("")

        lines.append("## obs 次元との相関（|r| 上位10）")
        if self._obs_buf and len(self._obs_buf) >= 10:
            obs_arr = np.stack(self._obs_buf)
            shaped_obs = np.array(self._obs_shaped_buf)
            if np.std(shaped_obs) > 1e-9:
                n_dims = obs_arr.shape[1]
                s_z = (shaped_obs - np.mean(shaped_obs)) / np.std(shaped_obs)
                obs_mean = np.mean(obs_arr, axis=0, keepdims=True)
                obs_std = np.std(obs_arr, axis=0)
                valid = obs_std > 1e-9
                corrs = np.zeros(n_dims)
                obs_z = obs_arr - obs_mean
                corrs[valid] = np.mean(obs_z[:, valid] / obs_std[valid] * s_z[:, None], axis=0)
                top_idx = np.argsort(np.abs(corrs))[::-1][:10]
                lines.append("| obs インデックス | 名前 | r |")
                lines.append("|---|---|---|")
                for idx in top_idx:
                    name = self.obs_schema.get(int(idx), f"obs[{idx}]")
                    lines.append(f"| [{idx}] | {name} | {corrs[idx]:.3f} |")
                et_idx = next((k for k, v in self.obs_schema.items() if v == "elapsed_time_norm"), None)
                if et_idx is not None and abs(corrs[et_idx]) > 0.5:
                    lines.append("")
                    lines.append(f"⚠️ elapsed_time_norm (obs[{et_idx}]) との相関が {corrs[et_idx]:.3f} と強い。")
                    lines.append("   生存ボーナス項が他の項を圧倒している可能性があります。各項のスケールを確認してください。")
            else:
                lines.append("（shaped_reward の分散が小さすぎて相関計算できません）")
        else:
            lines.append("（obs サンプル数不足）")
        lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        shaped = np.array(self._shaped_buf) if self._shaped_buf else np.array([])
        n_steps = len(shaped)
        result: dict = {"n_steps": n_steps, "n_episodes": len(self.ep_shaped_totals)}
        if n_steps > 0:
            result["distribution"] = {
                "zero_rate": float(np.mean(shaped == 0.0)),
                "pos_rate": float(np.mean(shaped > 0)),
                "neg_rate": float(np.mean(shaped < 0)),
                "mean": float(np.mean(shaped)),
                "std": float(np.std(shaped)),
                "p50": float(np.percentile(shaped, 50)),
                "p75": float(np.percentile(shaped, 75)),
                "p90": float(np.percentile(shaped, 90)),
                "p99": float(np.percentile(shaped, 99)),
                "min": float(np.min(shaped)),
                "max": float(np.max(shaped)),
            }
        if len(self.ep_shaped_totals) >= 2:
            corr_val = float(np.corrcoef(self.ep_base_totals, self.ep_shaped_totals)[0, 1])
            result["base_shaped_corr"] = corr_val if not np.isnan(corr_val) else None
        return result

    def save(
        self,
        log_dir: Path,
        prefix: str = "reward_analysis",
        metadata: dict | None = None,
    ) -> Path:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        md_path = log_dir / f"{prefix}.md"
        json_path = log_dir / f"{prefix}.json"
        md_path.write_text(self.to_markdown(metadata), encoding="utf-8")
        json_path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._saved = True
        return md_path


class RewardAnalysisCallback(BaseCallback):
    def __init__(self, logger: RewardAnalysisLogger):
        super().__init__(verbose=0)
        self._rl = logger

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        shaped = float(info.get("shaped_reward", 0.0))
        base = float(info.get("base_reward", 0.0))
        new_obs = self.locals.get("new_obs")
        obs_vec = new_obs[0] if new_obs is not None else None
        self._rl.on_step(shaped, base, obs_vec)
        if self.locals["dones"][0]:
            self._rl.on_episode_end()
        return True
