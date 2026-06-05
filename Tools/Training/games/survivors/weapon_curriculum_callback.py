"""
WeaponCurriculumCallback: weapon_phase に応じて /params を定期送信する SB3 コールバック。
train.py と eureka_loop.py の両方から使用する。
"""
from __future__ import annotations

import json
import logging
import os

from stable_baselines3.common.callbacks import BaseCallback

from games.survivors.survivors_weapon_curriculum import WEAPON_PHASES, get_params_for_phase

logger = logging.getLogger(__name__)


class WeaponCurriculumCallback(BaseCallback):
    """武器カリキュラムの遷移フェーズで weapon_weights を訓練中に定期更新するコールバック。

    weighted フェーズ（W0_to_W1 / W1_to_W2）では global_step に応じた線形補間で
    weapon_weights が変化するため、update_freq ステップごとに全 env に再送信する。
    固定フェーズ（W0〜W6）では _on_training_start のみ送信し、その後は no-op。

    フェーズ開始ステップは output_dir/weapon_curriculum_status.json に永続化される。
    --resume 時に同じ phase_key で再開すると保存済みの phase_start_step を復元するため、
    遷移フェーズの補間進捗が巻き戻らない。

    branch resume 対応:
      checkpoint_dir を指定すると、output_dir に status がない場合に checkpoint_dir の
      weapon_curriculum_status.json からも復元を試みる。これにより W0_to_W1 / W1_to_W2 の
      checkpoint から branch resume した際に phase_start_step が再初期化されるのを防ぐ。
    """

    def __init__(
        self,
        phase_key: str,
        update_freq: int = 10_000,
        output_dir: str = "",
        checkpoint_dir: str = "",
    ):
        super().__init__(verbose=0)
        self._phase_key = phase_key
        self.update_freq = max(update_freq, 1)
        self._last_update = 0
        self._is_transition = False
        self._phase_start_step: int = 0  # resume時に復元される絶対ステップ数
        self._output_dir = output_dir
        self._status_file = os.path.join(output_dir, "weapon_curriculum_status.json") if output_dir else ""
        self._checkpoint_dir = checkpoint_dir
        # EpisodeLogger: log_combination_rewards=True のフェーズでエピソードをJSONLに記録する
        _phase = WEAPON_PHASES.get(phase_key, {})
        self._episode_logger = None
        if _phase.get("log_combination_rewards") and output_dir:
            from games.survivors.survivors_episode_logger import EpisodeLogger
            _log_dir = os.path.join(output_dir, "episode_logs")
            self._episode_logger = EpisodeLogger(output_dir=_log_dir)
        # エピソードごとの報酬累積（env インデックス別）
        self._ep_total_reward: list[float] = []
        self._ep_id = 0

    def _on_training_start(self) -> None:
        phase = WEAPON_PHASES.get(self._phase_key, {})
        self._is_transition = phase.get("weapon_pool_mode") == "weighted"
        # ep_total_reward を env 数に合わせて初期化
        n = self.training_env.num_envs
        self._ep_total_reward = [0.0] * n

        # (a) 同一run resume: output_dir に status があればそれを使う
        if self._status_file and os.path.exists(self._status_file):
            try:
                with open(self._status_file) as f:
                    status = json.load(f)
                if status.get("phase_key") == self._phase_key:
                    # 同じフェーズから再開: 保存済みの start_step を使う
                    self._phase_start_step = status.get("phase_start_step", self.num_timesteps)
                    elapsed = self.num_timesteps - self._phase_start_step
                    print(
                        f"[INFO] WeaponCurriculumCallback: resume 復元 (output_dir) "
                        f"phase={self._phase_key}, phase_start_step={self._phase_start_step}, "
                        f"elapsed={elapsed}"
                    )
                    self._apply_params(elapsed)
                    self._last_update = self.num_timesteps
                    return
            except (json.JSONDecodeError, KeyError, OSError):
                pass

        # (b) branch resume: チェックポイントディレクトリに status がないか探す
        if self._checkpoint_dir:
            ckpt_status = os.path.join(self._checkpoint_dir, "weapon_curriculum_status.json")
            if os.path.exists(ckpt_status):
                try:
                    with open(ckpt_status) as f:
                        status = json.load(f)
                    if status.get("phase_key") == self._phase_key:
                        self._phase_start_step = status.get("phase_start_step", self.num_timesteps)
                        elapsed = self.num_timesteps - self._phase_start_step
                        print(
                            f"[INFO] WeaponCurriculumCallback: branch resume 復元 (checkpoint_dir) "
                            f"phase={self._phase_key}, phase_start_step={self._phase_start_step}, "
                            f"elapsed={elapsed}"
                        )
                        self._apply_params(elapsed)  # 内部で _save_status() を呼ぶ
                        self._last_update = self.num_timesteps
                        return
                except (json.JSONDecodeError, KeyError, OSError):
                    pass

        # (c) 初回起動: 現在の num_timesteps を start_step として記録
        self._phase_start_step = self.num_timesteps
        elapsed = 0  # 訓練開始直後はフェーズ内経過ステップ = 0
        self._apply_params(elapsed)  # 内部で _save_status() を呼ぶ
        self._last_update = self.num_timesteps

    def _save_status(self) -> None:
        """phase_start_step を weapon_curriculum_status.json に保存する。

        output_dir（現在のrun出力先）のみに保存する。
        checkpoint_dir は読み取り専用として扱い、元 run の状態を保護する。
        """
        if not self._status_file:
            return

        os.makedirs(self._output_dir, exist_ok=True)
        with open(self._status_file, "w") as f:
            json.dump({
                "phase_key": self._phase_key,
                "phase_start_step": self._phase_start_step,
            }, f)
        # checkpoint_dir には書き込まない（元 run の状態を保護）

    def _on_step(self) -> bool:
        # EpisodeLogger: エピソード終了時に報酬をJSONLに記録する
        if self._episode_logger is not None:
            infos = self.locals.get("infos", [])
            dones = self.locals.get("dones", [])
            rewards = self.locals.get("rewards", [])
            n = len(infos)
            if len(self._ep_total_reward) != n:
                self._ep_total_reward = [0.0] * n
            for i, (info, done, reward) in enumerate(zip(infos, dones, rewards)):
                self._ep_total_reward[i] += float(reward)
                if done:
                    self._ep_id += 1
                    self._episode_logger.log_episode({
                        "schema_version": "v1.0",
                        "episode_id": self._ep_id,
                        "training_phase": self._phase_key,
                        "training_step": self.num_timesteps,
                        "total_reward": self._ep_total_reward[i],
                        "survival_time": float(info.get("survival_time", 0.0)),
                    })
                    self._ep_total_reward[i] = 0.0

        if not self._is_transition:
            return True
        if self.num_timesteps - self._last_update >= self.update_freq:
            # フェーズ内経過ステップを計算して weapon_weights を補間する。
            # 累積 num_timesteps ではなく elapsed を渡すことで、--resume 再開時に
            # 遷移進捗がリセットされることなくフェーズ内の正しい位置から再開できる。
            elapsed = self.num_timesteps - self._phase_start_step
            self._apply_params(elapsed)
            self._last_update = self.num_timesteps
        return True

    def _apply_params(self, elapsed: int) -> None:
        params = get_params_for_phase(self._phase_key, elapsed)
        results = self.training_env.env_method("set_params", **params)
        # results は List[bool] または List[None] など環境実装次第
        if any(r is False for r in results):
            logger.warning(
                "weapon curriculum params rejected by %d envs (UE5 may not support these params yet)",
                sum(1 for r in results if r is False)
            )
        self._save_status()
        if self._is_transition:
            weights = params.get("weapon_weights", {})
            print(
                f"[INFO] WeaponCurriculumCallback: phase={self._phase_key}, "
                f"elapsed={elapsed}, weapon_weights={weights}"
            )
