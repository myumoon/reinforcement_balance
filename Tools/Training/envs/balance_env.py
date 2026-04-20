import time
import numpy as np
import gymnasium as gym
import requests


class BalanceEnv(gym.Env):
    """UE5 BalancePole gymnasium ラッパー。
    UE5 側の HTTPServer に接続して観測・行動をやり取りする。
    """

    metadata = {"render_modes": []}

    # カート位置・速度・下ポール角度・角速度・上ポール角度・角速度
    _OBS_LOW = np.array(
        [-4.8, -np.inf, -np.deg2rad(60), -np.inf, -np.deg2rad(60), -np.inf],
        dtype=np.float32,
    )
    _OBS_HIGH = np.array(
        [4.8, np.inf, np.deg2rad(60), np.inf, np.deg2rad(60), np.inf],
        dtype=np.float32,
    )

    def __init__(self, host: str = "127.0.0.1", port: int = 8765, connect_timeout: int = 30):
        super().__init__()
        self.base_url = f"http://{host}:{port}"
        self.connect_timeout = connect_timeout
        self.session = requests.Session()

        self.observation_space = gym.spaces.Box(
            low=self._OBS_LOW, high=self._OBS_HIGH, dtype=np.float32
        )
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._wait_for_server()
        payload = {"seed": seed}
        resp = self.session.post(f"{self.base_url}/reset", json=payload, timeout=10)
        resp.raise_for_status()
        obs = np.array(resp.json()["obs"], dtype=np.float32)
        return obs, {}

    def step(self, action):
        force = float(np.clip(action, -1.0, 1.0))
        resp = self.session.post(f"{self.base_url}/step", json={"force": force}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        obs = np.array(data["obs"], dtype=np.float32)
        return obs, float(data["reward"]), bool(data["done"]), bool(data["truncated"]), {}

    def close(self):
        try:
            self.session.post(f"{self.base_url}/close", json={}, timeout=5)
        except Exception:
            pass
        self.session.close()
        super().close()

    def _wait_for_server(self):
        deadline = time.time() + self.connect_timeout
        while time.time() < deadline:
            try:
                self.session.post(f"{self.base_url}/reset", json={"seed": None}, timeout=1)
                return
            except (requests.ConnectionError, requests.Timeout):
                time.sleep(0.1)
        raise TimeoutError(f"UE5 server not available at {self.base_url}")
