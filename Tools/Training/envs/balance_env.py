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

    def __init__(self, host: str = "127.0.0.1", port: int = 8765, connect_timeout: int = 120):
        super().__init__()
        self.base_url = f"http://{host}:{port}"
        self.connect_timeout = connect_timeout
        self.session = requests.Session()
        self._server_connected = False  # 初回接続済みフラグ

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if not self._server_connected:
            self._wait_for_server()  # PIE 起動待機（初回のみ）
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
        """PIE 起動を待機する。接続が確立したら _server_connected を True にする。"""
        print(f"[INFO] UE5 サーバー ({self.base_url}) を待機中...")
        print(f"[INFO] UE5 エディタで PIE (▶) を開始してください。"
              f"(タイムアウト: {self.connect_timeout}s)")
        deadline = time.time() + self.connect_timeout
        last_print = time.time()
        while time.time() < deadline:
            try:
                self.session.post(f"{self.base_url}/reset", json={"seed": None}, timeout=1)
                self._server_connected = True
                print("[INFO] UE5 サーバーに接続しました。")
                return
            except (requests.ConnectionError, requests.Timeout):
                if time.time() - last_print >= 5.0:
                    remaining = max(0, int(deadline - time.time()))
                    print(f"[INFO] 待機中... (残り約 {remaining}s)")
                    last_print = time.time()
                time.sleep(0.1)
        raise TimeoutError(
            f"UE5 サーバーが起動しませんでした ({self.base_url})\n"
            f"  → UE5 エディタで PIE (▶) を開始してから再実行してください。"
        )
