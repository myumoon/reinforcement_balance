"""UE5 HTTP サーバーと通信する gymnasium 環境の基底クラス。"""

import time
import numpy as np
import gymnasium as gym
import requests


class UE5ConnectionError(RuntimeError):
    """UE5 HTTP サーバーとの通信が復旧不能になったことを示す例外。"""

    def __init__(self, endpoint: str, attempts: int, cause: Exception):
        self.endpoint = endpoint
        self.attempts = attempts
        self.cause = cause
        super().__init__(
            f"UE5 HTTP request failed: endpoint={endpoint}, attempts={attempts}, "
            f"cause={type(cause).__name__}: {cause}"
        )


class BaseUE5Env(gym.Env):
    """HTTP 通信を抽象化した UE5 環境基底クラス。

    派生クラスは observation_space と action_space を定義し、
    _action_to_payload をオーバーライドして行動を JSON に変換する。
    """

    metadata = {"render_modes": []}

    def __init__(self, host: str, port: int, connect_timeout: int = 120, frame_skip: int = 1):
        super().__init__()
        self.base_url = f"http://{host}:{port}"
        self.connect_timeout = connect_timeout
        self.frame_skip = frame_skip
        self.session = requests.Session()
        self._server_connected = False
        self._request_seq = 0
        self._last_endpoint = ""

    def _new_session(self):
        try:
            self.session.close()
        except Exception:
            pass
        self.session = requests.Session()

    def _request_json(
        self,
        method: str,
        endpoint: str,
        *,
        json_payload: dict | None = None,
        timeout: float = 10.0,
        retries: int = 2,
        retry_delay: float = 0.5,
    ) -> dict:
        """UE5 HTTP API を呼び出し、一時的な接続断なら短く再試行する。"""
        self._request_seq += 1
        request_id = self._request_seq
        self._last_endpoint = endpoint
        attempts = max(1, retries + 1)
        last_error: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                if method == "GET":
                    resp = self.session.get(f"{self.base_url}{endpoint}", timeout=timeout)
                else:
                    resp = self.session.post(
                        f"{self.base_url}{endpoint}", json=json_payload or {}, timeout=timeout
                    )
                resp.raise_for_status()
                return resp.json()
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_error = exc
                print(
                    "[WARN] UE5 HTTP connection issue: "
                    f"endpoint={endpoint}, request_id={request_id}, "
                    f"attempt={attempt}/{attempts}, error={type(exc).__name__}: {exc}"
                )
                if attempt >= attempts:
                    break
                self._new_session()
                time.sleep(retry_delay * attempt)
            except requests.HTTPError as exc:
                last_error = exc
                print(
                    "[ERROR] UE5 HTTP status error: "
                    f"endpoint={endpoint}, request_id={request_id}, "
                    f"status={getattr(exc.response, 'status_code', 'unknown')}, error={exc}"
                )
                break
            except ValueError as exc:
                last_error = exc
                print(
                    "[ERROR] UE5 HTTP invalid JSON response: "
                    f"endpoint={endpoint}, request_id={request_id}, error={exc}"
                )
                break

        assert last_error is not None
        raise UE5ConnectionError(endpoint, attempts, last_error) from last_error

    def _get_json(self, endpoint: str, *, timeout: float = 10.0, retries: int = 2) -> dict:
        return self._request_json("GET", endpoint, timeout=timeout, retries=retries)

    def _post_json(
        self,
        endpoint: str,
        payload: dict | None = None,
        *,
        timeout: float = 10.0,
        retries: int = 2,
    ) -> dict:
        return self._request_json(
            "POST", endpoint, json_payload=payload, timeout=timeout, retries=retries
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if not self._server_connected:
            self._wait_for_server()
        data = self._post_json("/reset", {"seed": seed}, timeout=10, retries=3)
        self._on_reset(data)
        obs = np.array(data["obs"], dtype=np.float32)
        return obs, {}

    def step(self, action):
        payload = self._action_to_payload(action)
        if self.frame_skip > 1:
            payload["steps"] = self.frame_skip
        data = self._post_json("/step", payload, timeout=10, retries=2)
        obs = np.array(data["obs"], dtype=np.float32)
        return obs, float(data["reward"]), bool(data["done"]), bool(data["truncated"]), data.get("info", {})

    def close(self):
        try:
            self.session.post(f"{self.base_url}/close", json={}, timeout=5)
        except Exception:
            pass
        self.session.close()
        super().close()

    def _action_to_payload(self, action) -> dict:
        """行動を /step リクエストの JSON ペイロードに変換する。派生クラスでオーバーライド。"""
        raise NotImplementedError

    def _on_server_connected(self):
        """サーバー接続完了時に呼ばれるフック。派生クラスでオーバーライド可能。"""
        pass

    def _on_reset(self, data: dict):
        """/reset レスポンス受信時に呼ばれるフック。派生クラスでオーバーライド可能。"""
        pass

    def _wait_for_server(self):
        """PIE 起動を待機し、接続が確立したら _server_connected を True にする。"""
        print(f"[INFO] UE5 サーバー ({self.base_url}) を待機中...")
        print(f"[INFO] UE5 エディタで PIE (▶) を開始してください。"
              f" (タイムアウト: {self.connect_timeout}s)")
        deadline = time.time() + self.connect_timeout
        last_print = time.time()
        while time.time() < deadline:
            try:
                self.session.post(f"{self.base_url}/reset", json={"seed": None}, timeout=1)
                self._server_connected = True
                print("[INFO] UE5 サーバーに接続しました。")
                self._on_server_connected()
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
