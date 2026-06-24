"""SurvivorsEnv factory（SubprocVecEnv 向け top-level callable）。"""
from __future__ import annotations

from typing import Optional


def _is_multiprocessing_child() -> bool:
    try:
        import multiprocessing as mp

        parent_process = getattr(mp, "parent_process", None)
        if parent_process is not None:
            return parent_process() is not None
        return mp.current_process().name != "MainProcess"
    except Exception:
        return False


def _ignore_sigint_in_subprocess() -> None:
    if not _is_multiprocessing_child():
        return

    try:
        import signal

        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (AttributeError, OSError, ValueError):
        pass


class SurvivorsEnvFactory:
    """SubprocVecEnv 向けの SurvivorsEnv factory。

    Windows の spawn モードで pickle 可能なよう top-level class として定義する。
    reward_fn は関数オブジェクトを持たず、path 文字列を渡して子プロセス内でロードする。
    DummyVecEnv でも使用できる（既存の _make_survivors_fn を置き換える）。
    """

    def __init__(
        self,
        host: str,
        port: int,
        frame_skip: int,
        reward_fn_path: Optional[str] = None,
    ):
        self.host = host
        self.port = port
        self.frame_skip = frame_skip
        self.reward_fn_path = reward_fn_path

    def __call__(self):
        _ignore_sigint_in_subprocess()

        from games.survivors.survivors_env import SurvivorsEnv, SurvivorsMonitor
        env = SurvivorsEnv(host=self.host, port=self.port, frame_skip=self.frame_skip)
        try:
            if self.reward_fn_path:
                import importlib.util
                path = self.reward_fn_path  # 絶対パスで渡すこと（child process の cwd に依存しない）
                spec = importlib.util.spec_from_file_location("_reward_fn_mod", path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"reward_fn のロードに失敗しました: {path}")
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                env._reward_fn = mod.reward_shaping  # train.py の _load_reward_fn() と同じ属性名
            return SurvivorsMonitor(env)
        except Exception:
            env.close()
            raise
