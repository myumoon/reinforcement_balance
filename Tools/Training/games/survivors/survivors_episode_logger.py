"""Survivors エピソードログを JSONL 形式で保存するロガー。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


class EpisodeLogger:
    """エピソードデータを JSONL 形式でファイルに追記するロガー。

    ファイル名は ``survivors_episode_log_YYYYMMDD.jsonl`` 形式で日付ごとに作成される。

    Args:
        output_dir: 出力ディレクトリのパス（存在しない場合は自動作成）。
    """

    def __init__(self, output_dir: str | Path) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def _get_log_path(self) -> Path:
        date_str = datetime.now().strftime("%Y%m%d")
        return self._output_dir / f"survivors_episode_log_{date_str}.jsonl"

    def log_episode(self, episode_data: dict) -> None:
        """エピソードデータを JSONL ファイルに追記する。

        Args:
            episode_data: 記録するエピソードデータの dict。
                          ``timestamp`` キーが含まれていない場合は自動付与される。
        """
        if "timestamp" not in episode_data:
            episode_data = {
                "timestamp": datetime.now().isoformat(),
                **episode_data,
            }
        log_path = self._get_log_path()
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(episode_data, ensure_ascii=False) + "\n")
