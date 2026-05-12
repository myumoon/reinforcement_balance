"""YAML 設定ファイルと argparse の統合ユーティリティ。"""

import warnings
from pathlib import Path

import yaml


def load_yaml_config(config_path: Path) -> dict:
    """YAML ファイルを読み込んで dict を返す。"""
    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config はトップレベルが mapping である必要があります: {config_path}")
    return data


def apply_yaml_defaults(parser, yaml_config: dict) -> None:
    """YAML 値を argparse デフォルトとして登録する。CLI 引数が常に優先される。

    - type=Path の引数は自動的に Path へ変換する。
    - 未知のキーは警告し無視する。
    """
    known_dests = {a.dest for a in parser._actions}
    path_dests = {a.dest for a in parser._actions if getattr(a, "type", None) is Path}

    filtered = {}
    for key, val in yaml_config.items():
        dest = key if key in known_dests else key.replace("-", "_")
        if dest not in known_dests:
            warnings.warn(f"[config] 未知のキー '{key}' を無視します", stacklevel=2)
            continue
        if dest in path_dests and isinstance(val, str):
            val = Path(val)
        filtered[dest] = val

    parser.set_defaults(**filtered)
