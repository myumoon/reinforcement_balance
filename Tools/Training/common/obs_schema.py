"""obs_schema 取得・解析ユーティリティ。"""

import sys

import requests


def fetch_obs_schema(host: str, port: int) -> dict:
    url = f"http://{host}:{port}/obs_schema"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        print(f"[ERROR] UE5 サーバー ({host}:{port}) に接続できませんでした。")
        print(f"[ERROR] UE5 エディタで PIE (▶) を起動してから再実行してください。")
        sys.exit(1)


def build_obs_layout(schema: dict) -> tuple[str, dict[str, int]]:
    """obs レイアウト文字列と各セグメントの開始インデックス dict を返す。"""
    lines = []
    offsets: dict[str, int] = {}
    offset = 0
    for seg in schema["segments"]:
        name = seg["name"]
        dim = seg["dim"]
        offsets[name] = offset
        if dim == 1:
            lines.append(f"  obs[{offset}]         = {name}")
        else:
            lines.append(f"  obs[{offset}:{offset + dim}] = {name}  (dim={dim})")
        offset += dim
    lines.append(f"  合計: {schema['total_dim']} 次元")
    return "\n".join(lines), offsets
