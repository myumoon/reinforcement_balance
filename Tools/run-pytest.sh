#!/usr/bin/env bash
# reinbalance Python 環境を自動検出して pytest を実行する。
#
# 優先順位: WSL native conda → Windows conda（Claude Code セッション直接実行時）
# Codex は Windows .exe を実行できないため WSL native 環境が見つからない場合はエラーにする。

set -euo pipefail

CANDIDATES=(
  "/home/$USER/miniconda-wsl/envs/reinbalance/bin/python"
  "/home/$USER/miniconda3/envs/reinbalance/bin/python"
  "/home/$USER/miniforge3/envs/reinbalance/bin/python"
  "/mnt/c/Users/$USER/anaconda3/envs/reinbalance/python.exe"
  "/mnt/c/Users/$USER/miniconda3/envs/reinbalance/python.exe"
)

PYTHON=""
for candidate in "${CANDIDATES[@]}"; do
  if [ -f "$candidate" ]; then
    PYTHON="$candidate"
    break
  fi
done

if [ -z "$PYTHON" ]; then
  echo "Error: reinbalance Python environment not found." >&2
  echo "  WSL native: ~/miniconda-wsl/envs/reinbalance/bin/python" >&2
  echo "  Windows:    /mnt/c/Users/\$USER/anaconda3/envs/reinbalance/python.exe" >&2
  exit 1
fi

echo "Using: $PYTHON"
"$PYTHON" -m pip install -e Tools/Common -q
"$PYTHON" -m pytest "$@"
