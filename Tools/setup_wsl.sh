#!/bin/bash
# WSL から Windows 側の Anaconda reinbalance 環境を使えるようにする一度きりのセットアップ

set -e

MARKER="# reinbalance WSL setup"

# シェルに応じて設定ファイルを選択
if [ -n "$ZSH_VERSION" ] || [ "$SHELL" = "/bin/zsh" ]; then
    RC_FILE="$HOME/.zshrc"
else
    RC_FILE="$HOME/.bashrc"
fi

# 既に設定済みか確認
if grep -q "$MARKER" "$RC_FILE" 2>/dev/null; then
    echo "Already configured in $RC_FILE. Skipping."
    exit 0
fi

# Windows ホームディレクトリを取得
_WIN_HOME_WIN=$(cmd.exe /c "echo %USERPROFILE%" 2>/dev/null | tr -d '\r\n')
WIN_HOME=$(wslpath "$_WIN_HOME_WIN")

# conda env の存在確認
PYTHON_PATH="$WIN_HOME/anaconda3/envs/reinbalance/python.exe"
if [ ! -f "$PYTHON_PATH" ]; then
    echo "Error: $PYTHON_PATH が見つかりません。"
    echo "先に reinbalance conda 環境を作成してください。"
    exit 1
fi

# RC ファイルに追記（シェル起動時に毎回 Windows ホームを解決する）
cat >> "$RC_FILE" << 'EOF'

# reinbalance WSL setup
_WIN_HOME_WIN=$(cmd.exe /c "echo %USERPROFILE%" 2>/dev/null | tr -d '\r\n')
export PATH="$(wslpath "$_WIN_HOME_WIN")/anaconda3/envs/reinbalance:$(wslpath "$_WIN_HOME_WIN")/anaconda3/envs/reinbalance/Scripts:$PATH"
unset _WIN_HOME_WIN
alias python3=python
EOF

echo "セットアップ完了: $RC_FILE に追記しました。"
echo "反映するには以下を実行してください:"
echo "  source $RC_FILE"
