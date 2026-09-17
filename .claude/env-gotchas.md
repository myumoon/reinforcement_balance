Windows PowerShell から Tools/run-pytest.sh を実行する前に `$env:USER=$env:USERNAME` を設定する（set -u により USER 未定義で停止するため）。
Git Bash はWindowsドライブを `/c` にmountし、実環境のConda名は大文字`Anaconda3`なので、pytestランナーは両表記の候補を持つ必要がある。
