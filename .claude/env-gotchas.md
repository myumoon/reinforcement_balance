Windows PowerShell から Tools/run-pytest.sh を実行する前に `$env:USER=$env:USERNAME` を設定する（set -u により USER 未定義で停止するため）。
Git Bash はWindowsドライブを `/c` にmountし、実環境のConda名は大文字`Anaconda3`なので、pytestランナーは両表記の候補を持つ必要がある。
Claude の Bash ツール(Git Bash)でも USER は未定義なので `USER=$USERNAME bash Tools/run-pytest.sh ...` の形で実行する。
Windows既定の`%TEMP%/pytest-of-USER`はsandboxでWinError 5になるため、pytestの`--basetemp`はworktree内の実行ごとに一意なパスへ向ける。
Edit ツールの new_string 末尾の空白は落ちることがある（"after_frame: "→"frame: " の replace_all が "frame:5" になり YAML flow map のキーが壊れた）。末尾空白に依存する置換は前後の文字まで含めて書く。
Windows の Anaconda で全 Deployment テストを回すとき、`subprocess.run(text=True)` の子 CLI 出力が CP932 decode failure になる場合があるため、Git Bash 実行時に `PYTHONUTF8=1` を付ける。
