Windows PowerShell から Tools/run-pytest.sh を実行する前に `$env:USER=$env:USERNAME` を設定する（set -u により USER 未定義で停止するため）。
Git Bash はWindowsドライブを `/c` にmountし、実環境のConda名は大文字`Anaconda3`なので、pytestランナーは両表記の候補を持つ必要がある。
Claude の Bash ツール(Git Bash)でも USER は未定義なので `USER=$USERNAME bash Tools/run-pytest.sh ...` の形で実行する。
Windows既定の`%TEMP%/pytest-of-USER`はsandboxでWinError 5になるため、pytestの`--basetemp`はworktree内の実行ごとに一意なパスへ向ける。
Edit ツールの new_string 末尾の空白は落ちることがある（"after_frame: "→"frame: " の replace_all が "frame:5" になり YAML flow map のキーが壊れた）。末尾空白に依存する置換は前後の文字まで含めて書く。
Windows の Anaconda で全 Deployment テストを回すとき、`subprocess.run(text=True)` の子 CLI 出力が CP932 decode failure になる場合があるため、Git Bash 実行時に `PYTHONUTF8=1` を付ける。
Claude の Bash ツールで quoted heredoc(<<'EOF')に書いた Python raw 文字列の `\\` が `\` に潰れることがある（named pipe 名 `\\.\pipe\...` が壊れた）。Windows path/pipe 名は定数（PIPE_PREFIX 等）から組み立てる。
Windows の Python で `Path.write_text()` / `open(..., "w")` を newline 指定なしで使うと LF ファイルが CRLF に変わり diff 全行変更になる。既存ファイルの書き換えは `newline="\n"` を付けるか Edit ツールを使う。
ReinBalanceLogicTests (LLT) 内では FFileHelper::SaveStringToFile が false を返す（UE file manager が書けない）。fixture の読み書きは std::ofstream/std::ifstream を使い、パスは __FILE__ + NormalizeFilename + CollapseRelativeDirectories で作る。
UE5.4 の Build.bat 出力には "Result: Succeeded" 行が出ない。成功判定は exit code 0 と "Target is up to date"/WriteMetadata 行で行う。
Claude の Bash ツールで長い quoted heredoc (cat >> file <<'EOF') が 'unexpected EOF while looking for matching' で丸ごと失敗することがある。長い追記は Write ツールで scratchpad に書いてから cat scratch | tr -d '\r' >> target で足す。
PowerShell の Start-Process で Python をバックグラウンド実行する場合も `-X utf8` を付ける。PYTHONUTF8 は既存シェルから自動で引き継がれるとは限らず、リダイレクトした日本語ログが CP932 になる。
Codex の通常 sandbox で exec_command が `helper_unknown_error: setup refresh had errors` により起動できない場合は、読み取り確認にも require_escalated が必要になる。PowerShell で日本語 JSON を読むときは Get-Content -Encoding UTF8 と UTF-8 の Console.OutputEncoding を明示する。
Windows PowerShell の native `python -c '..."key"...'` は内部の二重引用符が落ちることがある。Python 内の文字列は単引用符にし、PowerShell 側を二重引用符で包むか scratch script を使う。
- 2026-10-11: Deployment の controller test は単独実行時に controller が top-level package になるため `..vision.conftest` は ImportError。vision fixture を相対 import せず、必要な合成画素を対象 test 内で描く。
