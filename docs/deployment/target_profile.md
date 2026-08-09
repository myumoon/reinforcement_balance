# Survivors deployment target v1

正式 target は `mad_forest_standard_v1.yaml` に固定する。Windows Win64、1920×1080 borderless、日本語 UI、Mad Forest、Antonio、standard 30 分で、Hyper/Hurry/Inverse/Arcana/Limit Break/Golden Eggs/DLC は無効である。build、canonical save、progression、power-up、key binding、choice taxonomy、reference hardware の全フィールドが一致した audit report とその canonical hash がなければ capture/model/runtime/campaign manifest を開始できない。

各 formal attempt は game/launcher 停止および cloud sync 明示無効を確認し、original backup を保持する。canonical save は temp copy、canonical hash 検証、atomic replace の順に復元し semantic attestation を再検証する。post-run hash は証跡としてのみ記録し次 run の親にはしない。終了後は operator 承認のもと original backup を同じ方法で復元する。save 内 RNG coupling は否定できないため `rng_control=uncontrolled` とし、統計的独立性は主張しない。

semantic attestation は caller の真偽値ではない。versioned record に記録された canonical save hash、save format、unlock/item/stage/power-up と reroll/skip/banish capability を canonical bytes の hash および attested profile と再照合する。record の欠落・差分は preflight を拒否する。canonical restore、original backup、original restore の source/target はすべて local fixed NTFS volume gate を通し、UNC/removable/non-NTFS はコピー前に拒否する。

repository の `mad_forest_standard_v1.yaml` は `provenance: test-fixture` であり real target の canonical 実測値を表さない。real-target audit は operator/date/evidence-hash を持つ `provenance: operator-attested` profile の expected 値へ observed metadata を束縛し、test fixture または未 attested profile を fail-closed にする。実 VS の値の採取・供給は real-target setup の deferred 作業である。

## 実機MachineInfoの採取と利用

`Tools/Utility/MachineInfo/MachineInfoGetter.ps1` は、実機の GPU、VRAM、ドライバー、OS build、Windows のディスプレイ拡大縮小を採取し、同じディレクトリの `output/machine_info_YYYYMMDD_HHMMSS.json` に保存する。この JSON は通常の UE5 シミュレータ訓練（`Tools/Training/train.py`）の入力ではなく、Vampire Survivors 実機の feasibility、capture、model/runtime、campaign を開始するための TargetProfile 証跡である。

### 採取手順

1. TargetProfile と同じ実機・同じモニター構成で、管理者として PowerShell を起動する。
2. 次を実行する。

   ```powershell
   powershell.exe -NoProfile -ExecutionPolicy Bypass `
     -File ".\Tools\Utility\MachineInfo\MachineInfoGetter.ps1"
   ```

3. `Tools/Utility/MachineInfo/output/machine_info_*.json` が生成されたことを確認する。
4. JSON の `errors` が空配列であることを確認する。GPU または OS が `アクセス拒否` になった場合、その JSON は正式証跡に採用せず、管理者 PowerShell で再実行する。
5. JSON を内部 artifact store に保存し、`artifact://sha256/<64-hex>` を operator-attested TargetProfile の evidence として記録する。`output` フォルダは採取時の作業場所であり、artifact store の代替ではない。

### TargetProfileへの対応

| MachineInfo JSON | TargetProfile | 備考 |
|---|---|---|
| `gpu[0].Name` | `hardware.gpu_name` | 使用するGPUを特定する |
| `gpu[0].VRAM_GiB × 1024` | `hardware.vram_mb` | MiB相当へ換算する |
| `gpu[0].DriverVersion` | `hardware.driver_version` | 実機値を記録する |
| `os.BuildNumber` | `hardware.os_build` | Windows buildを記録する |
| `display_scaling.scale_percent ÷ 100` | `display.windows_dpi_scale` | 100%は`1.0`、125%は`1.25` |

`display_scaling.scope` は `system/primary display` である。複数モニターで倍率が異なる場合は、ゲームを表示するモニターの Windows 設定も手動確認する。ゲーム内の `display.ui_scale`、`monitor_output`、capture backend、CUDA/PyTorch のバージョンは MachineInfo では確定しないため、TargetProfileの実機設定と別の証跡で記録する。

この成果物は環境が変わらない限り毎回の訓練で再生成しない。GPU、ドライバー、Windows build、Windows DPI、主モニター、capture backend を変更した場合、および別PCで収録する場合は再採取し、TargetProfileの evidence hash と hardware profile を更新する。MachineInfo JSON 単独では TargetProfile の audit verdict を発行できず、build、canonical save、progression、入力、choice taxonomy 等の他の必須証跡も揃える必要がある。

canonical profile が pin する全値は expected/actual の双方へ束縛する。特に executable/save hash は canonical 値との一致に加えて実ファイル bytes と一致しなければならず、save bytes に含まれる progression、unlock、購入済み power-up、reroll/skip/banish count もこの束縛で被覆する。`target_identity_hash` は同一 target で安定させるため operator/date/evidence-hash 等の audit-event `manual_attestation` を含めない。key binding、choice support、display の runtime 反映、実行中 character/stage 選択など非ファイル由来 runtime state の独立 observation 証跡は perception/runtime harness を必要とするため 05/06 へ deferred とする。

preflight attempt は run identity ではない。成功時だけ process 生成前に durable `LAUNCH_INTENT` を commit し、observer または reconciliation で process identity が確認された時点で `FORMAL_RUN_ACTIVATED` になる。`CREATE_PROCESS_FAILED` は `LAUNCH_GATE_FAILED` で outcome 分母外、曖昧な launch は campaign を block する。正式 store の support envelope は local fixed NTFS、SQLite WAL、`synchronous=FULL`、integrity/flush 成功、Win32 broker/process attestation である。

Win64 の intent 更新は temp file の flush 後に atomic replace し、置換後ファイルも flush する。Windows/Python では directory handle の `fsync` を提供できないため、directory entry の crash durability は local fixed NTFS journaling の保証範囲に依存し、電源断に対する絶対 durability は主張しない。UNC、removable、non-NTFS は commit 前に fail-closed とする。

ローカル campaign 実行 operator 自身に対する on-disk evidence の tamper-proofing は、この pure Python 実装(仕様)の脅威モデル外である。暗号 secret や OS/hardware trust anchor なしには、整合する hash chain と save bytes の手書き偽造、および偽の pinned canonical 値と整合 evidence の組み合わせを原理的に区別できない。gate は固定 path 上の lifecycle chain、live canonical save、original backup を再検証するが、operator に対する完全性保証は 05/06 の Win32 broker、kernel process/job attestation で扱う。

成功判定は timer 1800 到達だけでは確定せず `TARGET_REACHED_PENDING_TRANSITION` に入り、post-30 event を画面上で確認して確定する。off-screen entity、hidden HP/cooldown、global state count/density は release-observable ではない。
