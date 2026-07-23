# Survivors deployment target v1

正式 target は `mad_forest_standard_v1.yaml` に固定する。Windows Win64、1920×1080 borderless、日本語 UI、Mad Forest、Antonio、standard 30 分で、Hyper/Hurry/Inverse/Arcana/Limit Break/Golden Eggs/DLC は無効である。build、canonical save、progression、power-up、key binding、choice taxonomy、reference hardware の全フィールドが一致した audit report とその canonical hash がなければ capture/model/runtime/campaign manifest を開始できない。

各 formal attempt は game/launcher 停止および cloud sync 明示無効を確認し、original backup を保持する。canonical save は temp copy、canonical hash 検証、atomic replace の順に復元し semantic attestation を再検証する。post-run hash は証跡としてのみ記録し次 run の親にはしない。終了後は operator 承認のもと original backup を同じ方法で復元する。save 内 RNG coupling は否定できないため `rng_control=uncontrolled` とし、統計的独立性は主張しない。

semantic attestation は caller の真偽値ではない。versioned record に記録された canonical save hash、save format、unlock/item/stage/power-up と reroll/skip/banish capability を canonical bytes の hash および attested profile と再照合する。record の欠落・差分は preflight を拒否する。canonical restore、original backup、original restore の source/target はすべて local fixed NTFS volume gate を通し、UNC/removable/non-NTFS はコピー前に拒否する。

repository の `mad_forest_standard_v1.yaml` は `provenance: test-fixture` であり real target の canonical 実測値を表さない。real-target audit は operator/date/evidence-hash を持つ `provenance: operator-attested` profile の expected 値へ observed metadata を束縛し、test fixture または未 attested profile を fail-closed にする。実 VS の値の採取・供給は real-target setup の deferred 作業である。

preflight attempt は run identity ではない。成功時だけ process 生成前に durable `LAUNCH_INTENT` を commit し、observer または reconciliation で process identity が確認された時点で `FORMAL_RUN_ACTIVATED` になる。`CREATE_PROCESS_FAILED` は `LAUNCH_GATE_FAILED` で outcome 分母外、曖昧な launch は campaign を block する。正式 store の support envelope は local fixed NTFS、SQLite WAL、`synchronous=FULL`、integrity/flush 成功、Win32 broker/process attestation である。

Win64 の intent 更新は temp file の flush 後に atomic replace し、置換後ファイルも flush する。Windows/Python では directory handle の `fsync` を提供できないため、directory entry の crash durability は local fixed NTFS journaling の保証範囲に依存し、電源断に対する絶対 durability は主張しない。UNC、removable、non-NTFS は commit 前に fail-closed とする。

ローカル campaign 実行 operator 自身に対する on-disk evidence の tamper-proofing は、この pure Python 契約の脅威モデル外である。暗号 secret や OS/hardware trust anchor なしには、整合する hash chain と save bytes の手書き偽造、および偽の pinned canonical 値と整合 evidence の組み合わせを原理的に区別できない。gate は固定 path 上の lifecycle chain、live canonical save、original backup を再検証するが、operator に対する完全性保証は 05/06 の Win32 broker、kernel process/job attestation で扱う。

成功判定は timer 1800 到達だけでは確定せず `TARGET_REACHED_PENDING_TRANSITION` に入り、post-30 event を画面上で確認して確定する。off-screen entity、hidden HP/cooldown、global state count/density は release-observable ではない。
