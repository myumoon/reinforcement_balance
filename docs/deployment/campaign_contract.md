# Survivors live canary campaign契約

## 対象範囲とidentity

このパッケージは20 slot構成のSurvivors campaignに対する、pure Pythonの契約を定義する。processの起動やcontroller I/O、durable ledgerへの永続化は行わない。それらは後続phaseの責務である。

すべてのmanifestはschema `survivors.campaign.v1` を使い、固定された `stage`（`C0`–`C4`）と、そのstageのfrozen policyに一致する `expected_slots` を持つ。また `rng_control: "uncontrolled"` と `trial_separation: "unique_run_id_separate_process"` を別fieldとして必須とする。run IDとprocessが別であることは統計的独立性を保証しない。そのためseed値や独立性の主張は、manifest・prerequisite・event detailsのいずれにおいても拒否する。

eventの経路は次の通り。

```text
FORMAL_SLOT_RESERVED
  -> ATTEMPT_PREFLIGHT(attempt_id)
  -> PREFLIGHT_FAILED
  -> LAUNCH_INTENT_COMMITTED(reserved_run_id, gameplay_attempt_id, launch_nonce)
  -> BROKER_PROCESS_ATTESTED(process_ref, job_ref)
  -> PROCESS_LAUNCH_CONFIRMED
  -> FORMAL_RUN_ACTIVATED(activation_source=normal|reconciliation)
  -> SUCCESS | GAMEPLAY_FAILURE | SAFETY_FAILURE | ARTIFACT_FAILURE
```

`PREFLIGHT_FAILED`、`LAUNCH_GATE_FAILED`、`LAUNCH_UNCERTAIN` は、activation前にそのslotを終了させる。1つのslotが持てるpreflight attemptは1回まで、launch intentは最大1回までである。IDはそのidentity種別の中で一意でなければならない。slotの再予約はできず、terminal eventの上書きもできない。各activationは、1つのattempt・1つのattested process/job・1つのgameplay attemptにそれぞれ紐づく。

## Formal prerequisites

Formal prerequisitesは、`exact_runtime`、`target`、`save`、`build`、`hardware`、`perception_final`、`shadow`、`replay`、`input_safety`、`restore` の10個のevidence keyだけを持つ。各keyは小文字のSHA-256 hash、parent hash、`PASS` statusを持つ。すべてのparentは、manifestが期待するparent hashと一致しなければならない。

このbundleはさらに、検証済みのcloud sync、backup hash、pre-save/post-save contract hashを必須とする。欠落・古い値・parentの不一致・非PASS・development-only・未知fieldはいずれも拒否する。synthetic manifestは常に `development_only: true` であり、formal prerequisite evidenceを含むことはできない。formal manifestは、検証済みでdevelopment-onlyでないprerequisite bundleを必須とする。

このphaseでは、evidence hashの形式と共通parentへの束縛のみを検証する。これらevidence hashを独立に観測した期待値と突き合わせる処理は06-04のformal issuance flowの担当範囲であり、この契約単体ではsynthetic fixtureからformal campaignを発行することはない。

## Frozen stage policies

支給されたphase契約はfieldの名前だけを示し数値を与えていなかったため、この実装では以下の値を凍結する。変更するには明示的な契約改定が必要である。

| Stage | Duration | Slots | Promotion floor (successes) |
|---|---:|---:|---:|
| C0 | 30分 | 2 | 2 |
| C1 | 60分 | 4 | 3 |
| C2 | 120分 | 8 | 6 |
| C3 | 240分 | 16 | 12 |
| C4 | 480分 | 20 | 16 |

## 分母とreport規則

分母は `FORMAL_RUN_ACTIVATED` eventの件数である。成功率は `SUCCESS / activated runs` で計算する。preflight failure、launch-gate failure、uncertain launchは分母から除外し、stage promotionをblockする。これらは代替枠を作らない。activation後のgameplay failure、safety failure、artifact failureは分母に残り、代替できない。

計画された各slotは、activation前のfailureかactivated terminal outcomeのいずれかで終わらなければならない。欠落slot、reservedのままのslot、進行中のpreflight、terminal outcomeを持たないactivated slotは `incomplete_slot_ids` に現れる。該当slotが1つでもあればstageをblockし、promotion対象外とする。terminal outcomeを持たないactivated slotも分母には数える。

reportにはobserved rateと、両側95%のWilson score intervalを含める。20 activation中16 successの場合、rateは0.8、intervalはおよそ0.583–0.919として報告する。母集団の成功確率に関する主張は一切行わない。20分の15の場合はrate 0.75となり、C4のfrozen floorである16には届かない。

reportには `support_outside_ui`、failure taxonomy、blocked/superseded campaign ID、manifestのcanonical hashを含める。すべてのevent wireは `campaign_manifest_hash` を持ち、event validationとreport生成の両方で、各eventのhashと `event_manifest_hash` 引数がmanifestのcanonical hashに一致することを要求する。report wireには、存在する場合 `prerequisite_parent_hash` も記録する。synthetic reportは `development_only: true` と `formal_parent_eligible: false` を持ち、formal C0 parentにはなれない。

## Canonical fixtures

`Tools/Deployment/tests/campaign/fixtures/golden_campaigns.jsonl` には、normal 16/20、normal 15/20、reconciliation activation、preflight failure、uncertain launch、duplicate processの6種類のcanonical scenario envelopeを収録する。各行はstage、event-to-manifestの束縛、event・JSONL・manifest・reportそれぞれの期待hashを固定する（duplicate-processのreport hashはvalidationがrejectするためnullとなる）。fixtureはsynthetic値のみを使い、生成されたreportと同じ厳密なevent／manifest schemaを通して解析する。
