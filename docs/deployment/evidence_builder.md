# Survivors Goal Evidence Builder

## 目的

`Tools/Deployment/survivors/release/evidence_builder.py` は campaign artifact と durable launch ledger を読み戻し、Mad Forest C4 の goal evidence を再計算します。runner summary と report JSON の集計値は結果の根拠にせず、campaign schema、ledger 履歴、save artifact を検証してから数えます。

16/20 は20 slotの観測campaign基準です。Wilson区間は観測結果の補助値で、母集団成功率80%の証明や試行の統計的独立性を主張しません。`rng_control` と `trial_separation` は別々の出力fieldです。

## 読み取り契約

`build_goal_evidence(ArtifactStore, DurableLaunchStore)` は次の契約を独立して読みます。

- 06-02: `CampaignManifest.from_wire()`、`CampaignEvent.from_wire()`、`validate_campaign_events()`、C4の `STAGE_POLICIES`。
- 06-03: `attempt_ids()`、`history()`、`campaign_events()`、`check_storage()`。履歴の終端、row hash、process attestation、WAL/FULL/integrity、固定NTFSを確認します。`DurableLaunchStore` の生成時検証が成功していることも前提です。
- 06-04: runnerの `ArtifactStore`、`campaign/plan.json`、`campaign/summary.json`、stage manifest、per-run manifest、preflight/outcome/gate stream、save backup/canonical/restore artifact。

summary内のplan hash、formal eligibility、run manifest一覧は実ファイルと独立計算に照合します。per-run manifestは `save_hashes_complete` を実save記録から再計算し、単独の `formal_evidence_eligible` field は認めません。

## Artifact追加契約

### Release chain

`campaign/goal_release_chain.json` は `survivors.goal_release_chain.v1` schemaです。`nodes` は `target_profile`, `game_build`, `combat_model`, `vecnormalize`, `deploy_schema`, `error_profile`, `selector`, `parser`, `detector`, `controller`, `training`, `teacher`, `dataset`, `evaluation` の全14 nodeを持ちます。

各nodeは `schema_version`, `content`, `parents`, `sha256` を持ち、`sha256` は他の3 fieldのcanonical hashです。profile/build nodeを根に、deployment component nodeを両rootへ結び、training → teacher → dataset → evaluation の親hashを固定します。外側の `chain_sha256` は `chain_sha256` 自身を除いたchain objectのcanonical hashです。formal campaignではtarget/build hashを06-02 prerequisite bundleにも照合します。

### Remediation close

過去のpreflight failureとsafety failureはhistoryから除外しません。C0〜C4すべてのstage executionで該当issueを `campaign/remediation_closures.json` の `survivors.campaign_remediation_closures.v1` recordによりcloseし、証跡artifactのSHA-256と担当者・独立検証者の異なるIDを確認します。closeが無い、証跡hashが異なる、または担当者と検証者が同一なら拒否します。

`campaign/superseded.json` があるcampaignは、failure remediationのclose状況にかかわらずgoal evidenceを生成しません。後継campaignの記録はhistoryとして保持しますが、superseded root自体はpromotion対象にしません。

## 出力と実行

synthetic fixtureは開発用manifest/reportを返し、`development_only=true` と `goal_release_eligible=false` を付けます。release eligibilityを返すには非development campaignのrootを `formal_c4_root` として明示し、summaryのformal証跡も独立検証に通す必要があります。

`write_evidence_bundle()` は入力を再読込し、sanitized manifest/reportとhash参照だけを一時primary/backup storeに書きます。empty rootへbackupをrestoreしてobjectsを照合した後、2つのoutput directoryへ配置します。development outputの保存先はOSのtemporary directory配下の兄弟directoryです。`docs/goal.md` とGit管理release manifestは書き換えません。

```powershell
python Tools/Deployment/build_survivors_goal_evidence.py `
  --artifacts-root <campaign artifacts> `
  --ledger-directory <durable ledger directory> `
  --backup-directory <temporary backup output> `
  --output-directory <temporary primary output>
```

formal rootを検証する場合は同じartifact rootを `--formal-c4-root` に追加します。結果JSONはmanifest/report hashだけをstdoutへ返します。source payload、absolute user path、secret、raw frameはbundleに含めません。
