# D04 セッション記録

> live capture がこの雛形を使って `session_log.md` を自動生成します。手で書く欄はありません。annotation 工数は COCO export 時に自動で追記されます。save 本体や実機固有の証跡は保存しません。

## 収録

| 項目 | 記録 |
|---|---|
| session_id | {{session_id}} |
| 収録日・開始時刻（ISO 8601 UTC） | {{capture_started_at}} |
| operator | {{operator}} |
| planned split | {{planned_split}} |
| planned_at | {{planned_at}} |
| requested duration_sec | {{duration_sec}} |
| 実測 session 長（秒） | {{actual_duration_sec}} |
| death 検出 | {{run_end_death}} |
| result 検出 | {{run_end_result}} |
| 最後の gameplay frame の timer（秒） | {{last_gameplay_timer_seconds}} |
| Steam AutoUpdateBehavior | {{steam_auto_update_behavior}} |
| session manifest の保存先 | {{session_manifest_path}} |
| session 統計 JSON | {{session_stats_path}} |
| stdout の統計・最終結果 | {{stdout_stats}} |
| 開始時 target profile の保存先 | {{session_profile_path}} |
| resolved YAML SHA-256 | {{resolved_yaml_sha256}} |
| target_profile_hash | {{target_profile_hash}} |

## annotation 工数

COCO export 時に checked label の更新時刻と下書きとの差分から自動で記録します。

| work-root | annotator | 開始時刻 | 終了時刻 | checked frame 数 | 下書きから修正した frame 数 | 下書きなし frame 数 |
|---|---|---|---|---:|---:|---:|

- agreement は 04-10／04-14 では不要、04-08（04-15 の formal world detector 訓練）では必須です。1人の場合は A の最後の label 更新から設定 gap（既定24時間）以上待って `prepare_survivors_self_agreement.py` を実行し、A の label と prelabel を見ずに B を annotation して `build_survivors_annotation_agreement.py` を実行します。準備から agreement 作成まで A root の label は保存しません。gap は `min(manifest.created_at, B label の最小 mtime) - A label の最大 mtime` で測り、manifest より前の B label は拒否されます。必要件数は `minimum_counts.agreement` で確認してください。
