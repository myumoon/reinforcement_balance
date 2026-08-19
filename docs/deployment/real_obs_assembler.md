# Real observation assembler
`RealObsAssembler` は `HudStateV1` と `TrackedWorldStateV1` を同じ policy tick に結合し、`PerceptionSnapshot` を生成する 04-09 の本番観測境界です。

## 境界
- `DeployObservation` は visible (`on_screen=true`, `clipped=false`) な track と HUD 値だけから生成する。
- `UiPresentationSnapshotV1` の ROI はクリック実行専用で、deploy/item tensor と diagnostics に含めない。
- presentation は `PerceptionSnapshot` に atomic に格納し、別の `HudStateV1` side channel を公開しない。
- `source_content_hash` は ROI/confidence を 1e-4 量子化し、`ui_state_key` は連続値と時刻を除外する。
- 15 Hz tick で join し、50 ms 超の skew と 200 ms 超の stale source は fail-closed にする。
- `gameplay` は combat、level-up/fallback/chest は item context のみ有効にする。

## Formality
fixture は development-only です。04-10 の正式 parser artifact hash、calibration replay、fidelity verdict を発行する能力はありません。
