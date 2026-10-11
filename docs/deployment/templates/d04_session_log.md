# D04 セッション記録

> この雛形を Git 管理外の session log 用フォルダーへコピーし、1 session ごとに1枚記入する。save 本体や実機固有の証跡は記録せず、指定欄には SHA-256 と外部保管先だけを書く。

## 収録

| 項目 | 記録 |
|---|---|
| session_id | |
| 収録日・開始時刻（ISO 8601） | |
| operator | |
| planned split | `model_train` / `model_validation` / `error_calibration` / `final_e2e_test` |
| `--duration-sec` | |
| session manifest の保存先 | |
| session 統計 JSON | |
| stdout の統計・最終結果 | |

### run 結果

- run 全体をこの1 session に収録（ステージ選択から死亡または30:00到達後の result 画面まで）: はい / いいえ（理由: ）
- ゲーム内最終時刻: 
- 終了: 死亡 / 30:00到達 / その他（詳細: ）
- result 画面を記録: はい / いいえ（表示秒数: ）
- 実測 session 長（先頭〜末尾 timestamp）: 秒
- 予定用途の最短長を満たす: はい / いいえ / 未判定（適用した gate と理由: ）

### 選択ルールと画素設定

- level-up／宝箱: 固定ルールなし。通常どおり選択する。
- 上記からの例外と状況（なければ「なし」）: 
- session 開始前のオプション画面 PNG（Git 外の保存先）: 
- 収録期間中に設定変更なし: はい / いいえ（内容: ）

| 画素に影響する設定 | 値 |
|---|---|
| ダメージ数値 | |
| 画面効果 | |
| 点滅 | |
| 言語 | |
| 解像度・表示モード | |
| UI scale | |

### save・環境・profile

- Steam Cloud 停止を確認: はい / いいえ
- Vampire Survivors 自動更新・GPU driver 更新・Windows Update の停止を確認: はい / いいえ
- 使用した save backup 世代・保存先: 
- session 前に同じ save を復元: はい / いいえ
- session 前 save SHA-256: 
- session 後 save SHA-256: 
- `.env/target_profile.resolved.yaml` の artifact store 保存先: 
- 同 YAML の backup volume 保存先: 
- resolved YAML SHA-256: 
- `target_profile_hash`: 
- 収録期間中に `target_resolve` を再実行していない: はい / いいえ

### 実機確認（監査 D1／D2／D4／D5）

- D1 対象ウィンドウのフォーカス喪失時に収録が一時停止し、復帰時に再開することを確認: はい / いいえ（記録: ）
- D2 level-up／result 画面で HP bar の見え方を確認: はい / いいえ（結果: ）
- D4 ステージ選択と補正の出所を確認: はい / いいえ（出所: ）
- D5 Steam 自動更新と Steam Cloud が停止していることを確認: はい / いいえ

## split・formal gate

- `--planned-split` と `planned_at` を session manifest で確認: はい / いいえ
- 適用した `formal_gates.v1` YAML（04-18 merge 後のみ）: 
- formal gates YAML SHA-256（annotation 前に確定）: 
- evaluation window plan 保存先（sealed session のみ）: 
- evaluation window plan SHA-256: 
- publish descriptor の `formal_gates.sha256` が上記と一致: はい / いいえ / 未実施

## annotation 工数

| annotator / work-root | 開始時刻 | 終了時刻 | 対象 frame 数 | 下書きから修正した frame 数 |
|---|---|---|---:|---:|
| A | | | | |
| B（独立確認時） | | | | |

- 二重 annotation の場合、B は別 work-root を使用し、A の label と A の frame で学習した prelabel を見ていない: はい / いいえ / 対象外
- 画面全体を annotation し、sealed window 内で `labeled_region` を使用していない: はい / いいえ / 対象外
- HUD truth の保存先: 
- session 単位 COCO の保存先: 
- 04-16 merge 後の formal annotations 保存先（sealed のみ）: 

## 備考

<!-- coverage 不足、FAIL の区分（データ不足 / 性能不足）、stdout 統計の転記など -->

