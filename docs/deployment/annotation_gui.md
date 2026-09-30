# Survivors キャプチャの GUI アノテーション

候補フレームを抽出し、自動下書きを作ってから X-AnyLabeling で確認・修正し、COCO JSON を出力します。ゲーム本体や Python の依存関係を追加する必要はありません。

## 1. X-AnyLabeling を用意する

[X-AnyLabeling v4.0.6 の GitHub Releases](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v4.0.6) から Windows 用 exe を取得します。

- 通常は `X-AnyLabeling-v4.0.6-CPU.exe` を使います。
- 対応する CUDA 環境がある場合は `X-AnyLabeling-v4.0.6-Windows-CUDA12.exe` を選べます。

この手順はラベル JSON の `checked` と矩形形式を v4.0.6 に合わせています。

## 2. 候補フレームを抽出する

公開済みキャプチャから、foreground のフレームを選んで work-root に配置します。work-root を省略すると `<store-root>/annotation_work` が使われ、既定で最大 200 枚を stride 15 で抽出します。

```powershell
python Tools/Deployment/select_survivors_annotation_candidates.py `
  --store-root "D:\captures" `
  --session-id "session-0001"
```

必要なら `--work-root "D:\annotation_work"`、`--count 200`、`--stride 15` を指定します。`classes.txt` はこの作業領域に作成されます。

## 3. 自動下書きを作る

確認済みラベルから切り出した見本と固定矩形を使い、指定セッション内の JSON がまだ無い PNG に下書きを作ります。

```powershell
python Tools/Deployment/prelabel_survivors_frames.py `
  --work-root "D:\captures\annotation_work" `
  --session-id "session-0001"
```

初期設定は [annotation_prelabel_v1.yaml](../../Tools/Deployment/configs/annotation_prelabel_v1.yaml) です。固定矩形・照合ラベル・一致 threshold は調整できます。設定のラベル名は共通クラス一覧で検証され、矩形は 1920x1080 の範囲内に置いてください。

下書きは `checked: false` で保存されます。既存 JSON はスキップされます。テンプレート見本には work-root 内の `checked: true` の JSON だけを使います。

## 4. X-AnyLabeling で確認・修正する

1. X-AnyLabeling で `work-root/<session-id>` フォルダを開きます。
2. `Upload` → `Upload Label Classes File` を選び、`work-root/classes.txt` を読み込みます。
3. `R` で矩形を作成し、右側のラベル一覧からクラスを選びます。下書きの位置やラベルが違う場合は修正します。
4. `D` / `A` で次 / 前の画像へ移動します。
5. 確認が終わった画像で `Ctrl+Alt+K` を押して確認済みにします。`Ctrl+Shift+D` は次の未確認画像へ移動します。

自動保存は既定で ON です。COCO 出力に含める画像は `checked: true` にしてください。確認済みで矩形が 0 個の画像も負例として出力されます。

## 5. COCO JSON を出力する

確認済みラベルだけを出力します。`--session-id` を省略すると work-root 直下の全セッションを対象にします。

```powershell
python Tools/Deployment/export_survivors_annotations_coco.py `
  --work-root "D:\captures\annotation_work" `
  --output-dir "D:\datasets\survivors"
```

`world_coco.json` と `ui_coco.json` が作られます。

## クラス早見表

| クラス | 対象 |
|---|---|
| `player_anchor` | プレイヤーの位置 |
| `enemy_normal` / `enemy_elite` / `enemy_boss` | 通常 / エリート / ボス敵 |
| `gem_blue` / `gem_green` / `gem_red` | 青 / 緑 / 赤の経験値ジェム |
| `pickup_heal` / `pickup_special` | 回復 / 特殊ピックアップ |
| `hazard_projectile` / `hazard_area` | 飛来する弾 / 範囲攻撃 |
| `hud_hp` / `hud_xp` | HP / XP の HUD 領域 |
| `card` / `button` | レベルアップカード / 選択ボタン |
| `death_result` | 死亡・結果画面の領域 |

宝箱は `chest` ではなく `pickup_special` として付けます。`card`、`button`、`death_result` は自動下書きされないため、GUI で必要な矩形を追加してください。

`hazard_projectile` / `hazard_area` は円（circle）ツールで描いても構いません。中心点と円周上の一点から外接する矩形へ自動変換されます。

タイプミスで未知のラベル名が付いた shape や、rectangle/circle 以外の未対応図形は、そのファイルを読む際に警告を表示して読み飛ばされます（ファイル全体は失敗しません）。警告は `prelabel_survivors_frames.py` や `export_survivors_annotations_coco.py` の実行時に標準エラー出力へ表示されるので、意図しない読み飛ばしがないか確認してください。

## 2周目以降の運用

確認済みの矩形は次の実行からテンプレート見本に使われます。先に一部の画像を確認済みにし、同じ work-root の未処理セッションで下書きを作ると、見本が増えた状態で照合できます。既存 JSON は上書きされないため、下書きが無い PNG にだけ新しい JSON が作られます。

同一セッション内で先に一部だけ確認済みにし、残りの未確認フレームにも増えた見本を反映したい場合は `--refresh-unchecked` を付けます。`checked: false` の既存下書きだけが新しい見本で再生成され、`checked: true` にした矩形は変更されません。

```powershell
python Tools/Deployment/prelabel_survivors_frames.py `
  --work-root "D:\captures\annotation_work" `
  --session-id "session-0001" `
  --refresh-unchecked
```

未確認のまま GUI で手動編集を始めているフレームがある場合、`--refresh-unchecked` を実行すると下書きが編集前の状態に戻ってしまいます。手を付けたフレームは先に確認済みにしてから実行してください。
