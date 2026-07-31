# Survivors Simulator-to-Real Fidelity

この監査は UE5 simulator 内部の整合性ではなく、固定した本家 target と simulator の
画面から測れる差を判定する。内部 state を本家同等性の証拠には使わない。

## 棚卸しと gate

| 分類 | 計測対象 | baseline gate |
|---|---|---|
| action/time | 9方向、速度、physics/decision cadence、画面変位、pause/level-up timing | action は常に blocking 対象 |
| world | viewport visibility、wave/spawn/density、coarse enemy、boss timing | 未計測は accepted uncertainty |
| progression | XP、gem、level cadence、offer/pool/fallback、slot、evolution/union/chest | offer は常に blocking 対象 |
| combat | cooldown、range、projectile/AoE、画面上の damage pattern | sensitivity と mitigation が必要 |
| terminal | 30:00、post-30 遷移、Reaper/death/result precedence | terminal は常に blocking 対象 |
| observation | screen-space range/count/density、occlusion/clipping/staleness | integration 以後 DeployObs visibility を blocking 対象 |

計測不能 field は推測値で埋めず、policy 入力から除外するか、理由・owner・mitigation を
伴う accepted uncertainty として残す。exact target の verdict と all-content /
generalization coverage report は別 artifact とする。

## One-way schema

`FSurvivorsGameLogic` が enum、定数、XP/進化 table、runtime config から content と
action/time schema を生成する。`ASurvivorsGame` と `/content_schema` は薄い委譲だけを
行う。Python の `fidelity_schema.py` は export に target relevance を注釈するだけで、
ID、level、evolution/union pair を複製しない。

## Verdict lifecycle

- `baseline`: DeployObs schema/release adapter は `absent`。action/offer/terminal の
  blocking 行を必須とし、formal teacher collection、FR4、student release を解禁しない。
- `integration`: 01-02、01-03、03-01、03-03 後に、現在の全 producer hash で新規発行する。
  DeployObs visibility を gate に含める。
- `post_curriculum`: 03-04 の C++/config 変更後に、新しい immutable node として発行する。
  旧 verdict は更新しない。

`provenance` は Git commit、dirty summary、tool/dependency version、operator、timestamp の
再現補助であり、差だけでは失効しない。`gating_producer_hashes` は versioned path
manifest、manifest bytes、generated schema、compiled TU/module implementation closure
を含み、いずれかが現在値と異なれば再監査を要求する。

consumer は利用直前に `verify_current_fidelity` を呼ぶ。allowlist version/key の差、
producer 欠落、stage 不足、hash 差、baseline 流用、blocking 行を warning で継続せず
例外にする。

## 計測と発行

target video/telemetry と simulator run を同じ profile/time band へ揃え、座標を
viewport-normalized unit に変換する。差は session ごとの平均を標本とする cluster CI
で集計する。pilot から session 数、sim run 数、wall-clock、storage、resume と parallel
worker 上限を発行 manifest に固定する。

verdict JSON と Markdown report は同じ directory の temp file を fsync 後に rename し、
片側の準備失敗で孤立 artifact を残さない。blocking が 0 で、accepted gap 全件に理由、
sensitivity、mitigation、owner がある場合だけ pass と表現できる。
