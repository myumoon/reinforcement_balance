# 武器別パフォーマンスメトリクス（W&B）

## 概要

`SurvivorsEvalCallback` の評価ロールアウト終了後に `_aggregate_weapon_metrics` が呼ばれ、
W&B へ武器別・カテゴリ別のメトリクスを記録する。

---

## 1. メトリクス一覧

| W&B キー | 計算式 | 単位 | 備考 |
|---|---|---|---|
| `weapon/{cat}.{name}/active_score_mean` | `mean(active_score)` | — | 武器を持っていたエピソードの active_score 平均 |
| `weapon/{cat}.{name}/kills_per_step_mean` | `mean(kills / ep_length)` | 1/step | 単位時間当たり撃破数 |
| `weapon/{cat}.{name}/enemy_count_mean` | `mean(kills)` | — | kills の平均（名称は enemy_count_mean） |
| `weapon/{cat}.{name}/enemy_dist_mean` | `mean(enemy_dist)` | m | エピソード平均の最近傍敵距離 |
| `weapon/{cat}.{name}/alive_steps_mean_norm` | `mean((ep_len - acq_step) / max(_MAX_EP_STEPS - acq_step, 1))` | 0〜1 | 武器取得後の生存率（0=即死, 1=タイムアップまで生存） |
| `weapon/{cat}.{name}/survival_rate` | `mean(1 - terminated)` | 0〜1 | 時間切れ（タイムアップ）で終了した割合 |
| `weapon/{cat}.{name}/gem_pickups_mean` | `mean(gem_pickups)` | — | ジェム取得数の平均 |
| `weapon/{cat}.{name}/first_weapon_active_score_mean` | first_weapon エピソードの `active_score_mean` | — | 最初に装備した武器として使ったエピソードのスコア |
| `weapon/{cat}.{name}/first_weapon_alive_steps_mean_norm` | first_weapon エピソードの `alive_steps_mean_norm` | 0〜1 | 最初に装備した武器として使ったエピソードの生存率 |

カテゴリ集計は `weapon/{cat}/` プレフィックス（同一指標）。

- 未使用の武器（エピソード中に一度も取得されなかった）は **-1** で記録される。
- `_MAX_EP_STEPS = 3000`（実測 eval/ep_length ≈ 3000 steps に基づく定数）

---

## 2. 武器カテゴリ定義

| カテゴリ | 武器ID | 武器名 |
|---|---|---|
| melee | 1, 2, 7, 16, 17, 22 | Garlic, Whip, KingBible, SoulEater, BloodyTear, UnholyVespers |
| ranged | 3, 4, 5, 6, 8, 10, 13, 14, 18, 19, 20, 21, 23, 25, 28 | MagicWand, Knife, Axe, Cross, FireWand, Runetracer*, Peachone, EbonyWings, HolyWand, ThousandEdge, DeathSpiral, HeavenSword, Hellfire, NoFuture, Vandalier |
| area | 9, 11, 12, 24, 26, 27 | SantaWater, LightningRing, Pentagram, LaBorra, ThunderLoop, GorgeousMoon |
| defensive | 15 | Laurel |

**注意**: Runetracer(ID=10) は `reward_fn.py` の `AREA_IDS` に含まれるが、
メトリクス分類ではユーザー合意により **ranged** に分類する。

---

## 3. 4 観点の比較方法（W&B ダッシュボード）

### 3-1. 立ち回り評価

`weapon/{cat}/enemy_dist_mean` を melee / ranged / area で比較する。

**正常な序列**: `melee < area < ranged`

melee は敵に近づいて攻撃するため最小値になるはず。逆転している武器は間合いを習得できていない疑いがある。

### 3-2. 撃破効率

`weapon/{cat}.{name}/kills_per_step_mean` でカテゴリ内比較。

同カテゴリで突出して低い武器は、報酬設計か UE5 実装側の問題である可能性がある。

### 3-3. 外れ値検出

`active_score_mean` と `alive_steps_mean_norm` を散布図で確認。

**右下（スコア低・生存短）** に位置する武器が優先調査対象。
特定エピソードだけスコアが極端に低い場合は実装不具合を疑う。

### 3-4. 組み合わせ評価

`first_weapon_active_score_mean` vs `active_score_mean` の差を確認。

- **差が正（first > all）**: 単体では強いが、他武器との組み合わせで弱くなる
- **差が負（all > first）**: 他の武器と組み合わせた方が強くなる

---

## 4. 正常時の期待値（Phase 0 目安）

| 武器 | active_score_mean | alive_steps_mean_norm | kills_per_step_mean |
|---|---|---|---|
| Garlic（melee） | 300〜600 | 0.5〜1.0 | 0.003〜0.008 |
| SantaWater（area） | 200〜500 | 0.4〜0.9 | 0.002〜0.006 |
| MagicWand（ranged） | 250〜550 | 0.4〜0.9 | 0.003〜0.007 |

※ Phase・カリキュラム進行に応じて変動する。上記は参考値。

---

## 5. 異常パターン集

| パターン | 考えられる原因 |
|---|---|
| `alive_steps_mean_norm` が全武器で 0.1 以下 | 難易度が高すぎる、または報酬関数の問題 |
| `enemy_dist_mean` が melee > ranged | 間合い管理が未習得、または obs_schema のオフセットずれ |
| `active_score_mean` が特定武器のみ -1 | エピソード中にその武器が一度も取得されていない（weapon_pool_mode 確認） |
| `first_weapon_alive_steps_mean_norm` が `alive_steps_mean_norm` より大幅に低い | 最初の武器で単独行動は苦手だが複数武器で補完できている（正常な場合あり） |
| `kills_per_step_mean` がランダム行動以下（< 0.001） | 武器の射程・判定が機能していない実装バグの疑い |

---

## 6. 実装詳細

### 呼び出しフロー

```
SurvivorsEvalCallback._run_eval_and_log()
  └─ run_survivors_eval_episodes()          # 各 step で weapon_types を収集
  └─ _log_results(metrics)                  # eval/ メトリクスを W&B へ
  └─ _aggregate_weapon_metrics(results)     # 武器別集計
       └─ wandb_logger.log(payload, step=)  # weapon/ メトリクスを W&B へ
```

### `weapon_types` の抽出（`survivors_env.py`）

```python
weapon_slot_i = offset("weapon_slots", 23)
for s in range(6):
    idx = weapon_slot_i + s * 3
    tn = float(obs[idx])
    if tn > 1e-4:
        wtype_id = int(round(tn * 64.0))
```

obs_schema に `weapon_slots` セグメントが存在する場合はオフセットを自動取得し、
存在しない場合はデフォルト 23 を使用する。
