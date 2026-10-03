# T1 → T2 引き継ぎ（deploy_raw C++ producer）

## fixture（Python テストの唯一の正）

- `Tools/Training/tests/survivors/fixtures/deploy_raw_llt_v1.json`
- LLT `SurvivorsDeployRawTests.cpp` の `[fixture]` テストが、現在の C++ 出力と完全一致することを毎回確認する。
  再生成: `cd /w && REINBALANCE_WRITE_DEPLOY_RAW_FIXTURE=1 ./ReinBalance/Binaries/Win64/ReinBalanceLogicTests/ReinBalanceLogicTests.exe -r console "[fixture]"`
- 実 UE5 PIE からの取得ではない（WAITING_MANUAL）。
- トップレベル: `description`, `seed`(73013), `action_rule`, `initial_weapons`, `obs_schema_hash`, `responses[8]`
  - `responses[0]` = `{"endpoint":"/reset","step":0,"deploy_raw":{...}}`（HTTP /reset 応答のトップレベル `deploy_raw` に相当）
  - `responses[1..]` = `{"endpoint":"/step","step":k,"info":{"deploy_raw":{...}}}`（HTTP /step 応答の `info.deploy_raw` に相当）。k = 60,61,62,63（連続 tick）,180,181,360
  - fixture には obs / reward 等は入れていない（必要なら T2 のテスト側で包む）

## HTTP 仕様

- 有効化: `POST /params {"deploy_raw": true}`（JSON bool 以外は `{"error":"deploy_raw must be bool"}` で拒否、他 field も更新しない）。`false` で無効化。reset では解除しない。既定は無効。
- 有効時: `/reset` 応答 = `{"obs":[...],"obs_schema_hash":"...","deploy_raw":{...}}`、`/step` 応答 = `{..., "info":{...既存キー..., "deploy_raw":{...}}}`。
- 無効時: 応答文字列は main と同一（キー自体を出さない）。

## deploy_raw JSON（キー順固定・全キー必須）

```
{
  "schema_version": "survivors_deploy_raw.v1",
  "elapsed_s": float,
  "camera": {"center_x", "center_y", "half_width": 400, "half_height": 225, "cull_margin": 100},   // center = 自機位置
  "player": {"world_x", "world_y", "hp_ratio": 0..1, "level": int},
  "duration_mult": float,                                   // CachedPassiveEffects.DurationMult
  "weapon_slots":  [{"index": 0..5, "type_id": int, "level": int}] * 6,   // type_id = EWeaponType 値。空き = 0 (None)
  "passive_slots": [{"index": 0..5, "type_id": int, "level": int}] * 6,   // type_id = EPassiveItemType 値
  "entities": [{
      "entity_id": int,            // int64。上位(>>40) = id 空間 1 敵 / 2 ジェム / 3 projectile / 4 zone / 5 orbit / 6 aura
      "class_name": str,           // enemy_normal | enemy_boss | gem_blue | gem_green | gem_red | weapon_projectile | weapon_zone | weapon_orbit | weapon_aura
      "world_x": float, "world_y": float,
      "radius_world": float,       // 敵 = CollisionRadius、ジェム = 0（sim に定数が無い）、エフェクト = GetProjectileObsView の Radius
      "slot": int | null,          // 武器エフェクトのみ（武器スロット）。敵・ジェムは null
      "ttl_true_s": float | null,  // 武器エフェクトのみ（sim の真の残り時間。oracle 専用。aura は MaxProjectileObsTtl=8 固定）
      "warning": bool              // 武器エフェクトの warning 中（SantaWater 予兆など）。敵・ジェムは false
  }]
}
```

- `type_id` → 名前: Common `deploy_obs_v2_features.yaml` の `weapon_vocabulary` / `passive_vocabulary` は C++ enum 値と同じ並び（parity テストで保証）なので添字で引く。範囲外は `unknown`。
- entities は カメラ範囲 + 余白（|dx| <= 400+100, |dy| <= 225+100、境界含む）で除外済み。最終の可視判定（中心が画面内）は Python 側。
- `occluded` / `timestamp_ns` は出していない（sim に遮蔽は無い = 常に false。timestamp は `elapsed_s` から作る）。
- 武器エフェクトの範囲は `GetProjectileObsView()` と同じ（orbit は KingBible / UnholyVespers のみ、aura は Garlic / SoulEater のみ）。

## id 規則

- 敵・ジェム: 既存 `UniqueId`。projectile / zone: 生成時に `NextEffectId++`（新規 `EffectId` フィールド）。
- orbit: KingBible の `ActivateOrbs`（周期開始）ごとに `OrbitCycleId = AllocateEffectId()`、id のローカル部 = `OrbitCycleId * 256 + 本番号`。周期が変われば全ての本が新 id。
- aura: ローカル部 = `slot * 256 + EWeaponType 値`（常時存在で不変、進化で武器種が変われば新 id）。
- Reset で全カウンタが 0 に戻る（同じ seed・行動列なら同じ id 列）。

## 追加した C++ 公開 API

- `Public/Survivors/SurvivorsDeployRaw.h`: `ESurvivorsDeployRawIdSpace`, `SurvivorsDeployRaw::{CullMarginU, LocalIdBits, MakeEntityId, IsWithinCullRange, ToJson, SchemaVersion}`, `FSurvivorsDeployRawState / Entity / Slot`
- `FSurvivorsGameLogic::BuildDeployRawState(float CullMarginU = 100) const`, `AllocateEffectId()`, `NextEffectId`
- `FProjectileState::EffectId`, `FGroundZoneState::EffectId`, `FProjectileObsState::EntityId`, `FSurvivorsWeaponLogic::GetOrbitOrbCycleId()`
- `ASurvivorsGame::bDeployRawEnabled`（/params の deploy_raw が書く）
