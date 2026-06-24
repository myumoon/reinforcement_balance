#pragma once
#include "CoreMinimal.h"
#include "Survivors/Game/Weapons/SurvivorsWeaponBase.h"
#include "SurvivorsAxeWeapon.generated.h"

/**
 * Axe / DeathSpiral: 上方向弧弾（重力あり・放物線弾道）。
 * Axe は 0.2s 間隔の順次発射。1発目は真上、追加弾は横方向オフセット付き。
 * Area は hitbox 半径に効く（wiki: "Axe scales with Area × 1.3"）。
 * Speed は弧の高さに効く（wiki: "Speed determines how far up axes can be thrown"）。
 * DeathSpiral は全弾同時発射（仕様変更なし）。
 */
UCLASS()
class REINBALANCE_API USurvivorsAxeWeapon : public USurvivorsWeaponBase
{
	GENERATED_BODY()
public:
	virtual void Tick(float Dt) override;
	virtual void OnLevelChanged(FWeaponLevel NewLevel) override;

private:
	float CachedDamage    = 20.f;
	float CachedCooldown  = 4.00f;
	float CachedSpeed     = 360.f;
	float CachedArcHeight = 120.f;
	int32 CachedAmount    = 1;
	int32 CachedPierce    = 3;

	// sequential burst state (Axe only; DeathSpiral fires simultaneously)
	int32 PendingAxeShots      = 0;
	float AxeBurstTimer        = 0.f;
	int32 BurstShotsFiredCount = 0;  // which shot index we're on (for HorizScale)
	// burst snapshot
	float BurstDamage    = 0.f;
	float BurstSpeed     = 0.f;
	float BurstRadius    = 0.f;
	float BurstArcHeight = 0.f;
	float BurstHorizDir  = 1.f;
	int32 BurstPierce    = 3;

	void CacheParams();
	void StartBurst();
	void SpawnAxeShot();
};
