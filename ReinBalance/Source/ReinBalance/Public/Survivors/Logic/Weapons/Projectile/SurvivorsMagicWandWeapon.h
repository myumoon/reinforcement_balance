#pragma once
#include "CoreMinimal.h"
#include "Survivors/Logic/Weapons/SurvivorsWeaponBase.h"
#include "SurvivorsMagicWandWeapon.generated.h"

/**
 * MagicWand / HolyWand: 最近傍敵へ Amount 本の弾を 0.1s 間隔で順次発射。
 * 各弾はスポーン時点で最近傍の敵を個別に狙う（wiki: projectile interval = 0.1s）。
 */
UCLASS()
class REINBALANCE_API USurvivorsMagicWandWeapon : public USurvivorsWeaponBase
{
	GENERATED_BODY()
public:
	virtual void Tick(float Dt) override;
	virtual void OnLevelChanged(FWeaponLevel NewLevel) override;

private:
	float CachedDamage   = 10.f;
	float CachedCooldown = 1.20f;
	float CachedSpeed    = 600.f;
	int32 CachedAmount   = 1;
	int32 CachedPierce   = 1;

	// sequential shot state
	int32 PendingWandShots = 0;
	float WandBurstTimer   = 0.f;
	// burst snapshot (captured when burst starts)
	float BurstDamage   = 0.f;
	float BurstSpeed    = 0.f;
	float BurstRadius   = 0.f;
	float BurstLifeTime = 0.f;
	int32 BurstPierce   = 1;

	void CacheParams();
	void StartBurst();
	void SpawnWandShot();
};
