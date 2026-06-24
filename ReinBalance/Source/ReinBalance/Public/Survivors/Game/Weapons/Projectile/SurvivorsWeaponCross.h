#pragma once
#include "CoreMinimal.h"
#include "Survivors/Logic/SurvivorsGameConstants.h"
#include "Survivors/Game/Weapons/SurvivorsWeaponBase.h"
#include "SurvivorsWeaponCross.generated.h"

/**
 * Cross / HeavenSword: ブーメラン（0.1s 間隔の順次発射、各弾が発射時点で最近傍敵を再評価）。
 * 折り返し後も寿命まで継続移動し、次 volley が来ても消えない。
 * wiki: Projectile Interval = 0.1s、Amount > 1 は fan spread なし。
 */
UCLASS()
class REINBALANCE_API USurvivorsWeaponCross : public USurvivorsWeaponBase
{
	GENERATED_BODY()
public:
	virtual void Tick(float Dt) override;
	virtual void OnLevelChanged(FWeaponLevel NewLevel) override;

private:
	float CachedDamage   = 50.f;
	float CachedCooldown = 1.50f;
	float CachedSpeed    = 320.f;
	float CachedRadius   = 12.f;
	int32 CachedAmount   = 1;
	float CachedKnockbackStrength = SurvivorsGameConstants::KnockbackSim_1;

	// sequential burst state
	int32 PendingCrossShots  = 0;
	float CrossBurstTimer    = 0.f;
	// burst snapshot
	float BurstDamage        = 0.f;
	float BurstSpeed         = 0.f;
	float BurstRadius        = 0.f;
	float BurstLifeTime      = 0.f;
	float BurstReverseTime   = 0.f;
	float BurstKnockback     = SurvivorsGameConstants::KnockbackSim_1;

	void CacheParams();
	void StartBurst();
	void SpawnCrossShot();
};
