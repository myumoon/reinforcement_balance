#pragma once
// このファイルは UObject 系ヘッダーをインクルードしてはならない。
#include "CoreMinimal.h"
#include "Survivors/Weapons/SurvivorsWeaponLogic.h"
#include "Survivors/SurvivorsGameConstants.h"

class REINBALANCELOGIC_API FSurvivorsWeaponCrossLogic : public FSurvivorsWeaponLogic
{
public:
	virtual void Tick(float Dt) override;
	virtual void OnLevelChanged(FWeaponLevel NewLevel) override;

private:
	float CachedDamage            = 50.f;
	float CachedCooldown          = 1.50f;
	float CachedSpeed             = 320.f;
	float CachedRadius            = 12.f;
	int32 CachedAmount            = 1;
	float CachedKnockbackStrength = SurvivorsGameConstants::KnockbackSim_1;

	int32 PendingCrossShots = 0;
	float CrossBurstTimer   = 0.f;

	float BurstDamage      = 0.f;
	float BurstSpeed       = 0.f;
	float BurstRadius      = 0.f;
	float BurstLifeTime    = 0.f;
	float BurstReverseTime = 0.f;
	float BurstKnockback   = SurvivorsGameConstants::KnockbackSim_1;

	void CacheParams();
	void StartBurst();
	void SpawnCrossShot();
};
