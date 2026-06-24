#pragma once
// このファイルは UObject 系ヘッダーをインクルードしてはならない。
#include "CoreMinimal.h"
#include "Survivors/Logic/Weapons/SurvivorsWeaponLogic.h"

class REINBALANCELOGIC_API FSurvivorsWeaponRunetracerLogic : public FSurvivorsWeaponLogic
{
public:
	virtual void Tick(float Dt) override;
	virtual void OnLevelChanged(FWeaponLevel NewLevel) override;

private:
	float CachedDamage    = 15.f;
	float CachedCooldown  = 0.60f;
	float CachedSpeed     = 440.f;
	float CachedDuration  = 2.25f;
	int32 CachedAmount    = 1;
	int32 CachedMaxBounce = 3;

	int32 PendingRuneShots = 0;
	float RuneBurstTimer   = 0.f;

	float BurstDamage    = 0.f;
	float BurstSpeed     = 0.f;
	float BurstDuration  = 0.f;
	float BurstRadius    = 0.f;
	int32 BurstMaxBounce = 3;

	void CacheParams();
	void StartBurst();
	void SpawnRuneShot();
};
