#pragma once
// このファイルは UObject 系ヘッダーをインクルードしてはならない。
#include "CoreMinimal.h"
#include "Survivors/Weapons/SurvivorsWeaponLogic.h"

class REINBALANCELOGIC_API FSurvivorsWeaponLightningRingLogic : public FSurvivorsWeaponLogic
{
public:
	virtual void Tick(float Dt) override;
	virtual void OnLevelChanged(FWeaponLevel NewLevel) override;
	virtual void ComputeHits(FSurvivorsHitFrame& HitFrame) override;

private:
	float CachedDamage   = 40.f;
	float CachedCooldown = 1.00f;
	float CachedRadius   = 30.f;
	int32 CachedAmount   = 1;

	bool bPendingFire = false;

	void CacheParams();
};
