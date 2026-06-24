#pragma once
// このファイルは UObject 系ヘッダーをインクルードしてはならない。
#include "CoreMinimal.h"
#include "Survivors/Logic/Weapons/SurvivorsWeaponBaseF.h"

class REINBALANCE_API FSurvivorsFireWandWeapon : public FSurvivorsWeaponBase
{
public:
	virtual void Tick(float Dt) override;
	virtual void OnLevelChanged(FWeaponLevel NewLevel) override;
	virtual void ComputeHits(FSurvivorsHitFrame& HitFrame) override;

private:
	float CachedDamage          = 40.f;
	float CachedCooldown        = 1.40f;
	float CachedSpeed           = 360.f;
	float CachedExplosionRadius = 30.f;
	int32 CachedAmount          = 4;

	void CacheParams();
};
