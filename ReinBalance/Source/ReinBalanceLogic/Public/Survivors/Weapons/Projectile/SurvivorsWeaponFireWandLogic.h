#pragma once
// このファイルは UObject 系ヘッダーをインクルードしてはならない。
#include "CoreMinimal.h"
#include "Survivors/Weapons/SurvivorsWeaponLogic.h"

class REINBALANCELOGIC_API FSurvivorsWeaponFireWandLogic : public FSurvivorsWeaponLogic
{
public:
	virtual void Tick(float Dt) override;
	virtual void OnLevelChanged(FWeaponLevel NewLevel) override;
	virtual void ComputeHits(FSurvivorsHitFrame& HitFrame) override;
	virtual float GetCooldownObsDenominator() const override;

private:
	float CachedDamage          = 40.f;
	float CachedCooldown        = 1.40f;
	float CachedSpeed           = 360.f;
	float CachedExplosionRadius = 30.f;
	int32 CachedAmount          = 4;

	void CacheParams();
};
