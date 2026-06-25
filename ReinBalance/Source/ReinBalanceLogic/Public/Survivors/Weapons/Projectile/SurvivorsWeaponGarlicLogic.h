#pragma once
// このファイルは UObject 系ヘッダーをインクルードしてはならない。
#include "CoreMinimal.h"
#include "Survivors/Weapons/SurvivorsWeaponLogic.h"

class REINBALANCELOGIC_API FSurvivorsWeaponGarlicLogic : public FSurvivorsWeaponLogic
{
public:
	virtual void Tick(float Dt) override;
	virtual void OnLevelChanged(FWeaponLevel NewLevel) override;
	virtual void ComputeHits(FSurvivorsHitFrame& HitFrame) override;
	virtual float GetCooldownObsDenominator() const override;

private:
	float CachedDamage      = 5.f;
	float CachedHitInterval = 1.30f;
	float CachedAreaRadius  = 25.f;

	void CacheParams();
};
