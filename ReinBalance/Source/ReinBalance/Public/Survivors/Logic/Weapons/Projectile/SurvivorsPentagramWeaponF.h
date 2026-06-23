#pragma once
// このファイルは UObject 系ヘッダーをインクルードしてはならない。
#include "CoreMinimal.h"
#include "Survivors/Logic/Weapons/SurvivorsWeaponBaseF.h"

class REINBALANCE_API FSurvivorsPentagramWeapon : public FSurvivorsWeaponBase
{
public:
	virtual void Tick(float Dt) override;
	virtual void OnLevelChanged(FWeaponLevel NewLevel) override;
	virtual void ComputeHits(FSurvivorsHitFrame& HitFrame) override;

private:
	float CachedDamage   = 999.f;
	float CachedCooldown = 15.0f;
	float CachedRadius   = 9999.f;

	bool bPendingFire = false;

	void CacheParams();
};
