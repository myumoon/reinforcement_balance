#pragma once
// このファイルは UObject 系ヘッダーをインクルードしてはならない。
#include "CoreMinimal.h"
#include "Survivors/Logic/Weapons/SurvivorsWeaponBaseF.h"

class REINBALANCE_API FSurvivorsLaurelWeapon : public FSurvivorsWeaponBase
{
public:
	virtual void Tick(float Dt) override;
	virtual void OnLevelChanged(FWeaponLevel NewLevel) override;

private:
	float CachedShieldDuration = 1.0f;
	float CachedCooldown       = 8.0f;

	void CacheParams();
};
