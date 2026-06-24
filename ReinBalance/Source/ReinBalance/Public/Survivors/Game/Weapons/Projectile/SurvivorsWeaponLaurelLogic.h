#pragma once
// このファイルは UObject 系ヘッダーをインクルードしてはならない。
#include "CoreMinimal.h"
#include "Survivors/Game/Weapons/SurvivorsWeaponLogic.h"

class REINBALANCE_API FSurvivorsWeaponLaurelLogic : public FSurvivorsWeaponLogic
{
public:
	virtual void Tick(float Dt) override;
	virtual void OnLevelChanged(FWeaponLevel NewLevel) override;

private:
	float CachedShieldDuration = 1.0f;
	float CachedCooldown       = 8.0f;

	void CacheParams();
};
