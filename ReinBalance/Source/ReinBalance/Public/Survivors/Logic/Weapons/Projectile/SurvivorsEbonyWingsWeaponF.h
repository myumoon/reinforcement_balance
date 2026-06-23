#pragma once
// このファイルは UObject 系ヘッダーをインクルードしてはならない。
#include "CoreMinimal.h"
#include "Survivors/Logic/Weapons/Projectile/SurvivorsPeachoneWeaponF.h"

class REINBALANCE_API FSurvivorsEbonyWingsWeapon : public FSurvivorsPeachoneWeapon
{
public:
	virtual void OnLevelChanged(FWeaponLevel NewLevel) override;

protected:
	virtual void CacheParams() override;
};
