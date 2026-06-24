#pragma once
// このファイルは UObject 系ヘッダーをインクルードしてはならない。
#include "CoreMinimal.h"
#include "Survivors/Game/Weapons/Projectile/SurvivorsWeaponPeachoneLogic.h"

class REINBALANCE_API FSurvivorsWeaponEbonyWingsLogic : public FSurvivorsWeaponPeachoneLogic
{
public:
	virtual void OnLevelChanged(FWeaponLevel NewLevel) override;

protected:
	virtual void CacheParams() override;
};
