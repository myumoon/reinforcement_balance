#pragma once
// このファイルは UObject 系ヘッダーをインクルードしてはならない。
#include "CoreMinimal.h"
#include "Survivors/Weapons/Projectile/SurvivorsWeaponPeachoneLogic.h"

class REINBALANCELOGIC_API FSurvivorsWeaponEbonyWingsLogic : public FSurvivorsWeaponPeachoneLogic
{
public:
	virtual void OnLevelChanged(FWeaponLevel NewLevel) override;

protected:
	virtual void CacheParams() override;
};
