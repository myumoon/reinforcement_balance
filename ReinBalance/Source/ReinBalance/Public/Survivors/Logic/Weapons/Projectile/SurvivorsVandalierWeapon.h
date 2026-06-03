#pragma once
#include "Survivors/Logic/Weapons/SurvivorsWeaponBase.h"
#include "SurvivorsVandalierWeapon.generated.h"

/** TODO(PR2): Vandalier の実装 */
UCLASS()
class REINBALANCE_API USurvivorsVandalierWeapon : public USurvivorsWeaponBase
{
	GENERATED_BODY()
public:
	virtual void Tick(float Dt) override {}
};
