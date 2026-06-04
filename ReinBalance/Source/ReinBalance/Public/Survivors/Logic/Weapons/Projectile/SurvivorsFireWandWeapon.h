#pragma once
#include "Survivors/Logic/Weapons/SurvivorsWeaponBase.h"
#include "SurvivorsFireWandWeapon.generated.h"

/** TODO(PR2): FireWand / Hellfire の実装 */
UCLASS()
class REINBALANCE_API USurvivorsFireWandWeapon : public USurvivorsWeaponBase
{
	GENERATED_BODY()
public:
	virtual void Tick(float Dt) override {}
};
