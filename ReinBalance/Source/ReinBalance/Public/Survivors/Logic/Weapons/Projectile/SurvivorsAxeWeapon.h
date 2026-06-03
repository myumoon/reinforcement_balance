#pragma once
#include "Survivors/Logic/Weapons/SurvivorsWeaponBase.h"
#include "SurvivorsAxeWeapon.generated.h"

/** TODO(PR2): Axe / DeathSpiral の実装 */
UCLASS()
class REINBALANCE_API USurvivorsAxeWeapon : public USurvivorsWeaponBase
{
	GENERATED_BODY()
public:
	virtual void Tick(float Dt) override {}
};
