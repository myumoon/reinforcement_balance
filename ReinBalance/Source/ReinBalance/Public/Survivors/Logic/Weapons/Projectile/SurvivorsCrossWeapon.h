#pragma once
#include "Survivors/Logic/Weapons/SurvivorsWeaponBase.h"
#include "SurvivorsCrossWeapon.generated.h"

/** TODO(PR2): Cross / HeavenSword の実装 */
UCLASS()
class REINBALANCE_API USurvivorsCrossWeapon : public USurvivorsWeaponBase
{
	GENERATED_BODY()
public:
	virtual void Tick(float Dt) override {}
};
