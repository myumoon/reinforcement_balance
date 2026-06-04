#pragma once
#include "Survivors/Logic/Weapons/SurvivorsWeaponBase.h"
#include "SurvivorsSantaWaterWeapon.generated.h"

/** TODO(PR2): SantaWater / LaBorra の実装 */
UCLASS()
class REINBALANCE_API USurvivorsSantaWaterWeapon : public USurvivorsWeaponBase
{
	GENERATED_BODY()
public:
	virtual void Tick(float Dt) override {}
};
