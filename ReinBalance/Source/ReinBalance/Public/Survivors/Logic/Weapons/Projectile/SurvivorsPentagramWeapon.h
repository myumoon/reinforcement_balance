#pragma once
#include "Survivors/Logic/Weapons/SurvivorsWeaponBase.h"
#include "SurvivorsPentagramWeapon.generated.h"

/** TODO(PR2): Pentagram / GorgeousMoon の実装 */
UCLASS()
class REINBALANCE_API USurvivorsPentagramWeapon : public USurvivorsWeaponBase
{
	GENERATED_BODY()
public:
	virtual void Tick(float Dt) override {}
};
