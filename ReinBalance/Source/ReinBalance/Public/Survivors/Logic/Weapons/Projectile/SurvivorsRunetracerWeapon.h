#pragma once
#include "Survivors/Logic/Weapons/SurvivorsWeaponBase.h"
#include "SurvivorsRunetracerWeapon.generated.h"

/** TODO(PR2): Runetracer / NoFuture の実装 */
UCLASS()
class REINBALANCE_API USurvivorsRunetracerWeapon : public USurvivorsWeaponBase
{
	GENERATED_BODY()
public:
	virtual void Tick(float Dt) override {}
};
