#pragma once
#include "Survivors/Logic/Weapons/SurvivorsWeaponBase.h"
#include "SurvivorsLaurelWeapon.generated.h"

/** TODO(PR2): Laurel の実装 */
UCLASS()
class REINBALANCE_API USurvivorsLaurelWeapon : public USurvivorsWeaponBase
{
	GENERATED_BODY()
public:
	virtual void Tick(float Dt) override {}
};
