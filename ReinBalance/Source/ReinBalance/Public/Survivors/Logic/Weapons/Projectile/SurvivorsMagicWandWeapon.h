#pragma once
#include "Survivors/Logic/Weapons/SurvivorsWeaponBase.h"
#include "SurvivorsMagicWandWeapon.generated.h"

/** TODO(PR2): MagicWand / HolyWand の実装 */
UCLASS()
class REINBALANCE_API USurvivorsMagicWandWeapon : public USurvivorsWeaponBase
{
	GENERATED_BODY()
public:
	virtual void Tick(float Dt) override {}
};
