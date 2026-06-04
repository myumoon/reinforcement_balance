#pragma once
#include "Survivors/Logic/Weapons/SurvivorsWeaponBase.h"
#include "SurvivorsWhipWeapon.generated.h"

/** TODO(PR2): Whip / BloodyTear の実装 */
UCLASS()
class REINBALANCE_API USurvivorsWhipWeapon : public USurvivorsWeaponBase
{
	GENERATED_BODY()
public:
	virtual void Tick(float Dt) override {}
};
