#pragma once
#include "Survivors/Logic/Weapons/SurvivorsWeaponBase.h"
#include "SurvivorsLightningRingWeapon.generated.h"

/** TODO(PR2): LightningRing / ThunderLoop の実装 */
UCLASS()
class REINBALANCE_API USurvivorsLightningRingWeapon : public USurvivorsWeaponBase
{
	GENERATED_BODY()
public:
	virtual void Tick(float Dt) override {}
};
