#pragma once
#include "Survivors/Logic/Weapons/SurvivorsWeaponBase.h"
#include "SurvivorsKingBibleWeapon.generated.h"

/** TODO(PR2): KingBible / UnholyVespers の実装 */
UCLASS()
class REINBALANCE_API USurvivorsKingBibleWeapon : public USurvivorsWeaponBase
{
	GENERATED_BODY()
public:
	virtual void Tick(float Dt) override {}
};
