#pragma once
#include "CoreMinimal.h"
#include "Survivors/Game/Weapons/Projectile/SurvivorsPeachoneWeapon.h"
#include "SurvivorsEbonyWingsWeapon.generated.h"

/**
 * EbonyWings: Peachone と同じ挙動・反時計回り・初期角度 π オフセット
 */
UCLASS()
class REINBALANCE_API USurvivorsEbonyWingsWeapon : public USurvivorsPeachoneWeapon
{
	GENERATED_BODY()
public:
	virtual void OnLevelChanged(FWeaponLevel NewLevel) override;

protected:
	virtual void CacheParams() override;
};
