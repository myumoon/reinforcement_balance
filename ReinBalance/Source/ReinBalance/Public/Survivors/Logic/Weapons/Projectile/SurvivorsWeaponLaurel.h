#pragma once
#include "CoreMinimal.h"
#include "Survivors/Logic/Weapons/SurvivorsWeaponBase.h"
#include "SurvivorsWeaponLaurel.generated.h"

/**
 * Laurel: シールド付与（PlayerShieldTimer / bShieldActive を ASurvivorsGame に設定）
 */
UCLASS()
class REINBALANCE_API USurvivorsWeaponLaurel : public USurvivorsWeaponBase
{
	GENERATED_BODY()
public:
	virtual void Tick(float Dt) override;
	virtual void OnLevelChanged(FWeaponLevel NewLevel) override;

private:
	float CachedShieldDuration = 1.0f;
	float CachedCooldown       = 8.0f;

	void CacheParams();
};
