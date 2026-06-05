#pragma once
#include "CoreMinimal.h"
#include "Survivors/Logic/Weapons/SurvivorsWeaponBase.h"
#include "SurvivorsSantaWaterWeapon.generated.h"

/**
 * SantaWater / LaBorra: 指定位置に GroundZone を生成する武器
 */
UCLASS()
class REINBALANCE_API USurvivorsSantaWaterWeapon : public USurvivorsWeaponBase
{
	GENERATED_BODY()
public:
	virtual void Tick(float Dt) override;
	virtual void OnLevelChanged(FWeaponLevel NewLevel) override;

private:
	float CachedDamage    = 10.f;
	float CachedCooldown  = 2.00f;
	float CachedRadius    = 30.f;
	float CachedDuration  = 3.0f;
	int32 CachedAmount    = 1;

	void CacheParams();
};
