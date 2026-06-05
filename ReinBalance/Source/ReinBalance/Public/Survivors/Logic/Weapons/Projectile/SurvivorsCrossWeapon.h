#pragma once
#include "CoreMinimal.h"
#include "Survivors/Logic/Weapons/SurvivorsWeaponBase.h"
#include "SurvivorsCrossWeapon.generated.h"

/**
 * Cross / HeavenSword: ブーメラン（LifeTime 半分消費時に折り返す）
 */
UCLASS()
class REINBALANCE_API USurvivorsCrossWeapon : public USurvivorsWeaponBase
{
	GENERATED_BODY()
public:
	virtual void Tick(float Dt) override;
	virtual void OnLevelChanged(FWeaponLevel NewLevel) override;

private:
	float CachedDamage   = 50.f;
	float CachedCooldown = 1.50f;
	float CachedSpeed    = 160.f;

	void CacheParams();
};
