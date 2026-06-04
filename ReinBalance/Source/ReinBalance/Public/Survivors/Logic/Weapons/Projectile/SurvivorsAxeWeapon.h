#pragma once
#include "CoreMinimal.h"
#include "Survivors/Logic/Weapons/SurvivorsWeaponBase.h"
#include "SurvivorsAxeWeapon.generated.h"

/**
 * Axe / DeathSpiral: 上方向弧弾（重力あり・放物線弾道）
 */
UCLASS()
class REINBALANCE_API USurvivorsAxeWeapon : public USurvivorsWeaponBase
{
	GENERATED_BODY()
public:
	virtual void Tick(float Dt) override;
	virtual void OnLevelChanged(FWeaponLevel NewLevel) override;

private:
	float CachedDamage    = 50.f;
	float CachedCooldown  = 1.20f;
	float CachedSpeed     = 180.f;
	float CachedArcHeight = 120.f;

	void CacheParams();
};
