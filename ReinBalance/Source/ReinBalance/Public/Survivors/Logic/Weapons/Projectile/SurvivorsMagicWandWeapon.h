#pragma once
#include "CoreMinimal.h"
#include "Survivors/Logic/Weapons/SurvivorsWeaponBase.h"
#include "SurvivorsMagicWandWeapon.generated.h"

/**
 * MagicWand / HolyWand: 最近傍敵へ Amount 本の追尾弾を発射
 */
UCLASS()
class REINBALANCE_API USurvivorsMagicWandWeapon : public USurvivorsWeaponBase
{
	GENERATED_BODY()
public:
	virtual void Tick(float Dt) override;
	virtual void OnLevelChanged(FWeaponLevel NewLevel) override;

private:
	float CachedDamage   = 20.f;
	float CachedCooldown = 0.50f;
	float CachedSpeed    = 300.f;
	int32 CachedAmount   = 1;

	void CacheParams();
};
