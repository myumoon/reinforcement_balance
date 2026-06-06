#pragma once
#include "CoreMinimal.h"
#include "Survivors/Logic/Weapons/SurvivorsWeaponBase.h"
#include "SurvivorsRunetracerWeapon.generated.h"

/**
 * Runetracer / NoFuture: バウンス弾（壁ヒット時に反射、MaxBounce 回まで）
 */
UCLASS()
class REINBALANCE_API USurvivorsRunetracerWeapon : public USurvivorsWeaponBase
{
	GENERATED_BODY()
public:
	virtual void Tick(float Dt) override;
	virtual void OnLevelChanged(FWeaponLevel NewLevel) override;

private:
	float CachedDamage    = 15.f;
	float CachedCooldown  = 0.60f;
	float CachedSpeed     = 220.f;
	float CachedDuration  = 2.25f;
	int32 CachedAmount    = 1;
	int32 CachedMaxBounce = 3;

	void CacheParams();
};
