#pragma once
#include "CoreMinimal.h"
#include "Survivors/Logic/Weapons/SurvivorsWeaponBase.h"
#include "SurvivorsRunetracerWeapon.generated.h"

/**
 * Runetracer / NoFuture: バウンス弾（壁ヒット時に反射、MaxBounce 回まで）。
 * Runetracer は 0.2s 間隔の順次発射、各弾がランダム方向（wiki/OBSERVED）。
 * NoFuture は Amount=1 のため事実上即時発射と同等。
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
	float CachedSpeed     = 440.f;
	float CachedDuration  = 2.25f;
	int32 CachedAmount    = 1;
	int32 CachedMaxBounce = 3;

	// sequential burst state
	int32 PendingRuneShots  = 0;
	float RuneBurstTimer    = 0.f;
	// burst snapshot
	float BurstDamage    = 0.f;
	float BurstSpeed     = 0.f;
	float BurstDuration  = 0.f;
	float BurstRadius    = 0.f;
	int32 BurstMaxBounce = 3;

	void CacheParams();
	void StartBurst();
	void SpawnRuneShot();
};
