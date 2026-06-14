#pragma once
#include "CoreMinimal.h"
#include "Survivors/Logic/Weapons/SurvivorsWeaponBase.h"
#include "SurvivorsSantaWaterWeapon.generated.h"

/**
 * SantaWater / LaBorra: 0.3s 間隔で順次 GroundZone を生成する武器。
 * Amount < 4: 最近傍敵へ落下。Amount >= 4: プレイヤー周囲の時計回り円形配置（wiki由来）。
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

	// sequential drop state
	TArray<FVector2D> PendingDropPositions;
	float DropTimer      = 0.f;
	// burst snapshot
	float BurstDamage    = 0.f;
	float BurstRadius    = 0.f;
	float BurstDuration  = 0.f;

	void CacheParams();
	void StartDropSequence();
	void SpawnDrop(FVector2D DropPos);
};
