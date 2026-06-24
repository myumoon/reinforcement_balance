#pragma once
// このファイルは UObject 系ヘッダーをインクルードしてはならない。
#include "CoreMinimal.h"
#include "Survivors/Logic/Weapons/SurvivorsWeaponLogic.h"

class REINBALANCELOGIC_API FSurvivorsWeaponSantaWaterLogic : public FSurvivorsWeaponLogic
{
public:
	virtual void Tick(float Dt) override;
	virtual void OnLevelChanged(FWeaponLevel NewLevel) override;

private:
	float CachedDamage   = 10.f;
	float CachedCooldown = 2.00f;
	float CachedRadius   = 30.f;
	float CachedDuration = 3.0f;
	int32 CachedAmount   = 1;

	TArray<FVector2D> PendingDropPositions;
	float DropTimer    = 0.f;

	float BurstDamage   = 0.f;
	float BurstRadius   = 0.f;
	float BurstDuration = 0.f;

	void CacheParams();
	void StartDropSequence();
	void SpawnDrop(FVector2D DropPos);
};
