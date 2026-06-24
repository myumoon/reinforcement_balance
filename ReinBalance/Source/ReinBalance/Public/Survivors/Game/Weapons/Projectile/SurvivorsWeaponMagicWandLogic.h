#pragma once
// このファイルは UObject 系ヘッダーをインクルードしてはならない。
#include "CoreMinimal.h"
#include "Survivors/Game/Weapons/SurvivorsWeaponLogic.h"

class REINBALANCE_API FSurvivorsWeaponMagicWandLogic : public FSurvivorsWeaponLogic
{
public:
	virtual void Tick(float Dt) override;
	virtual void OnLevelChanged(FWeaponLevel NewLevel) override;

private:
	float CachedDamage   = 10.f;
	float CachedCooldown = 1.20f;
	float CachedSpeed    = 600.f;
	int32 CachedAmount   = 1;
	int32 CachedPierce   = 1;

	int32     PendingWandShots = 0;
	float     WandBurstTimer   = 0.f;
	FVector2D BurstNoTargetDir = FVector2D(1.f, 0.f);

	float BurstDamage   = 0.f;
	float BurstSpeed    = 0.f;
	float BurstRadius   = 0.f;
	float BurstLifeTime = 0.f;
	int32 BurstPierce   = 1;

	void CacheParams();
	void StartBurst();
	void SpawnWandShot();
};
