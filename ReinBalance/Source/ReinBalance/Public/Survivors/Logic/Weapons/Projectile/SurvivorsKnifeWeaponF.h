#pragma once
// このファイルは UObject 系ヘッダーをインクルードしてはならない。
#include "CoreMinimal.h"
#include "Survivors/Logic/Weapons/SurvivorsWeaponBaseF.h"

class REINBALANCE_API FSurvivorsKnifeWeapon : public FSurvivorsWeaponBase
{
public:
	virtual void Tick(float Dt) override;
	virtual void OnLevelChanged(FWeaponLevel NewLevel) override;

private:
	float CachedDamage   = 6.5f;
	float CachedCooldown = 1.00f;
	float CachedSpeed    = 800.f;
	int32 CachedAmount   = 1;
	int32 CachedPierce   = 1;

	int32     PendingKnifeShots = 0;
	float     KnifeBurstTimer   = 0.f;
	FVector2D LastFacingDir     = FVector2D(1.f, 0.f);

	float BurstDamage   = 6.5f;
	float BurstSpeed    = 800.f;
	float BurstRadius   = 6.f;
	float BurstLifeTime = 1.5f;
	int32 BurstPierce   = 1;

	void CacheParams();
	void StartBurst();
	void SpawnKnifeShot();
};
