#pragma once
// このファイルは UObject 系ヘッダーをインクルードしてはならない。
#include "CoreMinimal.h"
#include "Survivors/Logic/Weapons/SurvivorsWeaponLogic.h"

class REINBALANCELOGIC_API FSurvivorsWeaponVandalierLogic : public FSurvivorsWeaponLogic
{
public:
	virtual void Tick(float Dt) override;
	virtual void OnLevelChanged(FWeaponLevel NewLevel) override;

	virtual int32     GetOrbitOrbCount()           const override { return 2; }
	virtual FVector2D GetOrbitOrbPos(int32 OrbIdx) const override;
	virtual float     GetOrbitOrbVisualRadius()    const override { return CachedTargetZoneRadius; }

private:
	float CachedDamage           = 55.f;
	float CachedCooldown         = 1.6f;
	float CachedOrbitRadius      = 178.f;
	float CachedOrbitRotSpeed    = 0.8f;
	float CachedTargetZoneRadius = 59.f;
	float CachedImpactRadius     = 4.5f;
	int32 CachedAmount           = 4;

	float OrbitAngle[2]      = { 0.f, UE_HALF_PI };
	int32 PendingBombShots[2]= { 0, 0 };
	float BombShotTimer[2]   = { 0.f, 0.f };

	float BurstDamage           = 0.f;
	float BurstImpactRadius     = 0.f;
	float BurstTargetZoneRadius = 0.f;

	void CacheParams();
	void StartBombing();
	void SpawnBombShot(int32 OrbIdx);
};
