#pragma once
// このファイルは UObject 系ヘッダーをインクルードしてはならない。
#include "CoreMinimal.h"
#include "Survivors/Logic/Weapons/SurvivorsWeaponLogic.h"

class REINBALANCELOGIC_API FSurvivorsWeaponPeachoneLogic : public FSurvivorsWeaponLogic
{
public:
	virtual void Tick(float Dt) override;
	virtual void OnLevelChanged(FWeaponLevel NewLevel) override;

	virtual int32     GetOrbitOrbCount()           const override { return 1; }
	virtual FVector2D GetOrbitOrbPos(int32 OrbIdx) const override { return OrbIdx == 0 ? CurrentOrbitPos : FVector2D::ZeroVector; }
	virtual float     GetOrbitOrbVisualRadius()    const override { return CachedTargetZoneRadius; }

protected:
	float CachedDamage           = 10.f;
	float CachedCooldown         = 1.0f;
	float CachedOrbitRadius      = 168.f;
	float CachedOrbitRotSpeed    = 0.8f;
	float CachedTargetZoneRadius = 49.f;
	float CachedImpactRadius     = 4.5f;
	int32 CachedAmount           = 4;

	float RotDir   = 1.f;
	float PhaseOff = 0.f;

	FVector2D CurrentOrbitPos;
	float     OrbitAngle = 0.f;

	virtual void CacheParams();
	void UpdateOrbitPos();

private:
	int32 PendingBombShots  = 0;
	float BombShotTimer     = 0.f;
	float BurstDamage       = 0.f;
	float BurstImpactRadius = 0.f;
	float BurstTargetZoneRadius = 0.f;

	void StartBombing();
	void SpawnBombShot();
};
