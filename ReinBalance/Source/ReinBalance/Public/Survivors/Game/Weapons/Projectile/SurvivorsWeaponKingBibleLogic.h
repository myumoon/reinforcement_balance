#pragma once
// このファイルは UObject 系ヘッダーをインクルードしてはならない。
#include "CoreMinimal.h"
#include "Survivors/Game/Weapons/SurvivorsWeaponLogic.h"
#include "Survivors/Game/SurvivorsGameConstants.h"

class REINBALANCE_API FSurvivorsWeaponKingBibleLogic : public FSurvivorsWeaponLogic
{
public:
	virtual void Tick(float Dt) override;
	virtual void OnLevelChanged(FWeaponLevel NewLevel) override;
	virtual void ComputeHits(FSurvivorsHitFrame& HitFrame) override;

	virtual int32     GetOrbitOrbCount()           const override { return OrbPositions.Num(); }
	virtual FVector2D GetOrbitOrbPos(int32 OrbIdx) const override { return OrbPositions.IsValidIndex(OrbIdx) ? OrbPositions[OrbIdx] : FVector2D::ZeroVector; }
	virtual float     GetOrbitOrbVisualRadius()    const override { return OrbVisualRadius; }

private:
	float CachedDamage            = 10.f;
	float CachedCooldown          = 3.f;
	float CachedDuration          = 3.f;
	float CachedOrbitRadius       = 50.f;
	int32 CachedAmount            = 1;
	float CachedRotSpeed          = 4.0f;
	float CachedKnockbackStrength = SurvivorsGameConstants::KnockbackSim_1;

	float MasterAngle = 0.f;
	float ActiveTimer = 0.f;
	bool  bOrbsActive = false;

	TArray<FVector2D> OrbPositions;

	static constexpr float OrbVisualRadius = 10.f;

	void CacheParams();
	void RebuildOrbPositions();
	void ActivateOrbs(const FPassiveEffects& PE);
};
