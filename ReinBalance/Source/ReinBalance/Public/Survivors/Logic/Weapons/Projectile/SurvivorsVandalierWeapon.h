#pragma once
#include "CoreMinimal.h"
#include "Survivors/Logic/Weapons/SurvivorsWeaponBase.h"
#include "SurvivorsVandalierWeapon.generated.h"

/**
 * Vandalier: Peachone + EbonyWings Union（2 軌道 target zone から同時砲撃）。
 * 各 zone が独立して PeachoneProjectileInterval 間隔で砲撃弾を発射する。
 */
UCLASS()
class REINBALANCE_API USurvivorsVandalierWeapon : public USurvivorsWeaponBase
{
	GENERATED_BODY()
public:
	virtual void Tick(float Dt) override;
	virtual void OnLevelChanged(FWeaponLevel NewLevel) override;

	// ---- 軌道オーブ View API ----
	virtual int32     GetOrbitOrbCount()           const override { return 2; }
	virtual FVector2D GetOrbitOrbPos(int32 OrbIdx) const override;
	virtual float     GetOrbitOrbVisualRadius()    const override { return CachedBombRadius * GetPassiveEffects().AreaMult; }

private:
	float CachedDamage      = 55.f;
	float CachedCooldown    = 1.6f;
	float CachedOrbitRadius = 80.f;
	float CachedBombRadius  = 50.f;
	int32 CachedAmount      = 4;

	// 2 軌道（0=時計回り, 1=反時計回り）
	float OrbitAngle[2] = { 0.f, UE_HALF_PI };

	// 砲撃バースト状態（2 zone 分）
	int32 PendingBombShots[2] = { 0, 0 };
	float BombShotTimer[2]    = { 0.f, 0.f };
	float BurstDamage         = 0.f;
	float BurstImpactRadius   = 0.f;
	float BurstBombRadius     = 0.f;

	void CacheParams();
	void StartBombing();
	void SpawnBombShot(int32 OrbIdx);
};
