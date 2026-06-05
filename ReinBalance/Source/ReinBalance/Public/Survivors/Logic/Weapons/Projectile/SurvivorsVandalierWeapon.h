#pragma once
#include "CoreMinimal.h"
#include "Survivors/Logic/Weapons/SurvivorsWeaponBase.h"
#include "SurvivorsVandalierWeapon.generated.h"

/**
 * Vandalier: Peachone + EbonyWings Union（2 軌道を同時に持つ）
 * 内部で 2 つの軌道オーブを管理する。
 */
UCLASS()
class REINBALANCE_API USurvivorsVandalierWeapon : public USurvivorsWeaponBase
{
	GENERATED_BODY()
public:
	virtual void Tick(float Dt) override;
	virtual void OnLevelChanged(FWeaponLevel NewLevel) override;
	virtual void ComputeHits(USurvivorsCollisionComponent* CollComp, FSurvivorsHitFrame& HitFrame) override;

private:
	float CachedDamage      = 55.f;
	float CachedCooldown    = 1.6f;
	float CachedOrbitRadius = 80.f;
	float CachedBombRadius  = 50.f;

	// 2 軌道（0=時計回り, 1=反時計回り）
	float OrbitAngle[2] = { 0.f, HALF_PI };
	bool  bPendingFire  = false;

	void CacheParams();
};
