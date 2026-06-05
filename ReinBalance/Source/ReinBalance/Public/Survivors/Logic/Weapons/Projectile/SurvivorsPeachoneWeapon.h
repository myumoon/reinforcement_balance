#pragma once
#include "CoreMinimal.h"
#include "Survivors/Logic/Weapons/SurvivorsWeaponBase.h"
#include "SurvivorsPeachoneWeapon.generated.h"

/**
 * Peachone: 周回軌道から爆弾ドロップ（CD 毎に現在位置から爆発半径ダメージ）
 * 時計回り回転。
 */
UCLASS()
class REINBALANCE_API USurvivorsPeachoneWeapon : public USurvivorsWeaponBase
{
	GENERATED_BODY()
public:
	virtual void Tick(float Dt) override;
	virtual void OnLevelChanged(FWeaponLevel NewLevel) override;
	virtual void ComputeHits(USurvivorsCollisionComponent* CollComp, FSurvivorsHitFrame& HitFrame) override;

protected:
	float CachedDamage      = 25.f;
	float CachedCooldown    = 3.0f;
	float CachedOrbitRadius = 60.f;
	float CachedBombRadius  = 30.f;

	// 1 = 時計回り / -1 = 反時計回り（EbonyWings で上書き）
	float RotDir   = 1.f;
	// 初期位相オフセット（EbonyWings は π ずらす）
	float PhaseOff = 0.f;

	FVector2D CurrentOrbitPos;
	float     OrbitAngle = 0.f;

	bool bPendingFire = false;

	virtual void CacheParams();

private:
	void UpdateOrbitPos();
};
