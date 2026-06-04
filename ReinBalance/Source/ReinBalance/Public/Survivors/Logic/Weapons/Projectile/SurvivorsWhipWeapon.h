#pragma once
#include "CoreMinimal.h"
#include "Survivors/Logic/Weapons/SurvivorsWeaponBase.h"
#include "SurvivorsWhipWeapon.generated.h"

/**
 * Whip / BloodyTear: piercing 横線攻撃（左右 2 方向へ矩形プロジェクタイルを発射）
 */
UCLASS()
class REINBALANCE_API USurvivorsWhipWeapon : public USurvivorsWeaponBase
{
	GENERATED_BODY()
public:
	virtual void Tick(float Dt) override;
	virtual void OnLevelChanged(FWeaponLevel NewLevel) override;
	virtual void ComputeHits(USurvivorsCollisionComponent* CollComp, FSurvivorsHitFrame& HitFrame) override;

private:
	float CachedDamage   = 10.f;
	float CachedCooldown = 1.50f;
	float CachedWidth    = 50.f;
	float CachedHeight   = 15.f;

	void CacheParams();
};
