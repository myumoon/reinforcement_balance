#pragma once
#include "CoreMinimal.h"
#include "Survivors/Game/Weapons/SurvivorsWeaponBase.h"
#include "SurvivorsPentagramWeapon.generated.h"

/**
 * Pentagram / GorgeousMoon: 全敵への即死攻撃（プロジェクタイル不要、一定 CD で全敵に大ダメージ）
 */
UCLASS()
class REINBALANCE_API USurvivorsPentagramWeapon : public USurvivorsWeaponBase
{
	GENERATED_BODY()
public:
	virtual void Tick(float Dt) override;
	virtual void OnLevelChanged(FWeaponLevel NewLevel) override;
	virtual void ComputeHits(USurvivorsCollisionComponent* CollComp, FSurvivorsHitFrame& HitFrame) override;

private:
	float CachedDamage   = 999.f;
	float CachedCooldown = 15.0f;
	float CachedRadius   = 9999.f;

	bool bPendingFire = false;

	void CacheParams();
};
