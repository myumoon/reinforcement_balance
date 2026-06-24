#pragma once
#include "CoreMinimal.h"
#include "Survivors/Game/Weapons/SurvivorsWeaponBase.h"
#include "SurvivorsWeaponLightningRing.generated.h"

/**
 * LightningRing / ThunderLoop: 即時範囲攻撃（プロジェクタイル不要、ランダム敵 Amount 体に即ダメージ）
 */
UCLASS()
class REINBALANCE_API USurvivorsWeaponLightningRing : public USurvivorsWeaponBase
{
	GENERATED_BODY()
public:
	virtual void Tick(float Dt) override;
	virtual void OnLevelChanged(FWeaponLevel NewLevel) override;
	virtual void ComputeHits(USurvivorsCollisionComponent* CollComp, FSurvivorsHitFrame& HitFrame) override;

private:
	float CachedDamage   = 40.f;
	float CachedCooldown = 1.00f;
	float CachedRadius   = 30.f;
	int32 CachedAmount   = 1;

	// クールダウン終了時に発動フラグを Tick でセット
	bool bPendingFire = false;

	void CacheParams();
};
