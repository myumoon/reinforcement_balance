#pragma once
#include "CoreMinimal.h"
#include "Survivors/Logic/Weapons/SurvivorsWeaponBase.h"
#include "SurvivorsKingBibleWeapon.generated.h"

/**
 * KingBible / UnholyVespers: 周回オーブ（Amount 個が等間隔でプレイヤーを周回し接触ダメージ）
 */
UCLASS()
class REINBALANCE_API USurvivorsKingBibleWeapon : public USurvivorsWeaponBase
{
	GENERATED_BODY()
public:
	virtual void Tick(float Dt) override;
	virtual void OnLevelChanged(FWeaponLevel NewLevel) override;
	virtual void ComputeHits(USurvivorsCollisionComponent* CollComp, FSurvivorsHitFrame& HitFrame) override;

private:
	float CachedDamage      = 10.f;
	float CachedOrbitRadius = 50.f;
	int32 CachedAmount      = 1;
	float CachedRotSpeed    = 2.0f;  // rad/sec

	// 基準軌道角度（Tick で更新）
	float MasterAngle = 0.f;

	// ヒット間隔（オーブの高頻度当たり判定を制限）
	static constexpr float OrbHitInterval = 0.5f;

	void CacheParams();
	void RebuildOrbProjectiles();
};
