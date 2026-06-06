#pragma once
#include "CoreMinimal.h"
#include "Survivors/Logic/SurvivorsGameConstants.h"
#include "Survivors/Logic/Weapons/SurvivorsWeaponBase.h"
#include "SurvivorsKingBibleWeapon.generated.h"

/**
 * KingBible / UnholyVespers: 周回オーブ（Amount 個が等間隔でプレイヤーを周回し接触ダメージ）
 *
 * 設計方針:
 *   - オーブは Projectiles リストに登録しない（ComputeProjectileHits との二重当たりを防ぐ）
 *   - Tick() でオーブ位置を OrbPositions キャッシュに更新
 *   - ComputeHits() が OrbPositions を使って当たり判定（WeaponLastHitTime でヒット間隔管理）
 *   - View 描画は GetOrbPositions() / GetOrbRadius() を使う
 */
UCLASS()
class REINBALANCE_API USurvivorsKingBibleWeapon : public USurvivorsWeaponBase
{
	GENERATED_BODY()
public:
	virtual void Tick(float Dt) override;
	virtual void OnLevelChanged(FWeaponLevel NewLevel) override;
	virtual void ComputeHits(USurvivorsCollisionComponent* CollComp, FSurvivorsHitFrame& HitFrame) override;

	/** View 向けアクセサ: オーブのワールド位置一覧 */
	const TArray<FVector2D>& GetOrbPositions() const { return OrbPositions; }

	/** View 向けアクセサ: オーブ当たり半径 */
	static constexpr float GetOrbRadius() { return OrbVisualRadius; }

private:
	float CachedDamage      = 10.f;
	float CachedCooldown    = 3.f;
	float CachedDuration    = 3.f;
	float CachedOrbitRadius = 50.f;
	int32 CachedAmount      = 1;
	float CachedRotSpeed    = 2.0f;  // rad/sec
	float CachedKnockbackStrength = SurvivorsGameConstants::KnockbackSim_1;

	// 基準軌道角度（Tick で更新）
	float MasterAngle = 0.f;

	float ActiveTimer = 0.f;
	bool bOrbsActive = false;

	// オーブ位置キャッシュ（Tick で更新、ComputeHits で参照）
	TArray<FVector2D> OrbPositions;

	// ヒット間隔: 仕様 Hitbox Delay=1.7s（同一オーブが同一敵を再ヒットするまでの間隔）
	static constexpr float OrbHitInterval  = 1.7f;
	static constexpr float OrbVisualRadius = 10.f;

	void CacheParams();
	void RebuildOrbProjectiles();
	void ActivateOrbs(const FPassiveEffects& PE);
};
