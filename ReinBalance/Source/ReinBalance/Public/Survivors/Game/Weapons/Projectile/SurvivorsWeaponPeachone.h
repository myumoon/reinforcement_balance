#pragma once
#include "CoreMinimal.h"
#include "Survivors/Game/Weapons/SurvivorsWeaponBase.h"
#include "SurvivorsWeaponPeachone.generated.h"

/**
 * Peachone: 周回する target zone 内のランダム位置へ高頻度で砲撃弾を発射。
 * CD 毎に activation が開始し、Amount × PeachoneSetsPerActivation 発を
 * PeachoneProjectileInterval(0.025s) 間隔で順次発射する（wiki・動画由来）。
 */
UCLASS()
class REINBALANCE_API USurvivorsWeaponPeachone : public USurvivorsWeaponBase
{
	GENERATED_BODY()
public:
	virtual void Tick(float Dt) override;
	virtual void OnLevelChanged(FWeaponLevel NewLevel) override;

	// ---- 軌道オーブ View API ----
	virtual int32     GetOrbitOrbCount()           const override { return 1; }
	virtual FVector2D GetOrbitOrbPos(int32 OrbIdx) const override { return OrbIdx == 0 ? CurrentOrbitPos : FVector2D::ZeroVector; }
	// target-zone radius は固定（Area不使用）
	virtual float     GetOrbitOrbVisualRadius()    const override { return CachedTargetZoneRadius; }

protected:
	float CachedDamage           = 10.f;
	float CachedCooldown         = 1.0f;
	float CachedOrbitRadius      = 168.f;
	float CachedOrbitRotSpeed    = 0.8f;
	float CachedTargetZoneRadius = 49.f;  // fixed, not area-scaled
	float CachedImpactRadius     = 4.5f;  // area-scaled (passive + level)
	int32 CachedAmount           = 4;

	// 1 = 時計回り / -1 = 反時計回り（EbonyWings で上書き）
	float RotDir   = 1.f;
	// 初期位相オフセット（EbonyWings は π ずらす）
	float PhaseOff = 0.f;

	FVector2D CurrentOrbitPos;
	float     OrbitAngle = 0.f;

	// 砲撃バースト状態
	int32 PendingBombShots  = 0;
	float BombShotTimer     = 0.f;
	float BurstDamage            = 0.f;
	float BurstImpactRadius      = 0.f;  // passive AreaMult 適用済み
	float BurstTargetZoneRadius  = 0.f;  // fixed (Area不使用)

	virtual void CacheParams();
	void StartBombing();
	void SpawnBombShot();

private:
	void UpdateOrbitPos();
};
