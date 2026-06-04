#pragma once
#include "CoreMinimal.h"
#include "Survivors/Logic/Weapons/SurvivorsWeaponBase.h"
#include "SurvivorsFireWandWeapon.generated.h"

/**
 * FireWand / Hellfire: 爆発弾（ヒットまたは LifeTime 切れで爆発範囲ダメージ）
 * 爆発はプロジェクタイル期限切れ時に WeaponComponent の GroundZone として処理するため
 * ここでは発射のみ担当し、爆発は Tick で期限切れプロジェクタイルを検出して GroundZone 生成。
 */
UCLASS()
class REINBALANCE_API USurvivorsFireWandWeapon : public USurvivorsWeaponBase
{
	GENERATED_BODY()
public:
	virtual void Tick(float Dt) override;
	virtual void OnLevelChanged(FWeaponLevel NewLevel) override;

private:
	float CachedDamage          = 40.f;
	float CachedCooldown        = 1.40f;
	float CachedSpeed           = 180.f;
	float CachedExplosionRadius = 30.f;

	// 発射中プロジェクタイルの追跡（インデックスではなく発射時の位置を追跡）
	// 簡略化: 爆発は発射時に最初に当たった敵の位置で単回処理

	void CacheParams();
};
