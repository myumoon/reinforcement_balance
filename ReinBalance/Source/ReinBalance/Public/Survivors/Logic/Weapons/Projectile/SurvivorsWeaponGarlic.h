#pragma once

#include "CoreMinimal.h"
#include "Survivors/Logic/Weapons/SurvivorsWeaponBase.h"
#include "SurvivorsWeaponGarlic.generated.h"

/**
 * Garlic / Soul Eater 武器（円形オーラで周囲の敵にダメージ）。
 * WeaponType で Garlic/SoulEater を判別し対応するパラメータテーブルを参照する。
 * 旧 EnemyComponent::ApplyAuraDamage() の挙動をここに移管する。
 */
UCLASS()
class REINBALANCE_API USurvivorsWeaponGarlic : public USurvivorsWeaponBase
{
	GENERATED_BODY()

public:
	virtual void Tick(float Dt) override;
	virtual void OnLevelChanged(FWeaponLevel NewLevel) override;
	virtual void ComputeHits(USurvivorsCollisionComponent* CollComp, FSurvivorsHitFrame& HitFrame) override;

private:
	/** レベル変更時にキャッシュする現在のパラメータ */
	float CachedDamage      = 5.f;
	float CachedHitInterval = 1.30f;
	float CachedAreaRadius  = 25.f;

	void CacheParams();
};
