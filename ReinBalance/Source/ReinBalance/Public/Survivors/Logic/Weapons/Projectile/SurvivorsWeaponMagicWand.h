#pragma once
#include "CoreMinimal.h"
#include "Survivors/Logic/Weapons/SurvivorsWeaponBase.h"
#include "SurvivorsWeaponMagicWand.generated.h"

/**
 * MagicWand / HolyWand: 最近傍敵へ Amount 本の弾を 0.1s 間隔で順次発射。
 * 画面内に敵がいない場合、バースト開始時に1度だけランダム方向を決定し全弾が同一方向へ飛ぶ。
 */
UCLASS()
class REINBALANCE_API USurvivorsWeaponMagicWand : public USurvivorsWeaponBase
{
	GENERATED_BODY()
public:
	virtual void Tick(float Dt) override;
	virtual void OnLevelChanged(FWeaponLevel NewLevel) override;

private:
	float CachedDamage   = 10.f;
	float CachedCooldown = 1.20f;
	float CachedSpeed    = 600.f;
	int32 CachedAmount   = 1;
	int32 CachedPierce   = 1;

	// sequential shot state
	int32     PendingWandShots  = 0;
	float     WandBurstTimer    = 0.f;
	FVector2D BurstNoTargetDir  = FVector2D(1.f, 0.f);  // 画面内に敵がいない場合のバースト共通方向
	// burst snapshot (captured when burst starts)
	float BurstDamage   = 0.f;
	float BurstSpeed    = 0.f;
	float BurstRadius   = 0.f;
	float BurstLifeTime = 0.f;
	int32 BurstPierce   = 1;

	void CacheParams();
	void StartBurst();
	void SpawnWandShot();
};
