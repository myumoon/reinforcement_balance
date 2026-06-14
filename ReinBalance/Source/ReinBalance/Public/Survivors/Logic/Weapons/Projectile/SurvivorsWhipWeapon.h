#pragma once
#include "CoreMinimal.h"
#include "Survivors/Logic/Weapons/SurvivorsWeaponBase.h"
#include "SurvivorsWhipWeapon.generated.h"

/**
 * Whip / BloodyTear: piercing 横線攻撃（Amount に応じて前後方向へ発射）
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
	int32 CachedAmount   = 1;

	int32 PendingWhips = 0;
	float WhipBurstTimer = 0.f;
	float LastFaceSign = 1.f;

	float BurstDamage = 10.f;
	float BurstHalfWidth = 50.f;
	float BurstHalfHeight = 15.f;
	float BurstLifeTime = 0.20f;
	float BurstFaceSign = 1.f;
	int32 BurstIndex = 0;

	void CacheParams();
	void StartBurst();
	void SpawnWhipSwing(float DirSign);
};
