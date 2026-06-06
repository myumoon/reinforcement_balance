#pragma once
#include "CoreMinimal.h"
#include "Survivors/Logic/Weapons/SurvivorsWeaponBase.h"
#include "SurvivorsKnifeWeapon.generated.h"

/**
 * Knife / ThousandEdge: 前方方向 piercing 高速弾（Amount 本を扇状に発射）
 */
UCLASS()
class REINBALANCE_API USurvivorsKnifeWeapon : public USurvivorsWeaponBase
{
	GENERATED_BODY()
public:
	virtual void Tick(float Dt) override;
	virtual void OnLevelChanged(FWeaponLevel NewLevel) override;

private:
	float CachedDamage   = 6.5f;
	float CachedCooldown = 1.00f;
	float CachedSpeed    = 400.f;
	int32 CachedAmount   = 1;
	int32 CachedPierce   = 1;

	void CacheParams();
};
