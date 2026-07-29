#pragma once
// このファイルは UObject 系ヘッダーをインクルードしてはならない。
#include "CoreMinimal.h"
#include "Survivors/Weapons/SurvivorsWeaponLogic.h"
#include "Survivors/SurvivorsGameConstants.h"

class REINBALANCELOGIC_API FSurvivorsWeaponCrossLogic : public FSurvivorsWeaponLogic
{
public:
	virtual void Tick(float Dt) override;
	virtual void OnLevelChanged(FWeaponLevel NewLevel) override;
	/** Crossのburst途中をsandboxへ複製する。
	 * 初心者向け: 発射待ち状態を保ったまま参照先Logicだけをcloneへ付け替える。 */
	virtual TUniquePtr<FSurvivorsWeaponLogic> CloneForPreview(
		FSurvivorsGameLogic* InLogic) const override
	{
		auto Clone = MakeUnique<FSurvivorsWeaponCrossLogic>(*this);
		Clone->Logic = InLogic;
		return Clone;
	}

private:
	float CachedDamage            = 50.f;
	float CachedCooldown          = 1.50f;
	float CachedSpeed             = 320.f;
	float CachedRadius            = 12.f;
	int32 CachedAmount            = 1;
	float CachedKnockbackStrength = SurvivorsGameConstants::KnockbackSim_1;

	int32 PendingCrossShots = 0;
	float CrossBurstTimer   = 0.f;

	float BurstDamage      = 0.f;
	float BurstSpeed       = 0.f;
	float BurstRadius      = 0.f;
	float BurstLifeTime    = 0.f;
	float BurstReverseTime = 0.f;
	float BurstKnockback   = SurvivorsGameConstants::KnockbackSim_1;

	void CacheParams();
	void StartBurst();
	void SpawnCrossShot();
};
