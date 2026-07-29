#pragma once
// このファイルは UObject 系ヘッダーをインクルードしてはならない。
#include "CoreMinimal.h"
#include "Survivors/Weapons/SurvivorsWeaponLogic.h"

class REINBALANCELOGIC_API FSurvivorsWeaponAxeLogic : public FSurvivorsWeaponLogic
{
public:
	virtual void Tick(float Dt) override;
	virtual void OnLevelChanged(FWeaponLevel NewLevel) override;
	/** Axeのburst途中をsandboxへ複製する。
	 * 初心者向け: 発射待ち数とtimerもそのまま保持し、参照先Logicだけを付け替える。 */
	virtual TUniquePtr<FSurvivorsWeaponLogic> CloneForPreview(
		FSurvivorsGameLogic* InLogic) const override
	{
		auto Clone = MakeUnique<FSurvivorsWeaponAxeLogic>(*this);
		Clone->Logic = InLogic;
		return Clone;
	}

private:
	float CachedDamage    = 20.f;
	float CachedCooldown  = 4.00f;
	float CachedSpeed     = 360.f;
	float CachedArcHeight = 120.f;
	int32 CachedAmount    = 1;
	int32 CachedPierce    = 3;

	int32 PendingAxeShots      = 0;
	float AxeBurstTimer        = 0.f;
	int32 BurstShotsFiredCount = 0;

	float BurstDamage    = 0.f;
	float BurstSpeed     = 0.f;
	float BurstRadius    = 0.f;
	float BurstArcHeight = 0.f;
	float BurstHorizDir  = 1.f;
	int32 BurstPierce    = 3;

	void CacheParams();
	void StartBurst();
	void SpawnAxeShot();
};
