#pragma once
// このファイルは UObject 系ヘッダーをインクルードしてはならない。
#include "CoreMinimal.h"
#include "Survivors/Weapons/SurvivorsWeaponLogic.h"

class REINBALANCELOGIC_API FSurvivorsWeaponSantaWaterLogic : public FSurvivorsWeaponLogic
{
public:
	virtual void Tick(float Dt) override;
	virtual void OnLevelChanged(FWeaponLevel NewLevel) override;
	/** Santa Waterのdrop sequenceをsandboxへ複製する。
	 * 初心者向け: 未投下位置とtimerを保ち、zone生成先Logicだけをcloneへ付け替える。 */
	virtual TUniquePtr<FSurvivorsWeaponLogic> CloneForPreview(
		FSurvivorsGameLogic* InLogic) const override
	{
		auto Clone = MakeUnique<FSurvivorsWeaponSantaWaterLogic>(*this);
		Clone->Logic = InLogic;
		return Clone;
	}

private:
	float CachedDamage   = 10.f;
	float CachedCooldown = 2.00f;
	float CachedRadius   = 30.f;
	float CachedDuration = 3.0f;
	int32 CachedAmount   = 1;

	TArray<FVector2D> PendingDropPositions;
	float DropTimer    = 0.f;

	float BurstDamage   = 0.f;
	float BurstRadius   = 0.f;
	float BurstDuration = 0.f;

	void CacheParams();
	void StartDropSequence();
	void SpawnDrop(FVector2D DropPos);
};
