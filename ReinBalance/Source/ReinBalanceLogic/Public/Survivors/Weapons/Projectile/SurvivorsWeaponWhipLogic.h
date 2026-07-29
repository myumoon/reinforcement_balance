#pragma once
// このファイルは UObject 系ヘッダーをインクルードしてはならない。
#include "CoreMinimal.h"
#include "Survivors/Weapons/SurvivorsWeaponLogic.h"

class REINBALANCELOGIC_API FSurvivorsWeaponWhipLogic : public FSurvivorsWeaponLogic
{
public:
	virtual void Tick(float Dt) override;
	virtual void OnLevelChanged(FWeaponLevel NewLevel) override;
	virtual void ComputeHits(FSurvivorsHitFrame& HitFrame) override;
	/** Whipのburst/facing stateをsandboxへ複製する。
	 * 初心者向け: 振り途中の向きとtimerを保ち、参照先Logicだけをcloneへ付け替える。 */
	virtual TUniquePtr<FSurvivorsWeaponLogic> CloneForPreview(
		FSurvivorsGameLogic* InLogic) const override
	{
		auto Clone = MakeUnique<FSurvivorsWeaponWhipLogic>(*this);
		Clone->Logic = InLogic;
		return Clone;
	}

private:
	float CachedDamage   = 10.f;
	float CachedCooldown = 1.50f;
	float CachedWidth    = 50.f;
	float CachedHeight   = 15.f;
	int32 CachedAmount   = 1;

	int32 PendingWhips   = 0;
	float WhipBurstTimer = 0.f;
	float LastFaceSign   = 1.f;

	float BurstDamage      = 10.f;
	float BurstHalfWidth   = 50.f;
	float BurstHalfHeight  = 15.f;
	float BurstLifeTime    = 0.20f;
	float BurstFaceSign    = 1.f;
	int32 BurstIndex       = 0;

	void CacheParams();
	void StartBurst();
	void SpawnWhipSwing(float DirSign);
};
