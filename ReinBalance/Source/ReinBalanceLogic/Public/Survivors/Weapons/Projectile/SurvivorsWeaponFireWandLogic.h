#pragma once
// このファイルは UObject 系ヘッダーをインクルードしてはならない。
#include "CoreMinimal.h"
#include "Survivors/Weapons/SurvivorsWeaponLogic.h"

class REINBALANCELOGIC_API FSurvivorsWeaponFireWandLogic : public FSurvivorsWeaponLogic
{
public:
	virtual void Tick(float Dt) override;
	virtual void OnLevelChanged(FWeaponLevel NewLevel) override;
	virtual void ComputeHits(FSurvivorsHitFrame& HitFrame) override;
	virtual float GetCooldownObsDenominator() const override;
	/** Fire Wandのcooldown/cacheをsandboxへ複製する。
	 * 初心者向け: 元episodeへのpointerだけを共有せず、clone側Logicへ明示的に再束縛する。 */
	virtual TUniquePtr<FSurvivorsWeaponLogic> CloneForPreview(
		FSurvivorsGameLogic* InLogic) const override
	{
		auto Clone = MakeUnique<FSurvivorsWeaponFireWandLogic>(*this);
		Clone->Logic = InLogic;
		return Clone;
	}

private:
	float CachedDamage          = 40.f;
	float CachedCooldown        = 1.40f;
	float CachedSpeed           = 360.f;
	float CachedExplosionRadius = 30.f;
	int32 CachedAmount          = 4;

	void CacheParams();
};
