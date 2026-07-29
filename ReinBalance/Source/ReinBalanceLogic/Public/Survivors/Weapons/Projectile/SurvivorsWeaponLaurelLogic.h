#pragma once
// このファイルは UObject 系ヘッダーをインクルードしてはならない。
#include "CoreMinimal.h"
#include "Survivors/Weapons/SurvivorsWeaponLogic.h"

class REINBALANCELOGIC_API FSurvivorsWeaponLaurelLogic : public FSurvivorsWeaponLogic
{
public:
	virtual void Tick(float Dt) override;
	virtual void OnLevelChanged(FWeaponLevel NewLevel) override;
	virtual float GetCooldownObsDenominator() const override;
	/** Laurelのshield cooldown/cacheをsandboxへ複製する。
	 * 初心者向け: runtime値を保ったまま、shieldを更新するLogicだけをcloneへ付け替える。 */
	virtual TUniquePtr<FSurvivorsWeaponLogic> CloneForPreview(
		FSurvivorsGameLogic* InLogic) const override
	{
		auto Clone = MakeUnique<FSurvivorsWeaponLaurelLogic>(*this);
		Clone->Logic = InLogic;
		return Clone;
	}

private:
	float CachedShieldDuration = 1.0f;
	float CachedCooldown       = 8.0f;

	void CacheParams();
};
