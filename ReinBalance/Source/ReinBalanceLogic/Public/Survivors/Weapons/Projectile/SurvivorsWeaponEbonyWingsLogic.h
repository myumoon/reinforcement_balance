#pragma once
// このファイルは UObject 系ヘッダーをインクルードしてはならない。
#include "CoreMinimal.h"
#include "Survivors/Weapons/Projectile/SurvivorsWeaponPeachoneLogic.h"

class REINBALANCELOGIC_API FSurvivorsWeaponEbonyWingsLogic : public FSurvivorsWeaponPeachoneLogic
{
public:
	virtual void OnLevelChanged(FWeaponLevel NewLevel) override;
	/** Ebony Wings固有型を保ってsandboxへ複製する。
	 * 初心者向け: 親Peachone型へ切り詰めず、同じ派生型とruntime stateを維持する。 */
	virtual TUniquePtr<FSurvivorsWeaponLogic> CloneForPreview(
		FSurvivorsGameLogic* InLogic) const override
	{
		auto Clone = MakeUnique<FSurvivorsWeaponEbonyWingsLogic>(*this);
		Clone->Logic = InLogic;
		return Clone;
	}

protected:
	virtual void CacheParams() override;
};
