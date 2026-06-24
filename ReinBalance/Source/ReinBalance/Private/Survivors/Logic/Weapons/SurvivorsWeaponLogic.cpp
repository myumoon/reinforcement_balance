#include "Survivors/Logic/Weapons/SurvivorsWeaponLogic.h"
#include "Survivors/Logic/SurvivorsGameLogic.h"

void FSurvivorsWeaponLogic::Initialize(FSurvivorsGameLogic* InLogic, int32 InSlotIdx)
{
	Logic   = InLogic;
	SlotIdx = InSlotIdx;
}

void FSurvivorsWeaponLogic::Reset()
{
	CooldownTimer = FCooldownSeconds(0.f);
}

void FSurvivorsWeaponLogic::SetWeaponType(EWeaponType InType)
{
	WeaponType = InType;
}

void FSurvivorsWeaponLogic::SetLevel(FWeaponLevel InLevel)
{
	Level = InLevel;
	OnLevelChanged(InLevel);
}

const FPassiveEffects& FSurvivorsWeaponLogic::GetPassiveEffects() const
{
	static FPassiveEffects DefaultEffects;
	if (!Logic) return DefaultEffects;
	return Logic->CachedPassiveEffects;
}
