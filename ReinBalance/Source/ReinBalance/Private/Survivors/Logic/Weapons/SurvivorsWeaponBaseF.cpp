#include "Survivors/Logic/Weapons/SurvivorsWeaponBaseF.h"
#include "Survivors/Logic/SurvivorsGameLogic.h"

void FSurvivorsWeaponBase::Initialize(FSurvivorsGameLogic* InLogic, int32 InSlotIdx)
{
	Logic   = InLogic;
	SlotIdx = InSlotIdx;
}

void FSurvivorsWeaponBase::Reset()
{
	CooldownTimer = FCooldownSeconds(0.f);
}

void FSurvivorsWeaponBase::SetWeaponType(EWeaponType InType)
{
	WeaponType = InType;
}

void FSurvivorsWeaponBase::SetLevel(FWeaponLevel InLevel)
{
	Level = InLevel;
	OnLevelChanged(InLevel);
}

const FPassiveEffects& FSurvivorsWeaponBase::GetPassiveEffects() const
{
	static FPassiveEffects DefaultEffects;
	if (!Logic) return DefaultEffects;
	return Logic->CachedPassiveEffects;
}
