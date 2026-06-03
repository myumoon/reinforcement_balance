#include "Survivors/Logic/Weapons/SurvivorsWeaponBase.h"
#include "Survivors/Logic/SurvivorsGame.h"
#include "Survivors/Logic/Weapons/SurvivorsWeaponComponent.h"

void USurvivorsWeaponBase::Initialize(ASurvivorsGame* InGame, USurvivorsWeaponComponent* InComp, int32 InSlotIdx)
{
	Game       = InGame;
	WeaponComp = InComp;
	SlotIdx    = InSlotIdx;
}

void USurvivorsWeaponBase::Reset()
{
	CooldownTimer = FCooldownSeconds(0.f);
}

void USurvivorsWeaponBase::SetWeaponType(EWeaponType InType)
{
	WeaponType = InType;
	OnLevelChanged(Level);
}

void USurvivorsWeaponBase::SetLevel(FWeaponLevel InLevel)
{
	Level = InLevel;
	OnLevelChanged(Level);
}

const FPassiveEffects& USurvivorsWeaponBase::GetPassiveEffects() const
{
	static FPassiveEffects DefaultEffects;
	if (Game) return Game->CachedPassiveEffects;
	return DefaultEffects;
}
