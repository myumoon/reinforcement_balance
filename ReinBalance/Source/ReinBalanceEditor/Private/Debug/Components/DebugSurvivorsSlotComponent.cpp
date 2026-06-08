// Fill out your copyright notice in the Description page of Project Settings.


#include "Debug/Components/DebugSurvivorsSlotComponent.h"
#include "Survivors/Logic/SurvivorsGame.h"
#include "Survivors/Logic/Weapons/SurvivorsWeaponComponent.h"

// Sets default values for this component's properties
UDebugSurvivorsSlotComponent::UDebugSurvivorsSlotComponent()
{
	PrimaryComponentTick.bCanEverTick = false;
}


// Called when the game starts
void UDebugSurvivorsSlotComponent::BeginPlay()
{
	Super::BeginPlay();
	
	if (Game == nullptr)
	{
		return;
	}
	
	SetupWeapons();
	SetupPassiveItems();
	// todo: SkipGetWeaponOnLevelUp を適用
	// todo: SkipGetPassiveItemOnLevelUp を適用
}


// Called every frame
void UDebugSurvivorsSlotComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
}
	
// 武器を設定
void UDebugSurvivorsSlotComponent::SetupWeapons()
{
	for (int32 i = 0; i < SurvivorsGameConstants::MaxWeaponSlots; ++i)
	{
		const auto& Slot = WeaponSlots[i];
		if (Slot.WeaponType == EWeaponType::None || Slot.Level <= 0)
		{
			continue;
		}
		Game->WeaponComponent->EquipWeapon(i, Slot.WeaponType, Slot.Level);
	}
}

// 武器を設定
void UDebugSurvivorsSlotComponent::SetupPassiveItems()
{
	for (int32 i = 0; i < SurvivorsGameConstants::MaxPassiveSlots; ++i)
	{
		const auto& Slot = PassiveSlots[i];
		if (Slot.PassiveType == EPassiveItemType::None || Slot.Level <= 0)
		{
			continue;
		}
		Game->PassiveSlots[i] = FPassiveSlot{ Slot.PassiveType, Slot.Level };
	}
}
