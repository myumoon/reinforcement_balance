// Fill out your copyright notice in the Description page of Project Settings.


#include "Debug/Components/DebugSurvivorsSlotComponent.h"
#include "Survivors/Game/SurvivorsGame.h"
#include "Survivors/SurvivorsDebugRegistry.h"
#include "Survivors/Game/Weapons/SurvivorsWeaponComponent.h"

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
	FSurvivorsDebugRegistry::RegisterSlotComponent(this);
}

// Called when the game ends
void UDebugSurvivorsSlotComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	FSurvivorsDebugRegistry::UnregisterSlotComponent(this);
	Super::EndPlay(EndPlayReason);
}


// Called every frame
void UDebugSurvivorsSlotComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
}
	
#if WITH_EDITOR
bool UDebugSurvivorsSlotComponent::FilterLevelUpChoices(TArray<FLevelUpChoice>& Choices) const
{
	if (!SkipGetWeaponOnLevelUp && !SkipGetPassiveItemOnLevelUp && !SkipSlotLevelUp)
	{
		return false;
	}

	const bool bSkipWeaponGet  = SkipGetWeaponOnLevelUp;
	const bool bSkipPassiveGet = SkipGetPassiveItemOnLevelUp;
	const bool bSkipSlotLvUp   = SkipSlotLevelUp;

	Choices = Choices.FilterByPredicate([bSkipWeaponGet, bSkipPassiveGet, bSkipSlotLvUp]
		(const FLevelUpChoice& C)
	{
		switch (C.ChoiceType)
		{
		case FLevelUpChoice::EChoiceType::WeaponNew:      return !bSkipWeaponGet;
		case FLevelUpChoice::EChoiceType::PassiveNew:     return !bSkipPassiveGet;
		// WeaponUpgrade / WeaponEvolve / PassiveUpgrade はいずれも「既存スロットの強化」であり、
		// SkipSlotLevelUp の対象としてまとめて扱う。PassiveUpgrade がここに含まれるのは、
		// パッシブアイテムの強化も「スロット強化」カテゴリに属するためであり、
		// 新規取得（PassiveNew）とは区別される。
		case FLevelUpChoice::EChoiceType::WeaponUpgrade:
		case FLevelUpChoice::EChoiceType::WeaponEvolve:
		case FLevelUpChoice::EChoiceType::PassiveUpgrade: return !bSkipSlotLvUp;
		default:                                           return true;
		}
	});
	return Choices.Num() == 0;
}
#endif

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
