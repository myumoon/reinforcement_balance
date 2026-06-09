// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Components/SceneComponent.h"
#include "Survivors/Logic/SurvivorsTypes.h"
#include "Survivors/Logic/SurvivorsGameConstants.h"
#include "Survivors/Logic/SurvivorsDebugRegistry.h"
#include "DebugSurvivorsSlotComponent.generated.h"


USTRUCT()
struct FDebugWeaponParam
{
	GENERATED_BODY()

	EWeaponType WeaponType = EWeaponType::None;
	int32 Level = 1;
};

USTRUCT()
struct FDebugPassiveParam
{
	GENERATED_BODY()
	
	EPassiveItemType PassiveType = EPassiveItemType::None;
	int32 Level = 1;
};


UCLASS( ClassGroup=(Custom), meta=(BlueprintSpawnableComponent) )
class REINBALANCEEDITOR_API UDebugSurvivorsSlotComponent : public USceneComponent
#if WITH_EDITOR
	, public ISurvivorsDebugSlot
#endif
{
	GENERATED_BODY()

public:
	// Sets default values for this component's properties
	UDebugSurvivorsSlotComponent();

protected:
	// Called when the game starts
	virtual void BeginPlay() override;

	// Called when the game ends
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

private:
	/** 武器を設定 */
	void SetupWeapons();

	/** パッシブアイテムを設定 */
	void SetupPassiveItems();

public:
	// Called every frame
	virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

	UPROPERTY(EditAnywhere, Category = "Survivors|Debug")
	ASurvivorsGame* Game;

	UPROPERTY(EditAnywhere, Category = "Survivors|Debug")
	FDebugWeaponParam WeaponSlots[SurvivorsGameConstants::MaxWeaponSlots];

	UPROPERTY(EditAnywhere, Category = "Survivors|Debug")
	FDebugPassiveParam PassiveSlots[SurvivorsGameConstants::MaxPassiveSlots];

	UPROPERTY(EditAnywhere, Category = "Survivors|Debug")
	bool SkipGetWeaponOnLevelUp;

	UPROPERTY(EditAnywhere, Category = "Survivors|Debug")
	bool SkipGetPassiveItemOnLevelUp;

	UPROPERTY(EditAnywhere, Category = "Survivors|Debug")
	bool SkipSlotLevelUp;

#if WITH_EDITOR
	virtual bool GetSkipGetWeaponOnLevelUp() const override { return SkipGetWeaponOnLevelUp; }
	virtual bool GetSkipGetPassiveItemOnLevelUp() const override { return SkipGetPassiveItemOnLevelUp; }
	virtual bool GetSkipSlotLevelUp() const override { return SkipSlotLevelUp; }
#endif
};
