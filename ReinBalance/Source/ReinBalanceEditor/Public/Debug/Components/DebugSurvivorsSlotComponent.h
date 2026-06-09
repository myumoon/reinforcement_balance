// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Components/SceneComponent.h"
#include "Survivors/Logic/SurvivorsTypes.h"
#include "Survivors/Logic/SurvivorsGameConstants.h"
#include "DebugSurvivorsSlotComponent.generated.h"

class ASurvivorsGame;


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
{
	GENERATED_BODY()

public:	
	// Sets default values for this component's properties
	UDebugSurvivorsSlotComponent();

protected:
	// Called when the game starts
	virtual void BeginPlay() override;

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
};
