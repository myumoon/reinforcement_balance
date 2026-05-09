#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "SurvivorsGUIViewComponent.generated.h"

class ASurvivorsGame;
class USceneComponent;
class UWidgetComponent;

UCLASS(ClassGroup=(Custom), meta=(BlueprintSpawnableComponent))
class REINBALANCE_API USurvivorsGUIViewComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	USurvivorsGUIViewComponent();

	void Initialize(ASurvivorsGame* InGame, USceneComponent* AttachParent);
	void UpdateView();

private:
	UPROPERTY()
	TObjectPtr<ASurvivorsGame> Game;

	UPROPERTY()
	TObjectPtr<UWidgetComponent> HPWidgetComp;
};
