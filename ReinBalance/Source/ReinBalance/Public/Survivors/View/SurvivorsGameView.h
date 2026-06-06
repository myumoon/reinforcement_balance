#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "SurvivorsGameView.generated.h"

class ASurvivorsGame;
class USurvivorsDebugViewComponent;
class USurvivorsEnemyViewComponent;
class USurvivorsGUIViewComponent;
class USurvivorsGemViewComponent;
class USurvivorsPlayerViewComponent;
class USurvivorsWeaponViewComponent;

/**
 * Coordinates Survivors visual components against an ASurvivorsGame simulation.
 */
UCLASS()
class REINBALANCE_API ASurvivorsGameView : public AActor
{
	GENERATED_BODY()

public:
	ASurvivorsGameView();

	UPROPERTY(EditAnywhere, Category = "SurvivorsGameView")
	TObjectPtr<ASurvivorsGame> Game;

protected:
	virtual void BeginPlay() override;
	virtual void Tick(float DeltaTime) override;

private:
	UPROPERTY(VisibleAnywhere, Category = "SurvivorsGameView|Components")
	TObjectPtr<USurvivorsDebugViewComponent> DebugView;

	UPROPERTY(VisibleAnywhere, Category = "SurvivorsGameView|Components")
	TObjectPtr<USurvivorsPlayerViewComponent> PlayerView;

	UPROPERTY(VisibleAnywhere, Category = "SurvivorsGameView|Components")
	TObjectPtr<USurvivorsGemViewComponent> GemView;

	UPROPERTY(VisibleAnywhere, Category = "SurvivorsGameView|Components")
	TObjectPtr<USurvivorsEnemyViewComponent> EnemyView;

	UPROPERTY(VisibleAnywhere, Category = "SurvivorsGameView|Components")
	TObjectPtr<USurvivorsGUIViewComponent> GUIView;

	UPROPERTY(VisibleAnywhere, Category = "SurvivorsGameView|Components")
	TObjectPtr<USurvivorsWeaponViewComponent> WeaponView;
};
