#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "SurvivorsGemComponent.generated.h"

class ASurvivorsGame;

UCLASS(ClassGroup=(Custom), meta=(BlueprintSpawnableComponent))
class REINBALANCE_API USurvivorsGemComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	USurvivorsGemComponent();

	void Initialize(ASurvivorsGame* InGame);
	void Reset();
	void DropGem(int32 TypeId, FVector2D Pos);
	void CheckCollections();

private:
	UPROPERTY()
	TObjectPtr<ASurvivorsGame> Game;
};
