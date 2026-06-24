#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Survivors/Game/SurvivorsTypes.h"
#include "SurvivorsObservationComponent.generated.h"

class ASurvivorsGame;

UCLASS(ClassGroup=(Custom), meta=(BlueprintSpawnableComponent))
class REINBALANCE_API USurvivorsObservationComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	USurvivorsObservationComponent();

	void Initialize(ASurvivorsGame* InGame);
	TArray<FSurvivorsObsSegment> GetObsSchema() const;
	FString GetObsSchemaHash() const;
	TArray<float> GetObservation() const;

private:
	UPROPERTY()
	TObjectPtr<ASurvivorsGame> Game;
};
