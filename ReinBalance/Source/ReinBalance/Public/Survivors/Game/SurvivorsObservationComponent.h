#pragma once

/**
 * legacy observation facadeのLogic委譲APIを定義する。
 * 初心者向け: Component自身はschemaや観測値を計算せず、canonical Game Logicだけを参照する。
 */
#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Survivors/SurvivorsTypes.h"
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
