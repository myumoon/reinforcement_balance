#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "SurvivorsPlayerComponent.generated.h"

class ASurvivorsGame;

UCLASS(ClassGroup=(Custom), meta=(BlueprintSpawnableComponent))
class REINBALANCE_API USurvivorsPlayerComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	USurvivorsPlayerComponent();

	void Initialize(ASurvivorsGame* InGame);
	void Reset();
	void ApplyAction(int32 ActionIdx);
	float XPRequiredForLevel(int32 Level) const;
	float CumulativeXPForLevel(int32 Level) const;
	void ProcessXPGain(float Amount);
	void OnLevelUp(int32 NextLevel);

private:
	UPROPERTY()
	TObjectPtr<ASurvivorsGame> Game;
};
