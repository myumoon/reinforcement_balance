#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Survivors/SurvivorsTypes.h"
#include "SurvivorsGemComponent.generated.h"

class ASurvivorsGame;
class USurvivorsCollisionComponent;

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
	void ComputePickupHits(USurvivorsCollisionComponent* CollComp, FSurvivorsHitFrame& HitFrame);
	void ApplyPickupHits(FSurvivorsHitFrame& HitFrame);

private:
	UPROPERTY()
	TObjectPtr<ASurvivorsGame> Game;
};
