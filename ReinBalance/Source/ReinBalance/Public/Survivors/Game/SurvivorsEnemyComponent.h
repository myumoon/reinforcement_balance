#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Survivors/Game/SurvivorsTypes.h"
#include "SurvivorsEnemyComponent.generated.h"

class ASurvivorsGame;
class USurvivorsCollisionComponent;

UCLASS(ClassGroup=(Custom), meta=(BlueprintSpawnableComponent))
class REINBALANCE_API USurvivorsEnemyComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	USurvivorsEnemyComponent();

	void Initialize(ASurvivorsGame* InGame);
	void Reset();
	void UpdateEnemies();
	void ApplyAuraDamage();
	void ApplyContactDamage();
	void ComputeContactHits(USurvivorsCollisionComponent* CollComp, FSurvivorsHitFrame& HitFrame);
	void ApplyContactHits(FSurvivorsHitFrame& HitFrame);
	float GetEnemySpeed(int32 TypeId) const;
	float GetEnemyTypeMaxHP(int32 TypeId) const;

private:
	UPROPERTY()
	TObjectPtr<ASurvivorsGame> Game;
};
