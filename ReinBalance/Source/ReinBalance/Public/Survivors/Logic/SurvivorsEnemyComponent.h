#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "SurvivorsEnemyComponent.generated.h"

class ASurvivorsGame;

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
	float GetEnemySpeed(int32 TypeId) const;
	float GetEnemyTypeMaxHP(int32 TypeId) const;

private:
	UPROPERTY()
	TObjectPtr<ASurvivorsGame> Game;
};
