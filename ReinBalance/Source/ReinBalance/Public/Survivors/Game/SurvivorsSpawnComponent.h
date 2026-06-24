#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Survivors/SurvivorsTypes.h"
#include "SurvivorsSpawnComponent.generated.h"

class ASurvivorsGame;

UCLASS(ClassGroup=(Custom), meta=(BlueprintSpawnableComponent))
class REINBALANCE_API USurvivorsSpawnComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	USurvivorsSpawnComponent();

	void Initialize(ASurvivorsGame* InGame);
	void Reset();
	void InitDefaultEnemyTable();
	void InitDefaultSpawnWaves();
	void StepSpawn();
	FVector2D RandomInsideField();
	FVector2D RandomOnEdge();
	FVector2D RandomSpawnPos();
	void SpawnEnemy(const FSpawnWave& Wave);
	void SpawnBoss();
	const FSpawnWave* GetCurrentWave() const;
	int32 GetCurrentWaveIndex() const;
	bool BuildSpawnWeights(const FSpawnWave& Wave, TArray<FEnemySpawnWeight>& OutWeights, bool& bOutUsedCurriculumPool) const;
	int32 SelectTypeByWeight(const TArray<FEnemySpawnWeight>& Weights);

private:
	UPROPERTY()
	TObjectPtr<ASurvivorsGame> Game;
};
