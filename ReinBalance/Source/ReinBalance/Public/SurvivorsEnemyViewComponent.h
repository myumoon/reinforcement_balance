#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "SurvivorsEnemyViewComponent.generated.h"

class ASurvivorsGame;
class UInstancedStaticMeshComponent;
class UMaterial;
class UMaterialInstanceDynamic;
class USceneComponent;
class UStaticMesh;

UCLASS(ClassGroup=(Custom), meta=(BlueprintSpawnableComponent))
class REINBALANCE_API USurvivorsEnemyViewComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	USurvivorsEnemyViewComponent();

	void Initialize(ASurvivorsGame* InGame, USceneComponent* AttachParent);
	void UpdateView();

	/** Material should use PerInstanceCustomData(0) as HP ratio and a TypeColor vector parameter. */
	UPROPERTY(EditAnywhere, Category = "Survivors|View")
	TObjectPtr<UMaterial> EnemyInstancedMaterial;

private:
	static constexpr int32 MaxEnemyTypeSlots = 11;

	UPROPERTY()
	TObjectPtr<ASurvivorsGame> Game;

	UPROPERTY()
	TArray<TObjectPtr<UInstancedStaticMeshComponent>> EnemyInstancesByType;

	UPROPERTY()
	TObjectPtr<UStaticMesh> CubeMeshAsset;

	UPROPERTY()
	TObjectPtr<UMaterial> BaseMaterialAsset;

	UPROPERTY()
	TObjectPtr<USceneComponent> AttachParentComponent;

	void LoadAssets();
	UInstancedStaticMeshComponent* EnsureEnemyComponent(int32 Type);
	static FLinearColor GetEnemyTypeColor(int32 Type);
	static FVector GetEnemyTypeScale(int32 Type);
	UMaterialInstanceDynamic* CreateEnemyMaterial(const FLinearColor& TypeColor);
};
