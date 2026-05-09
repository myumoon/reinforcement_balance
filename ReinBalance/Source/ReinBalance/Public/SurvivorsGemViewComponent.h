#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "SurvivorsGame.h"
#include "SurvivorsGemViewComponent.generated.h"

class UInstancedStaticMeshComponent;
class UMaterial;
class UMaterialInstanceDynamic;
class USceneComponent;
class UStaticMesh;

UCLASS(ClassGroup=(Custom), meta=(BlueprintSpawnableComponent))
class REINBALANCE_API USurvivorsGemViewComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	USurvivorsGemViewComponent();

	void Initialize(ASurvivorsGame* InGame, USceneComponent* AttachParent);
	void UpdateView();

	UPROPERTY(EditAnywhere, Category = "Survivors|View")
	TObjectPtr<UMaterial> BlueGemMaterial;

	UPROPERTY(EditAnywhere, Category = "Survivors|View")
	TObjectPtr<UMaterial> GreenGemMaterial;

	UPROPERTY(EditAnywhere, Category = "Survivors|View")
	TObjectPtr<UMaterial> RedGemMaterial;

private:
	UPROPERTY()
	TObjectPtr<ASurvivorsGame> Game;

	UPROPERTY()
	TObjectPtr<UInstancedStaticMeshComponent> BlueGemInstances;

	UPROPERTY()
	TObjectPtr<UInstancedStaticMeshComponent> GreenGemInstances;

	UPROPERTY()
	TObjectPtr<UInstancedStaticMeshComponent> RedGemInstances;

	UPROPERTY()
	TObjectPtr<UStaticMesh> SphereMeshAsset;

	UPROPERTY()
	TObjectPtr<UMaterial> BaseMaterialAsset;

	void LoadAssets();
	void SetupGemInstances(USceneComponent* AttachParent);
	UInstancedStaticMeshComponent* CreateGemComponent(
		USceneComponent* AttachParent,
		const FName& Name,
		UMaterial* Material,
		const FLinearColor& FallbackColor);
	UInstancedStaticMeshComponent* GetComponentForGemType(EGemType GemType) const;
	UMaterialInstanceDynamic* CreateColorMaterial(const FLinearColor& Color);
};
