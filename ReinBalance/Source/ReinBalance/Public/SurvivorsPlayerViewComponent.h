#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "SurvivorsPlayerViewComponent.generated.h"

class ASurvivorsGame;
class UMaterial;
class UMaterialInstanceDynamic;
class USceneComponent;
class UStaticMesh;
class UStaticMeshComponent;

UCLASS(ClassGroup=(Custom), meta=(BlueprintSpawnableComponent))
class REINBALANCE_API USurvivorsPlayerViewComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	USurvivorsPlayerViewComponent();

	void Initialize(ASurvivorsGame* InGame, USceneComponent* AttachParent);
	void UpdateView();

	UPROPERTY(EditAnywhere, Category = "Survivors|View")
	TObjectPtr<UMaterial> PlayerMaterial;

private:
	UPROPERTY()
	TObjectPtr<ASurvivorsGame> Game;

	UPROPERTY()
	TObjectPtr<UStaticMeshComponent> PlayerMesh;

	UPROPERTY()
	TObjectPtr<UStaticMesh> ConeMeshAsset;

	UPROPERTY()
	TObjectPtr<UMaterial> BaseMaterialAsset;

	void LoadAssets();
	void SetupPlayerMesh(USceneComponent* AttachParent);
	void DrawAura();

	UMaterialInstanceDynamic* CreateColorMaterial(const FLinearColor& Color);
};
