#include "SurvivorsGemViewComponent.h"

#include "Components/InstancedStaticMeshComponent.h"
#include "Components/SceneComponent.h"
#include "Engine/StaticMesh.h"
#include "Materials/Material.h"
#include "Materials/MaterialInstanceDynamic.h"

USurvivorsGemViewComponent::USurvivorsGemViewComponent()
{
	PrimaryComponentTick.bCanEverTick = false;
}

void USurvivorsGemViewComponent::Initialize(ASurvivorsGame* InGame, USceneComponent* AttachParent)
{
	Game = InGame;
	LoadAssets();
	SetupGemInstances(AttachParent);
}

void USurvivorsGemViewComponent::LoadAssets()
{
	SphereMeshAsset = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Sphere.Sphere"));
	BaseMaterialAsset = LoadObject<UMaterial>(nullptr,
		TEXT("/Engine/BasicShapes/BasicShapeMaterial.BasicShapeMaterial"));
}

void USurvivorsGemViewComponent::SetupGemInstances(USceneComponent* AttachParent)
{
	if (!AttachParent) return;

	BlueGemInstances = CreateGemComponent(
		AttachParent, TEXT("BlueGemInstances"), BlueGemMaterial, FLinearColor(0.3f, 0.7f, 1.0f, 1.f));
	GreenGemInstances = CreateGemComponent(
		AttachParent, TEXT("GreenGemInstances"), GreenGemMaterial, FLinearColor(0.2f, 0.9f, 0.2f, 1.f));
	RedGemInstances = CreateGemComponent(
		AttachParent, TEXT("RedGemInstances"), RedGemMaterial, FLinearColor(1.0f, 0.2f, 0.2f, 1.f));
}

UInstancedStaticMeshComponent* USurvivorsGemViewComponent::CreateGemComponent(
	USceneComponent* AttachParent,
	const FName& Name,
	UMaterial* Material,
	const FLinearColor& FallbackColor)
{
	UInstancedStaticMeshComponent* Comp = NewObject<UInstancedStaticMeshComponent>(GetOwner(), Name);
	Comp->RegisterComponent();
	Comp->AttachToComponent(AttachParent, FAttachmentTransformRules::KeepRelativeTransform);
	Comp->SetCollisionEnabled(ECollisionEnabled::NoCollision);
	if (SphereMeshAsset) Comp->SetStaticMesh(SphereMeshAsset);
	if (Material)
	{
		Comp->SetMaterial(0, Material);
	}
	else if (UMaterialInstanceDynamic* Mat = CreateColorMaterial(FallbackColor))
	{
		Comp->SetMaterial(0, Mat);
	}
	return Comp;
}

void USurvivorsGemViewComponent::UpdateView()
{
	if (!Game || !BlueGemInstances || !GreenGemInstances || !RedGemInstances) return;

	BlueGemInstances->ClearInstances();
	GreenGemInstances->ClearInstances();
	RedGemInstances->ClearInstances();

	const FVector Scale(0.35f, 0.35f, 0.35f);
	const int32 GemCount = Game->GetItemCount();
	for (int32 i = 0; i < GemCount; ++i)
	{
		UInstancedStaticMeshComponent* Comp = GetComponentForGemType(Game->GetItemGemType(i));
		if (!Comp) continue;

		const FVector2D GPos = Game->GetItemPos(i);
		const FTransform Transform(
			FRotator::ZeroRotator,
			FVector(GPos.X * Game->SimToUE, GPos.Y * Game->SimToUE, 0.f),
			Scale);
		Comp->AddInstance(Transform);
	}
}

UInstancedStaticMeshComponent* USurvivorsGemViewComponent::GetComponentForGemType(EGemType GemType) const
{
	switch (GemType)
	{
		case EGemType::Green: return GreenGemInstances;
		case EGemType::Red:   return RedGemInstances;
		default:              return BlueGemInstances;
	}
}

UMaterialInstanceDynamic* USurvivorsGemViewComponent::CreateColorMaterial(const FLinearColor& Color)
{
	if (!BaseMaterialAsset) return nullptr;
	UMaterialInstanceDynamic* Mat = UMaterialInstanceDynamic::Create(BaseMaterialAsset, this);
	if (Mat) Mat->SetVectorParameterValue(TEXT("Color"), Color);
	return Mat;
}
