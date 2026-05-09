#include "SurvivorsEnemyViewComponent.h"

#include "SurvivorsGame.h"
#include "Components/InstancedStaticMeshComponent.h"
#include "Components/SceneComponent.h"
#include "Engine/StaticMesh.h"
#include "Materials/Material.h"
#include "Materials/MaterialInstanceDynamic.h"

USurvivorsEnemyViewComponent::USurvivorsEnemyViewComponent()
{
	PrimaryComponentTick.bCanEverTick = false;
}

void USurvivorsEnemyViewComponent::Initialize(ASurvivorsGame* InGame, USceneComponent* AttachParent)
{
	Game = InGame;
	AttachParentComponent = AttachParent;
	EnemyInstancesByType.SetNum(MaxEnemyTypeSlots);
	LoadAssets();
}

void USurvivorsEnemyViewComponent::LoadAssets()
{
	CubeMeshAsset = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Cube.Cube"));
	BaseMaterialAsset = LoadObject<UMaterial>(nullptr,
		TEXT("/Engine/BasicShapes/BasicShapeMaterial.BasicShapeMaterial"));
}

void USurvivorsEnemyViewComponent::UpdateView()
{
	if (!Game) return;

	for (TObjectPtr<UInstancedStaticMeshComponent>& Comp : EnemyInstancesByType)
	{
		if (Comp) Comp->ClearInstances();
	}

	const int32 EnemyCount = Game->GetEnemyCount();
	for (int32 i = 0; i < EnemyCount; ++i)
	{
		const int32 Type = Game->GetEnemyType(i);
		const int32 DisplayType = FMath::Clamp(Type, 0, MaxEnemyTypeSlots - 1);
		UInstancedStaticMeshComponent* Comp = EnsureEnemyComponent(Type);
		if (!Comp) continue;

		const FVector2D EPos = Game->GetEnemyPos(i);
		const FTransform Transform(
			FRotator::ZeroRotator,
			FVector(EPos.X * Game->SimToUE, EPos.Y * Game->SimToUE, 0.f),
			GetEnemyTypeScale(DisplayType));
		const int32 InstanceIndex = Comp->AddInstance(Transform);

		const float HP = Game->GetEnemyHP(i);
		const float MaxHP = Game->GetEnemyMaxHP(i);
		const float HpRatio = FMath::Clamp(MaxHP > 0.f ? HP / MaxHP : 0.f, 0.f, 1.f);
		Comp->SetCustomDataValue(InstanceIndex, 0, HpRatio, false);
	}

	for (TObjectPtr<UInstancedStaticMeshComponent>& Comp : EnemyInstancesByType)
	{
		if (Comp) Comp->MarkRenderStateDirty();
	}
}

UInstancedStaticMeshComponent* USurvivorsEnemyViewComponent::EnsureEnemyComponent(int32 Type)
{
	const int32 Slot = FMath::Clamp(Type, 0, MaxEnemyTypeSlots - 1);
	if (EnemyInstancesByType[Slot]) return EnemyInstancesByType[Slot];
	if (!AttachParentComponent) return nullptr;

	const FLinearColor TypeColor = GetEnemyTypeColor(Slot);
	UInstancedStaticMeshComponent* Comp = NewObject<UInstancedStaticMeshComponent>(
		GetOwner(), *FString::Printf(TEXT("EnemyType_%d_Instances"), Slot));
	Comp->RegisterComponent();
	Comp->AttachToComponent(AttachParentComponent, FAttachmentTransformRules::KeepRelativeTransform);
	Comp->SetCollisionEnabled(ECollisionEnabled::NoCollision);
	Comp->NumCustomDataFloats = 1;
	if (CubeMeshAsset) Comp->SetStaticMesh(CubeMeshAsset);
	if (UMaterialInstanceDynamic* Mat = CreateEnemyMaterial(TypeColor))
	{
		Comp->SetMaterial(0, Mat);
	}

	EnemyInstancesByType[Slot] = Comp;
	return Comp;
}

FLinearColor USurvivorsEnemyViewComponent::GetEnemyTypeColor(int32 Type)
{
	switch (Type)
	{
		case 0:  return FLinearColor(1.f,  0.f,  0.f,  1.f);
		case 1:  return FLinearColor(1.f,  0.4f, 0.f,  1.f);
		case 2:  return FLinearColor(0.7f, 0.f,  0.5f, 1.f);
		case 3:  return FLinearColor(0.2f, 0.6f, 1.f,  1.f);
		case 4:  return FLinearColor(0.8f, 0.8f, 0.2f, 1.f);
		case 5:  return FLinearColor(0.5f, 0.9f, 0.4f, 1.f);
		case 6:  return FLinearColor(0.8f, 0.5f, 0.9f, 1.f);
		case 7:  return FLinearColor(0.4f, 0.9f, 0.9f, 1.f);
		case 8:  return FLinearColor(0.9f, 0.7f, 0.3f, 1.f);
		case 9:  return FLinearColor(0.9f, 0.3f, 0.7f, 1.f);
		case 10: return FLinearColor(0.6f, 0.2f, 1.f,  1.f);
		default: return FLinearColor(0.5f, 0.5f, 0.5f, 1.f);
	}
}

FVector USurvivorsEnemyViewComponent::GetEnemyTypeScale(int32 Type)
{
	switch (Type)
	{
		case 0:  return FVector(0.5f, 0.5f, 0.4f);
		case 1:  return FVector(0.4f, 0.8f, 0.4f);
		case 2:  return FVector(0.8f, 0.4f, 0.4f);
		case 10: return FVector(1.0f, 1.0f, 0.8f);
		default: return FVector(0.55f, 0.55f, 0.45f);
	}
}

UMaterialInstanceDynamic* USurvivorsEnemyViewComponent::CreateEnemyMaterial(const FLinearColor& TypeColor)
{
	UMaterial* SourceMaterial = EnemyInstancedMaterial ? EnemyInstancedMaterial.Get() : BaseMaterialAsset.Get();
	if (!SourceMaterial) return nullptr;

	UMaterialInstanceDynamic* Mat = UMaterialInstanceDynamic::Create(SourceMaterial, this);
	if (Mat)
	{
		Mat->SetVectorParameterValue(TEXT("TypeColor"), TypeColor);
		Mat->SetVectorParameterValue(TEXT("Color"), TypeColor);
	}
	return Mat;
}
