#include "Survivors/View/SurvivorsPlayerViewComponent.h"

#include "Survivors/Logic/SurvivorsGame.h"
#include "Components/SceneComponent.h"
#include "Components/StaticMeshComponent.h"
#include "DrawDebugHelpers.h"
#include "Engine/StaticMesh.h"
#include "Materials/Material.h"
#include "Materials/MaterialInstanceDynamic.h"

USurvivorsPlayerViewComponent::USurvivorsPlayerViewComponent()
{
	PrimaryComponentTick.bCanEverTick = false;
}

void USurvivorsPlayerViewComponent::Initialize(ASurvivorsGame* InGame, USceneComponent* AttachParent)
{
	Game = InGame;
	LoadAssets();
	SetupPlayerMesh(AttachParent);
}

void USurvivorsPlayerViewComponent::LoadAssets()
{
	ConeMeshAsset = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Cone.Cone"));
	BaseMaterialAsset = LoadObject<UMaterial>(nullptr,
		TEXT("/Engine/BasicShapes/BasicShapeMaterial.BasicShapeMaterial"));
}

void USurvivorsPlayerViewComponent::SetupPlayerMesh(USceneComponent* AttachParent)
{
	if (!AttachParent || PlayerMesh) return;

	PlayerMesh = NewObject<UStaticMeshComponent>(GetOwner(), TEXT("SurvivorsPlayerMesh"));
	PlayerMesh->RegisterComponent();
	PlayerMesh->AttachToComponent(AttachParent, FAttachmentTransformRules::KeepRelativeTransform);
	PlayerMesh->SetCollisionEnabled(ECollisionEnabled::NoCollision);
	if (ConeMeshAsset) PlayerMesh->SetStaticMesh(ConeMeshAsset);
	PlayerMesh->SetRelativeScale3D(FVector(0.6f, 0.6f, 0.6f));
	PlayerMesh->SetRelativeRotation(FRotator(-90.f, 0.f, 0.f));

	if (PlayerMaterial)
	{
		PlayerMesh->SetMaterial(0, PlayerMaterial);
	}
	else if (UMaterialInstanceDynamic* Mat = CreateColorMaterial(FLinearColor(0.f, 0.8f, 0.f, 1.f)))
	{
		PlayerMesh->SetMaterial(0, Mat);
	}
}

void USurvivorsPlayerViewComponent::UpdateView()
{
	if (!Game || !PlayerMesh) return;

	const FVector2D PPos = Game->GetPlayerPos();
	const FVector2D PVel = Game->GetPlayerVel();
	PlayerMesh->SetRelativeLocation(FVector(PPos.X * Game->SimToUE, PPos.Y * Game->SimToUE, 0.f));
	if (!PVel.IsNearlyZero(0.01f))
	{
		const float Yaw = FMath::RadiansToDegrees(FMath::Atan2(PVel.Y, PVel.X));
		PlayerMesh->SetRelativeRotation(FRotator(-90.f, Yaw, 0.f));
	}

	DrawAura();
}

void USurvivorsPlayerViewComponent::DrawAura()
{
	// 注意: Garlic オーラの描画は USurvivorsWeaponViewComponent::DrawWeaponAuras() に移管済み。
	// このメソッドは後方互換のため残しているが、WeaponViewComponent が登録されている場合は
	// GameView から呼ばれない（UpdateView の実装参照）。
	// 単体で使用する場合の互換用として空実装のまま残す。
}

UMaterialInstanceDynamic* USurvivorsPlayerViewComponent::CreateColorMaterial(const FLinearColor& Color)
{
	if (!BaseMaterialAsset) return nullptr;
	UMaterialInstanceDynamic* Mat = UMaterialInstanceDynamic::Create(BaseMaterialAsset, this);
	if (Mat) Mat->SetVectorParameterValue(TEXT("Color"), Color);
	return Mat;
}
