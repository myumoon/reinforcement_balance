#include "Survivors/View/SurvivorsWeaponViewComponent.h"

#include "Survivors/Logic/SurvivorsGame.h"
#include "Survivors/Logic/SurvivorsGameConstants.h"
#include "Survivors/Logic/Weapons/SurvivorsWeaponComponent.h"
#include "Components/InstancedStaticMeshComponent.h"
#include "Components/SceneComponent.h"
#include "DrawDebugHelpers.h"
#include "Engine/StaticMesh.h"
#include "Materials/Material.h"
#include "Materials/MaterialInstanceDynamic.h"

USurvivorsWeaponViewComponent::USurvivorsWeaponViewComponent()
{
	PrimaryComponentTick.bCanEverTick = false;
}

void USurvivorsWeaponViewComponent::Initialize(ASurvivorsGame* InGame, USceneComponent* AttachParent)
{
	Game      = InGame;
	Converter = FSimToWorldConverter(InGame ? InGame->SimToUE : 5.f);

	SphereMesh = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Sphere.Sphere"));
	BaseMat    = LoadObject<UMaterial>(nullptr,
		TEXT("/Engine/BasicShapes/BasicShapeMaterial.BasicShapeMaterial"));

	const int32 PaletteCount = static_cast<int32>(EViewPalette::Count);
	PaletteISMs.SetNum(PaletteCount);

	for (int32 p = 0; p < PaletteCount; ++p)
	{
		UInstancedStaticMeshComponent* ISM = NewObject<UInstancedStaticMeshComponent>(
			GetOwner(), *FString::Printf(TEXT("WeaponPalette_%d"), p));
		ISM->RegisterComponent();
		if (AttachParent)
			ISM->AttachToComponent(AttachParent, FAttachmentTransformRules::KeepRelativeTransform);
		ISM->SetCollisionEnabled(ECollisionEnabled::NoCollision);
		if (SphereMesh) ISM->SetStaticMesh(SphereMesh);
		if (UMaterialInstanceDynamic* Mat = CreateColorMaterial(ViewPaletteColors[p]))
			ISM->SetMaterial(0, Mat);
		PaletteISMs[p] = ISM;
	}
}

void USurvivorsWeaponViewComponent::UpdateView()
{
	if (!Game || !GetWorld()) return;

	// 1. ISM を全クリア
	for (auto& ISM : PaletteISMs)
		if (ISM) ISM->ClearInstances();

	// 2. プロジェクタイルを ISM に追加
	AddProjectileInstances();

	// 3. フロアアイテム・特殊アイテム・破壊物（PR2 で本実装、現在は空）
	AddPickupInstances();
	AddDestructibleInstances();

	// 4. GPU へ反映
	for (auto& ISM : PaletteISMs)
		if (ISM) ISM->MarkRenderStateDirty();

	// 5. 円形オブジェクト（DrawDebugCircle）
	DrawWeaponAuras();
	DrawGroundZones();
	DrawLaurelShield();
}

void USurvivorsWeaponViewComponent::AddProjectileInstances()
{
	if (!Game) return;

	for (int32 i = 0; i < Game->GetProjectileCount(); ++i)
	{
		const EWeaponType  WType   = Game->GetProjectileWeaponType(i);
		const int32        PIdx    = static_cast<int32>(GetWeaponPalette(WType));
		const FSimRadius   SimR    = Game->GetProjectileRadius(i);
		const float        WorldR  = FMath::Max(Converter.Radius(SimR), 6.f);
		const FVector      Pos3D   = Converter.ToWorld(Game->GetProjectilePos(i), FWorldLayerZ::Projectile());
		// UE の Sphere メッシュの基本スケール 1.0 = 半径 50cm → WorldR/50 でスケーリング
		const FTransform   T(FRotator::ZeroRotator, Pos3D, FVector(WorldR / 50.f));
		if (PaletteISMs.IsValidIndex(PIdx) && PaletteISMs[PIdx])
			PaletteISMs[PIdx]->AddInstance(T);
	}
}

void USurvivorsWeaponViewComponent::AddPickupInstances()
{
	if (!Game) return;

	// FloorChicken / LittleHeart → PickupHeal パレット
	for (int32 i = 0; i < Game->GetFloorPickupCount(); ++i)
	{
		const float Scale = (Game->GetFloorPickupType(i) == EFloorPickupType::FloorChicken)
			? 0.25f : 0.18f;
		const FVector2D P = Game->GetFloorPickupPos(i);
		const FTransform T(FRotator::ZeroRotator,
			Converter.ToWorld(P, FWorldLayerZ::Pickup()), FVector(Scale));
		const int32 PIdx = static_cast<int32>(EViewPalette::PickupHeal);
		if (PaletteISMs.IsValidIndex(PIdx) && PaletteISMs[PIdx])
			PaletteISMs[PIdx]->AddInstance(T);
	}

	// 特殊アイテム → PickupSpec パレット
	for (int32 i = 0; i < Game->GetSpecialPickupCount(); ++i)
	{
		const FVector2D P = Game->GetSpecialPickupPos(i);
		const FTransform T(FRotator::ZeroRotator,
			Converter.ToWorld(P, FWorldLayerZ::Pickup()), FVector(0.28f));
		const int32 PIdx = static_cast<int32>(EViewPalette::PickupSpec);
		if (PaletteISMs.IsValidIndex(PIdx) && PaletteISMs[PIdx])
			PaletteISMs[PIdx]->AddInstance(T);
	}
}

void USurvivorsWeaponViewComponent::AddDestructibleInstances()
{
	if (!Game) return;

	const int32 PIdx = static_cast<int32>(EViewPalette::Destructible);
	for (int32 i = 0; i < Game->GetDestructibleCount(); ++i)
	{
		if (!Game->IsDestructibleActive(i)) continue;
		const FVector2D P = Game->GetDestructiblePos(i);
		const FTransform T(FRotator::ZeroRotator,
			Converter.ToWorld(P, FWorldLayerZ::GroundZone()), FVector(0.20f));
		if (PaletteISMs.IsValidIndex(PIdx) && PaletteISMs[PIdx])
			PaletteISMs[PIdx]->AddInstance(T);
	}
}

void USurvivorsWeaponViewComponent::DrawWeaponAuras()
{
	if (!Game || !GetWorld()) return;

	for (int32 s = 0; s < SurvivorsGameConstants::MaxWeaponSlots; ++s)
	{
		const FWeaponSlot& Slot = Game->WeaponSlots[s];
		FColor Color  = FColor::Black;
		float  Radius = 0.f;

		if (Slot.Type == EWeaponType::Garlic)
		{
			const int32 Lv = FMath::Clamp(Slot.Level.Value, 1, SurvivorsGameConstants::MaxWeaponLevel);
			Color  = FColor(255, 220, 80);
			Radius = SurvivorsGameConstants::GarlicTable[Lv - 1].AreaRadius.Value * Game->SimToUE;
		}
		else if (Slot.Type == EWeaponType::SoulEater)
		{
			const int32 Lv = FMath::Clamp(Slot.Level.Value, 1, SurvivorsGameConstants::MaxWeaponLevel);
			Color  = FColor(140, 220, 60);
			Radius = SurvivorsGameConstants::SoulEaterTable[Lv - 1].AreaRadius.Value * Game->SimToUE;
		}

		if (Radius > 0.f)
		{
			const FVector Center = Converter.ToWorld(Game->GetPlayerPos(), FWorldLayerZ::Aura());
			DrawDebugCircle(GetWorld(), Center, Radius, 48, Color, false, 0.f, 1, 3.f,
				FVector(1, 0, 0), FVector(0, 1, 0));
		}
	}
}

void USurvivorsWeaponViewComponent::DrawGroundZones()
{
	if (!Game || !GetWorld()) return;

	for (int32 i = 0; i < Game->GetGroundZoneCount(); ++i)
	{
		const EWeaponType WType  = Game->GetGroundZoneWeaponType(i);
		const FColor Color = (WType == EWeaponType::LaBorra)
			? FColor(0, 50, 200) : FColor(30, 80, 220);
		const float  Radius = Game->GetGroundZoneRadius(i) * Game->SimToUE;
		const FVector Center = Converter.ToWorld(Game->GetGroundZonePos(i), FWorldLayerZ::GroundZone());
		DrawDebugCircle(GetWorld(), Center, Radius, 32, Color, false, 0.f, 1, 2.f,
			FVector(1, 0, 0), FVector(0, 1, 0));
	}
}

void USurvivorsWeaponViewComponent::DrawLaurelShield()
{
	if (!Game || !GetWorld() || !Game->bShieldActive) return;

	const float  Radius = (Game->PlayerRadius + 15.f) * Game->SimToUE;
	const FVector Center = Converter.ToWorld(Game->GetPlayerPos(), FWorldLayerZ::Shield());
	DrawDebugCircle(GetWorld(), Center, Radius, 32, FColor(80, 255, 120), false, 0.f, 1, 4.f,
		FVector(1, 0, 0), FVector(0, 1, 0));
}

UMaterialInstanceDynamic* USurvivorsWeaponViewComponent::CreateColorMaterial(const FLinearColor& Color)
{
	if (!BaseMat) return nullptr;
	UMaterialInstanceDynamic* Mat = UMaterialInstanceDynamic::Create(BaseMat, this);
	if (Mat) Mat->SetVectorParameterValue(TEXT("Color"), Color);
	return Mat;
}
