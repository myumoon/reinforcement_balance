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

// パレット色の FColor 定数（DrawDebug 用）
namespace WeaponViewColors
{
	static constexpr FColor Brown      ( 140,  69,  18);
	static constexpr FColor DarkBlue   (  23,  71, 181);
	static constexpr FColor Gray       ( 153, 153, 153);
	static constexpr FColor Cyan       ( 102, 217, 255);
	static constexpr FColor Blue       (  41, 102, 235);
	static constexpr FColor Orange     ( 255, 140,  26);
	static constexpr FColor Yellow     ( 255, 230,  26);
	static constexpr FColor Purple     ( 166,  38, 217);
	static constexpr FColor BluishWhite( 209, 224, 255);
}

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

	// 2. 球プロジェクタイルを ISM に追加（Whip/BloodyTear は除外）
	AddProjectileInstances();

	// 3. フロアアイテム・特殊アイテム・破壊物
	AddPickupInstances();
	AddDestructibleInstances();

	// 4. GPU へ反映
	for (auto& ISM : PaletteISMs)
		if (ISM) ISM->MarkRenderStateDirty();

	// 5. DrawDebug 系（ISM 非使用）
	DrawWhipInstances();
	DrawWeaponAuras();
	DrawGroundZones();
	DrawLaurelShield();
	DrawOrbitOrbs();
	DrawLightningRings();
	DrawPentagramFields();
}

void USurvivorsWeaponViewComponent::AddProjectileInstances()
{
	if (!Game) return;

	for (int32 i = 0; i < Game->GetProjectileCount(); ++i)
	{
		const EWeaponType WType = Game->GetProjectileWeaponType(i);

		// Whip / BloodyTear は DrawWhipInstances() で矩形描画するため除外
		if (WType == EWeaponType::Whip || WType == EWeaponType::BloodyTear) continue;

		const int32      PIdx   = static_cast<int32>(GetWeaponPalette(WType));
		const FSimRadius SimR   = Game->GetProjectileRadius(i);
		const float      WorldR = FMath::Max(Converter.Radius(SimR), 6.f);
		const FVector    Pos3D  = Converter.ToWorld(Game->GetProjectilePos(i), FWorldLayerZ::Projectile());
		// UE の Sphere メッシュの基本スケール 1.0 = 半径 50cm → WorldR/50 でスケーリング
		const FTransform T(FRotator::ZeroRotator, Pos3D, FVector(WorldR / 50.f));
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

void USurvivorsWeaponViewComponent::DrawWhipInstances()
{
	if (!Game || !GetWorld()) return;

	for (int32 i = 0; i < Game->GetProjectileCount(); ++i)
	{
		const EWeaponType WType = Game->GetProjectileWeaponType(i);
		if (WType != EWeaponType::Whip && WType != EWeaponType::BloodyTear) continue;

		const FSimRadius SimR   = Game->GetProjectileRadius(i);
		const float      WorldR = FMath::Max(Converter.Radius(SimR), 6.f);
		const float      WorldHalfWidth = FMath::Max(Game->GetProjectileBoxHalfWidth(i) * Game->SimToUE, WorldR * 3.f);
		const FVector    Pos3D  = Converter.ToWorld(Game->GetProjectilePos(i), FWorldLayerZ::Projectile());

		// Whip はロジック側の矩形当たり判定と同じ半径で描画する
		const FVector HalfExtents(WorldHalfWidth, WorldR, WorldR * 0.4f);
		DrawDebugBox(GetWorld(), Pos3D, HalfExtents, WeaponViewColors::Brown, false, 0.f, 1, 2.f);
	}
}

void USurvivorsWeaponViewComponent::DrawWeaponAuras()
{
	if (!Game || !GetWorld()) return;

	for (int32 s = 0; s < SurvivorsGameConstants::MaxWeaponSlots; ++s)
	{
		const FWeaponSlot& Slot = Game->GetWeaponSlot(s);
		FColor Color  = FColor::Black;
		float  Radius = 0.f;

		if (Slot.Type == EWeaponType::Garlic)
		{
			const int32 Lv = FMath::Clamp(Slot.Level.Value, 1, SurvivorsGameConstants::MaxWeaponLevel);
			Color  = WeaponViewColors::Yellow;
			Radius = SurvivorsGameConstants::GarlicTable[Lv - 1].AreaRadius.Value * Game->GetCachedPassiveEffects().AreaMult * Game->SimToUE;
		}
		else if (Slot.Type == EWeaponType::SoulEater)
		{
			const int32 Lv = FMath::Clamp(Slot.Level.Value, 1, SurvivorsGameConstants::MaxWeaponLevel);
			Color  = FColor(220, 255, 50);  // Garlic と区別するため黄緑寄り
			Radius = SurvivorsGameConstants::SoulEaterTable[Lv - 1].AreaRadius.Value * Game->GetCachedPassiveEffects().AreaMult * Game->SimToUE;
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
		const EWeaponType WType = Game->GetGroundZoneWeaponType(i);
		const bool bWarning = Game->IsGroundZoneWarning(i);
		const FColor Color = bWarning
			? WeaponViewColors::Orange
			: ((WType == EWeaponType::LaBorra) ? FColor(60, 190, 240) : WeaponViewColors::Cyan);
		const float Thickness = bWarning ? 3.f : 2.f;
		const float  Radius = Game->GetGroundZoneRadius(i) * Game->SimToUE;
		const FVector Center = Converter.ToWorld(Game->GetGroundZonePos(i), FWorldLayerZ::GroundZone());
		DrawDebugCircle(GetWorld(), Center, Radius, 32, Color, false, 0.f, 1, Thickness,
			FVector(1, 0, 0), FVector(0, 1, 0));
	}
}

void USurvivorsWeaponViewComponent::DrawLaurelShield()
{
	if (!Game || !GetWorld() || !Game->IsShieldActive()) return;

	const float  Radius = (Game->PlayerRadius + 15.f) * Game->SimToUE;
	const FVector Center = Converter.ToWorld(Game->GetPlayerPos(), FWorldLayerZ::Shield());
	DrawDebugCircle(GetWorld(), Center, Radius, 32, FColor(80, 255, 120), false, 0.f, 1, 4.f,
		FVector(1, 0, 0), FVector(0, 1, 0));
}

void USurvivorsWeaponViewComponent::DrawOrbitOrbs()
{
	if (!Game || !GetWorld()) return;

	const int32 OrbCount = Game->GetOrbitOrbCount();
	for (int32 i = 0; i < OrbCount; ++i)
	{
		const EWeaponType WType   = Game->GetOrbitOrbWeaponType(i);
		const FLinearColor LColor = GetWeaponColor(WType);
		const FColor Color        = LColor.ToFColor(true);
		const float SimRadius     = FMath::Max(Game->GetOrbitOrbVisualRadius(i), 8.f);
		const float WorldRadius   = SimRadius * Game->SimToUE;
		const FVector Center      = Converter.ToWorld(Game->GetOrbitOrbPos(i), FWorldLayerZ::Projectile());
		DrawDebugCircle(GetWorld(), Center, WorldRadius, 32, Color, false, 0.f, 1, 2.f,
			FVector(1, 0, 0), FVector(0, 1, 0));
	}
}

void USurvivorsWeaponViewComponent::DrawLightningRings()
{
	// プレイヤー中心の常時円表示を廃止。
	// 落雷位置は ComputeHits で生成した短寿命 GroundZone (strike marker) として
	// DrawGroundZones() 経由で表示される。ここでは何も描画しない。
}

void USurvivorsWeaponViewComponent::DrawPentagramFields()
{
	if (!Game || !GetWorld()) return;

	for (int32 s = 0; s < SurvivorsGameConstants::MaxWeaponSlots; ++s)
	{
		const FWeaponSlot& Slot = Game->GetWeaponSlot(s);
		if (Slot.Type != EWeaponType::Pentagram && Slot.Type != EWeaponType::GorgeousMoon) continue;

		// Radius=9999 はフィールド全域: FieldHalfSize でキャップして表示
		const float WorldRadius = Game->FieldHalfSize * Game->SimToUE * 0.95f;
		const FVector Center = Converter.ToWorld(Game->GetPlayerPos(), FWorldLayerZ::Aura());
		DrawDebugCircle(GetWorld(), Center, WorldRadius, 64, WeaponViewColors::Purple,
			false, 0.f, 1, 2.f, FVector(1, 0, 0), FVector(0, 1, 0));
	}
}

UMaterialInstanceDynamic* USurvivorsWeaponViewComponent::CreateColorMaterial(const FLinearColor& Color)
{
	if (!BaseMat) return nullptr;
	UMaterialInstanceDynamic* Mat = UMaterialInstanceDynamic::Create(BaseMat, this);
	if (Mat) Mat->SetVectorParameterValue(TEXT("Color"), Color);
	return Mat;
}
