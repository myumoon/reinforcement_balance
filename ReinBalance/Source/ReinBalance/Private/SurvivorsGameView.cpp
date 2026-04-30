#include "SurvivorsGameView.h"
#include "SurvivorsHPWidget.h"
#include "Engine/StaticMesh.h"
#include "Materials/Material.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "Components/WidgetComponent.h"
#include "DrawDebugHelpers.h"

ASurvivorsGameView::ASurvivorsGameView()
{
	PrimaryActorTick.bCanEverTick = true;

	USceneComponent* SceneRoot = CreateDefaultSubobject<USceneComponent>(TEXT("SceneRoot"));
	SetRootComponent(SceneRoot);

	PlayerMesh = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("PlayerMesh"));
	PlayerMesh->SetupAttachment(SceneRoot);
	PlayerMesh->SetCollisionEnabled(ECollisionEnabled::NoCollision);

	HPWidgetComp = CreateDefaultSubobject<UWidgetComponent>(TEXT("HPWidget"));
	HPWidgetComp->SetupAttachment(SceneRoot);
	HPWidgetComp->SetWidgetClass(USurvivorsHPWidget::StaticClass());
	HPWidgetComp->SetWidgetSpace(EWidgetSpace::Screen);
	HPWidgetComp->SetDrawSize(FVector2D(120.f, 20.f));
}

// ---- ライフサイクル ----

void ASurvivorsGameView::BeginPlay()
{
	Super::BeginPlay();

	if (!Game)
	{
		UE_LOG(LogTemp, Warning,
			TEXT("ASurvivorsGameView: Game が未設定です。Details パネルで ASurvivorsGame を設定してください。"));
		SetActorTickEnabled(false);
		return;
	}

	LoadAssets();
	SetupPlayerMesh();
	SetupItemMeshes();
	SetupBoundaryWalls();
	UpdatePositions();
}

void ASurvivorsGameView::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
	if (!Game) return;

	SyncItemMeshes();
	SyncEnemyMeshes();
	UpdateEnemyColors();
	UpdatePositions();
	DrawAura();

	// HP ウィジェット: プレイヤー頭上に追従
	const FVector2D PPos = Game->GetPlayerPos();
	HPWidgetComp->SetRelativeLocation(FVector(PPos.X * SimToUE, PPos.Y * SimToUE, 80.f));
	if (USurvivorsHPWidget* Widget = Cast<USurvivorsHPWidget>(HPWidgetComp->GetUserWidgetObject()))
	{
		Widget->UpdateDisplay(Game->GetPlayerHP(), Game->GetMaxPlayerHP());
	}
}

// ---- セットアップ ----

void ASurvivorsGameView::LoadAssets()
{
	ConeMeshAsset   = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Cone.Cone"));
	SphereMeshAsset = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Sphere.Sphere"));
	CubeMeshAsset   = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Cube.Cube"));
	BaseMaterialAsset = LoadObject<UMaterial>(nullptr,
		TEXT("/Engine/BasicShapes/BasicShapeMaterial.BasicShapeMaterial"));
}

void ASurvivorsGameView::SetupPlayerMesh()
{
	// 緑コーン: Pitch=-90° で XY 平面に伏せる（頂点が +X 方向を向く）
	if (ConeMeshAsset) PlayerMesh->SetStaticMesh(ConeMeshAsset);
	PlayerMesh->SetRelativeScale3D(FVector(0.6f, 0.6f, 0.6f));
	PlayerMesh->SetRelativeRotation(FRotator(-90.f, 0.f, 0.f));

	UMaterialInstanceDynamic* Mat = CreateColorMaterial(FLinearColor(0.f, 0.8f, 0.f, 1.f));
	if (Mat) PlayerMesh->SetMaterial(0, Mat);
}

void ASurvivorsGameView::SetupItemMeshes()
{
	const int32 Count = Game->GetItemCount();
	for (int32 i = 0; i < Count; ++i)
	{
		UStaticMeshComponent* Comp = NewObject<UStaticMeshComponent>(this,
			*FString::Printf(TEXT("ItemMesh_%d"), i));
		Comp->RegisterComponent();
		Comp->AttachToComponent(RootComponent, FAttachmentTransformRules::KeepRelativeTransform);
		Comp->SetCollisionEnabled(ECollisionEnabled::NoCollision);
		if (SphereMeshAsset) Comp->SetStaticMesh(SphereMeshAsset);
		Comp->SetRelativeScale3D(FVector(0.35f, 0.35f, 0.35f));

		UMaterialInstanceDynamic* Mat = CreateColorMaterial(FLinearColor(1.f, 0.85f, 0.f, 1.f));
		if (Mat) Comp->SetMaterial(0, Mat);
		ItemMeshComponents.Add(Comp);
	}
}

void ASurvivorsGameView::SetupBoundaryWalls()
{
	const float HalfSize   = Game->FieldHalfSize * SimToUE;
	const float Thick      = 10.f;
	const float Height     = 50.f;
	const FLinearColor WallColor(0.15f, 0.15f, 0.15f, 1.f);

	auto MakeWall = [&](const TCHAR* Name, FVector Pos, FVector Scale)
	{
		UStaticMeshComponent* Wall = NewObject<UStaticMeshComponent>(this, Name);
		Wall->RegisterComponent();
		Wall->AttachToComponent(RootComponent, FAttachmentTransformRules::KeepRelativeTransform);
		Wall->SetCollisionEnabled(ECollisionEnabled::NoCollision);
		if (CubeMeshAsset) Wall->SetStaticMesh(CubeMeshAsset);
		Wall->SetRelativeLocation(Pos);
		Wall->SetRelativeScale3D(Scale);
		UMaterialInstanceDynamic* Mat = CreateColorMaterial(WallColor);
		if (Mat) Wall->SetMaterial(0, Mat);
		BoundaryWalls.Add(Wall);
	};

	// キューブは 100×100×100 単位。Scale = desiredSize / 100
	const float LongScale  = (HalfSize * 2.f + Thick * 2.f) / 100.f; // 角を被覆
	const float ShortScale = HalfSize * 2.f / 100.f;
	const float ThinScale  = Thick  / 100.f;
	const float TallScale  = Height / 100.f;

	MakeWall(TEXT("WallNorth"), FVector( 0.f,  HalfSize, 0.f), FVector(LongScale,  ThinScale, TallScale));
	MakeWall(TEXT("WallSouth"), FVector( 0.f, -HalfSize, 0.f), FVector(LongScale,  ThinScale, TallScale));
	MakeWall(TEXT("WallEast"),  FVector( HalfSize, 0.f,  0.f), FVector(ThinScale, ShortScale, TallScale));
	MakeWall(TEXT("WallWest"),  FVector(-HalfSize, 0.f,  0.f), FVector(ThinScale, ShortScale, TallScale));
}

// ---- Tick 内処理 ----

void ASurvivorsGameView::SyncItemMeshes()
{
	const int32 GameCount   = Game->GetItemCount();
	const int32 VisualCount = ItemMeshComponents.Num();
	if (VisualCount >= GameCount) return;

	for (int32 i = VisualCount; i < GameCount; ++i)
	{
		UStaticMeshComponent* Comp = NewObject<UStaticMeshComponent>(this,
			*FString::Printf(TEXT("ItemMesh_%d"), i));
		Comp->RegisterComponent();
		Comp->AttachToComponent(RootComponent, FAttachmentTransformRules::KeepRelativeTransform);
		Comp->SetCollisionEnabled(ECollisionEnabled::NoCollision);
		if (SphereMeshAsset) Comp->SetStaticMesh(SphereMeshAsset);
		Comp->SetRelativeScale3D(FVector(0.35f, 0.35f, 0.35f));
		UMaterialInstanceDynamic* Mat = CreateColorMaterial(FLinearColor(1.f, 0.85f, 0.f, 1.f));
		if (Mat) Comp->SetMaterial(0, Mat);
		ItemMeshComponents.Add(Comp);
	}
}

void ASurvivorsGameView::SyncEnemyMeshes()
{
	const int32 GameCount   = Game->GetEnemyCount();
	const int32 VisualCount = EnemyMeshComponents.Num();

	if (GameCount < VisualCount)
	{
		// リセット発生: 全敵メッシュを破棄
		for (TObjectPtr<UStaticMeshComponent>& Comp : EnemyMeshComponents)
			if (Comp) Comp->DestroyComponent();
		EnemyMeshComponents.Empty();
		EnemyMaterials.Empty();
		EnemyTypeColors.Empty();
	}

	for (int32 i = EnemyMeshComponents.Num(); i < GameCount; ++i)
	{
		const int32 Type           = Game->GetEnemyType(i);
		const FLinearColor TypeColor = GetEnemyTypeColor(Type);

		UStaticMeshComponent* Comp = NewObject<UStaticMeshComponent>(this,
			*FString::Printf(TEXT("EnemyMesh_%d"), i));
		Comp->RegisterComponent();
		Comp->AttachToComponent(RootComponent, FAttachmentTransformRules::KeepRelativeTransform);
		Comp->SetCollisionEnabled(ECollisionEnabled::NoCollision);
		if (CubeMeshAsset) Comp->SetStaticMesh(CubeMeshAsset);
		Comp->SetRelativeScale3D(GetEnemyTypeScale(Type));

		UMaterialInstanceDynamic* Mat = CreateColorMaterial(TypeColor);
		if (Mat) Comp->SetMaterial(0, Mat);

		EnemyMeshComponents.Add(Comp);
		EnemyMaterials.Add(Mat);
		EnemyTypeColors.Add(TypeColor);
	}
}

void ASurvivorsGameView::UpdateEnemyColors()
{
	for (int32 i = 0; i < EnemyMaterials.Num(); ++i)
	{
		if (!EnemyMaterials[i]) continue;

		const float HP    = Game->GetEnemyHP(i);
		const float MaxHP = Game->GetEnemyMaxHP(i);
		const float Ratio = FMath::Clamp(MaxHP > 0.f ? HP / MaxHP : 0.f, 0.f, 1.f);

		const FLinearColor& TypeColor = EnemyTypeColors.IsValidIndex(i)
			? EnemyTypeColors[i] : FLinearColor::White;

		// 満HP=タイプ色, 瀕死=白 で線形補間
		const FLinearColor Color(
			FMath::Lerp(1.f, TypeColor.R, Ratio),
			FMath::Lerp(1.f, TypeColor.G, Ratio),
			FMath::Lerp(1.f, TypeColor.B, Ratio),
			1.f
		);
		EnemyMaterials[i]->SetVectorParameterValue(TEXT("Color"), Color);
	}
}

void ASurvivorsGameView::UpdatePositions()
{
	// プレイヤー: 速度方向にコーンを向ける
	const FVector2D PPos = Game->GetPlayerPos();
	const FVector2D PVel = Game->GetPlayerVel();
	PlayerMesh->SetRelativeLocation(FVector(PPos.X * SimToUE, PPos.Y * SimToUE, 0.f));
	if (!PVel.IsNearlyZero(0.01f))
	{
		const float Yaw = FMath::RadiansToDegrees(FMath::Atan2(PVel.Y, PVel.X));
		PlayerMesh->SetRelativeRotation(FRotator(-90.f, Yaw, 0.f));
	}

	// アイテム
	for (int32 i = 0; i < ItemMeshComponents.Num(); ++i)
	{
		const FVector2D IPos = Game->GetItemPos(i);
		ItemMeshComponents[i]->SetRelativeLocation(FVector(IPos.X * SimToUE, IPos.Y * SimToUE, 0.f));
	}

	// 敵
	for (int32 i = 0; i < EnemyMeshComponents.Num(); ++i)
	{
		const FVector2D EPos = Game->GetEnemyPos(i);
		EnemyMeshComponents[i]->SetRelativeLocation(FVector(EPos.X * SimToUE, EPos.Y * SimToUE, 0.f));
	}
}

void ASurvivorsGameView::DrawAura()
{
	if (!GetWorld()) return;

	const FVector2D PPos   = Game->GetPlayerPos();
	const float     Radius = Game->AuraRadius * SimToUE;

	DrawDebugCircle(
		GetWorld(),
		FVector(PPos.X * SimToUE, PPos.Y * SimToUE, 2.f),
		Radius,
		48,
		FColor(50, 150, 255, 255),
		false,
		0.f,    // 1フレームのみ表示（Tick 毎に再描画）
		1,
		3.f,
		FVector(1.f, 0.f, 0.f),
		FVector(0.f, 1.f, 0.f),
		false
	);
}

// ---- ファクトリー ----

FLinearColor ASurvivorsGameView::GetEnemyTypeColor(int32 Type)
{
	switch (Type)
	{
		case 0:  return FLinearColor(1.f,  0.f,  0.f,  1.f); // A: 赤
		case 1:  return FLinearColor(1.f,  0.4f, 0.f,  1.f); // B: オレンジ
		case 2:  return FLinearColor(0.7f, 0.f,  0.5f, 1.f); // C: 赤紫
		default: return FLinearColor(0.5f, 0.5f, 0.5f, 1.f);
	}
}

FVector ASurvivorsGameView::GetEnemyTypeScale(int32 Type)
{
	switch (Type)
	{
		case 0:  return FVector(0.5f, 0.5f, 0.4f); // A: 正方形
		case 1:  return FVector(0.4f, 0.8f, 0.4f); // B: 縦長
		case 2:  return FVector(0.8f, 0.4f, 0.4f); // C: 横長
		default: return FVector(0.5f, 0.5f, 0.4f);
	}
}

UMaterialInstanceDynamic* ASurvivorsGameView::CreateColorMaterial(const FLinearColor& Color)
{
	if (!BaseMaterialAsset) return nullptr;
	UMaterialInstanceDynamic* Mat = UMaterialInstanceDynamic::Create(BaseMaterialAsset, this);
	if (Mat) Mat->SetVectorParameterValue(TEXT("Color"), Color);
	return Mat;
}
