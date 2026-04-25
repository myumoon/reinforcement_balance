#include "CoinGameView.h"
#include "Engine/StaticMesh.h"
#include "Materials/Material.h"
#include "Materials/MaterialInstanceDynamic.h"

ACoinGameView::ACoinGameView()
{
	PrimaryActorTick.bCanEverTick = true;

	USceneComponent* SceneRoot = CreateDefaultSubobject<USceneComponent>(TEXT("SceneRoot"));
	SetRootComponent(SceneRoot);

	PlayerMesh = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("PlayerMesh"));
	PlayerMesh->SetupAttachment(SceneRoot);
	PlayerMesh->SetCollisionEnabled(ECollisionEnabled::NoCollision);
}

// ---- ライフサイクル ----

void ACoinGameView::BeginPlay()
{
	Super::BeginPlay();

	if (!Game)
	{
		UE_LOG(LogTemp, Warning,
			TEXT("ACoinGameView: Game が未設定です。Details パネルで ACoinGame を設定してください。"));
		SetActorTickEnabled(false);
		return;
	}

	LoadAssets();
	SetupPlayerMesh();
	SetupCoinMeshes();
	UpdatePositions(); // 初期位置に配置
}

void ACoinGameView::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
	if (!Game) return;

	SyncCoinMeshes();  // BeginPlay 順序ずれによる未生成コインを補完
	SyncEnemyMeshes(); // 敵数の変化（スポーン・リセット）に追従
	UpdatePositions(); // 全オブジェクトの位置を同期
}

// ---- セットアップ ----

void ACoinGameView::LoadAssets()
{
	ConeMeshAsset   = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Cone.Cone"));
	SphereMeshAsset = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Sphere.Sphere"));
	CubeMeshAsset   = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Cube.Cube"));
	BaseMaterialAsset = LoadObject<UMaterial>(nullptr,
		TEXT("/Engine/BasicShapes/BasicShapeMaterial.BasicShapeMaterial"));
}

void ACoinGameView::SetupPlayerMesh()
{
	// 緑コーン: Pitch=-90° で XY 平面に伏せる（頂点が +X 方向を向く）
	if (ConeMeshAsset) PlayerMesh->SetStaticMesh(ConeMeshAsset);
	PlayerMesh->SetRelativeScale3D(FVector(0.6f, 0.6f, 0.6f));
	PlayerMesh->SetRelativeRotation(FRotator(-90.f, 0.f, 0.f));

	UMaterialInstanceDynamic* Mat = CreateColorMaterial(FLinearColor(0.f, 0.8f, 0.f, 1.f));
	if (Mat) PlayerMesh->SetMaterial(0, Mat);
}

void ACoinGameView::SetupCoinMeshes()
{
	const int32 Count = Game ? Game->GetCoinCount() : 0;
	for (int32 i = 0; i < Count; ++i)
	{
		UStaticMeshComponent* Comp = NewObject<UStaticMeshComponent>(this,
			*FString::Printf(TEXT("CoinMesh_%d"), i));
		Comp->RegisterComponent();
		Comp->AttachToComponent(RootComponent, FAttachmentTransformRules::KeepRelativeTransform);
		Comp->SetCollisionEnabled(ECollisionEnabled::NoCollision);
		if (SphereMeshAsset) Comp->SetStaticMesh(SphereMeshAsset);
		Comp->SetRelativeScale3D(FVector(0.4f, 0.4f, 0.4f));

		UMaterialInstanceDynamic* Mat =
			CreateColorMaterial(FLinearColor(1.f, 0.85f, 0.f, 1.f));
		if (Mat) Comp->SetMaterial(0, Mat);
		CoinMeshComponents.Add(Comp);
	}
}

// ---- Tick 内処理 ----

void ACoinGameView::SyncCoinMeshes()
{
	const int32 GameCount   = Game->GetCoinCount();
	const int32 VisualCount = CoinMeshComponents.Num();
	if (VisualCount >= GameCount) return;

	// BeginPlay 順序の問題などでコインメッシュが未生成の場合に追加
	for (int32 i = VisualCount; i < GameCount; ++i)
	{
		UStaticMeshComponent* Comp = NewObject<UStaticMeshComponent>(this,
			*FString::Printf(TEXT("CoinMesh_%d"), i));
		Comp->RegisterComponent();
		Comp->AttachToComponent(RootComponent, FAttachmentTransformRules::KeepRelativeTransform);
		Comp->SetCollisionEnabled(ECollisionEnabled::NoCollision);
		if (SphereMeshAsset) Comp->SetStaticMesh(SphereMeshAsset);
		Comp->SetRelativeScale3D(FVector(0.4f, 0.4f, 0.4f));

		UMaterialInstanceDynamic* Mat =
			CreateColorMaterial(FLinearColor(1.f, 0.85f, 0.f, 1.f));
		if (Mat) Comp->SetMaterial(0, Mat);
		CoinMeshComponents.Add(Comp);
	}
}

void ACoinGameView::SyncEnemyMeshes()
{
	const int32 GameCount   = Game->GetEnemyCount();
	const int32 VisualCount = EnemyMeshComponents.Num();

	if (GameCount < VisualCount)
	{
		// リセット発生: 全敵メッシュを破棄
		for (TObjectPtr<UStaticMeshComponent>& Comp : EnemyMeshComponents)
		{
			if (Comp) Comp->DestroyComponent();
		}
		EnemyMeshComponents.Empty();
	}
	else if (GameCount > VisualCount)
	{
		// 新たな敵がスポーン: 差分だけメッシュを追加
		for (int32 i = VisualCount; i < GameCount; ++i)
		{
			EnemyMeshComponents.Add(CreateEnemyMesh(i, Game->GetEnemyType(i)));
		}
	}
}

void ACoinGameView::UpdatePositions()
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

	// コイン
	for (int32 i = 0; i < CoinMeshComponents.Num(); ++i)
	{
		const FVector2D CPos = Game->GetCoinPos(i);
		CoinMeshComponents[i]->SetRelativeLocation(
			FVector(CPos.X * SimToUE, CPos.Y * SimToUE, 0.f));
	}

	// 敵
	for (int32 i = 0; i < EnemyMeshComponents.Num(); ++i)
	{
		const FVector2D EPos = Game->GetEnemyPos(i);
		EnemyMeshComponents[i]->SetRelativeLocation(
			FVector(EPos.X * SimToUE, EPos.Y * SimToUE, 0.f));
	}
}

// ---- ファクトリー ----

UStaticMeshComponent* ACoinGameView::CreateEnemyMesh(int32 EnemyIndex, int32 Type)
{
	UStaticMeshComponent* Comp = NewObject<UStaticMeshComponent>(this,
		*FString::Printf(TEXT("EnemyMesh_%d_%d"), EnemyIndex, Type));
	Comp->RegisterComponent();
	Comp->AttachToComponent(RootComponent, FAttachmentTransformRules::KeepRelativeTransform);
	Comp->SetCollisionEnabled(ECollisionEnabled::NoCollision);
	if (CubeMeshAsset) Comp->SetStaticMesh(CubeMeshAsset);

	FLinearColor Color;
	FVector Scale;
	switch (Type)
	{
		case 0: // A: 遅い直進追跡 → 赤・正方形
			Color = FLinearColor(1.f, 0.f, 0.f, 1.f);
			Scale = FVector(0.5f, 0.5f, 0.4f);
			break;
		case 1: // B: 速い直進追跡 → オレンジ・縦長四角
			Color = FLinearColor(1.f, 0.4f, 0.f, 1.f);
			Scale = FVector(0.4f, 0.8f, 0.4f);
			break;
		case 2: // C: 予測追跡 → 赤紫・横長四角
			Color = FLinearColor(0.7f, 0.f, 0.5f, 1.f);
			Scale = FVector(0.8f, 0.4f, 0.4f);
			break;
		default:
			Color = FLinearColor::White;
			Scale = FVector(0.5f, 0.5f, 0.4f);
	}
	Comp->SetRelativeScale3D(Scale);

	UMaterialInstanceDynamic* Mat = CreateColorMaterial(Color);
	if (Mat) Comp->SetMaterial(0, Mat);
	return Comp;
}

UMaterialInstanceDynamic* ACoinGameView::CreateColorMaterial(const FLinearColor& Color)
{
	if (!BaseMaterialAsset) return nullptr;
	UMaterialInstanceDynamic* Mat =
		UMaterialInstanceDynamic::Create(BaseMaterialAsset, this);
	if (Mat) Mat->SetVectorParameterValue(TEXT("Color"), Color);
	return Mat;
}
