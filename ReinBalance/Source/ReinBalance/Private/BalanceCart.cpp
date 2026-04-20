#include "BalanceCart.h"
#include "Components/StaticMeshComponent.h"
#include "UObject/ConstructorHelpers.h"

ABalanceCart::ABalanceCart()
{
	PrimaryActorTick.bCanEverTick = true;

	// カート本体: Cube (100cm) を (200 x 50 x 40 cm) にスケール
	CartMesh = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("CartMesh"));
	RootComponent = CartMesh;
	CartMesh->SetSimulatePhysics(false);
	{
		static ConstructorHelpers::FObjectFinder<UStaticMesh> Asset(
			TEXT("/Engine/BasicShapes/Cube.Cube"));
		if (Asset.Succeeded())
		{
			CartMesh->SetStaticMesh(Asset.Object);
			CartMesh->SetRelativeScale3D(FVector(2.0f, 0.5f, 0.4f));
		}
	}

	// PolePivot: カート天面（関節）を回転軸とする中間コンポーネント
	// CartMesh のスケール Z=0.4 のため、天面の親ローカル座標 = 100/2 / 0.4 = 125 → 50 (補正済み)
	// 世界座標の天面高さ = 0.4 * 50 = 20 cm ✓
	PolePivot = CreateDefaultSubobject<USceneComponent>(TEXT("PolePivot"));
	PolePivot->SetupAttachment(RootComponent);
	PolePivot->SetRelativeLocation(FVector(0.f, 0.f, 50.f));

	// ポール: Cylinder (100cm 高, 径 100cm) を (10cm 径, 100cm 高) にスケール
	// PolePivot 基準で 50cm 上にオフセット → ポール下端が関節位置と一致
	PoleMesh = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("PoleMesh"));
	PoleMesh->SetupAttachment(PolePivot);
	PoleMesh->SetSimulatePhysics(false);
	{
		static ConstructorHelpers::FObjectFinder<UStaticMesh> Asset(
			TEXT("/Engine/BasicShapes/Cylinder.Cylinder"));
		if (Asset.Succeeded())
		{
			PoleMesh->SetStaticMesh(Asset.Object);
			PoleMesh->SetRelativeScale3D(FVector(0.1f, 0.1f, 1.0f));
			PoleMesh->SetRelativeLocation(FVector(0.f, 0.f, 50.f));
		}
	}
}

void ABalanceCart::BeginPlay()
{
	Super::BeginPlay();
	ResetState(TOptional<int32>());
}

void ABalanceCart::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
	UpdateVisuals();
}

void ABalanceCart::PhysicsStep(float NormalizedForce)
{
	if (bEpisodeDone)
	{
		return;
	}

	const float Force = FMath::Clamp(NormalizedForce, -1.f, 1.f) * MaxForceNewtons;
	Integrate(Force);

	const float MaxAngleRad = FMath::DegreesToRadians(MaxPoleAngleDegrees);
	bEpisodeDone = FMath::Abs(CartPos)   > MaxCartPositionMeters
	            || FMath::Abs(PoleAngle) > MaxAngleRad;
}

void ABalanceCart::ResetState(TOptional<int32> Seed)
{
	if (Seed.IsSet())
	{
		RandStream.Initialize(Seed.GetValue());
	}
	else
	{
		RandStream.GenerateNewSeed();
	}

	// gymnasium CartPole に合わせて [-0.05, 0.05] の一様乱数で初期化
	CartPos    = RandStream.FRandRange(-0.05f, 0.05f);
	CartVel    = RandStream.FRandRange(-0.05f, 0.05f);
	PoleAngle  = RandStream.FRandRange(-0.05f, 0.05f);
	PoleAngVel = RandStream.FRandRange(-0.05f, 0.05f);
	bEpisodeDone = false;
}

TArray<float> ABalanceCart::GetObservation() const
{
	// 後 2 要素はダブルポール拡張用プレースホルダー
	return { CartPos, CartVel, PoleAngle, PoleAngVel, 0.f, 0.f };
}

bool ABalanceCart::IsDone() const
{
	return bEpisodeDone;
}

float ABalanceCart::GetReward() const
{
	return bEpisodeDone ? 0.f : 1.f;
}

void ABalanceCart::Integrate(float Force)
{
	// Barto et al. (1983) の CartPole 運動方程式を RK4 で積分
	// 参考: https://coneural.org/florian/papers/05_cart_pole.pdf
	const float Dt = PhysicsDt;

	auto Deriv = [&](float X, float Xd, float Th, float Thd,
	                 float& Xdd, float& Thdd)
	{
		const float CosT      = FMath::Cos(Th);
		const float SinT      = FMath::Sin(Th);
		const float TotalMass = CartMass + PoleMass;
		const float MpL       = PoleMass * PoleHalfLength;

		const float Temp = (Force + MpL * Thd * Thd * SinT) / TotalMass;
		Thdd = (Gravity * SinT - CosT * Temp)
		     / (PoleHalfLength * (4.f / 3.f - PoleMass * CosT * CosT / TotalMass));
		Xdd  = Temp - MpL * Thdd * CosT / TotalMass;
	};

	// k1
	float Xdd1, Thdd1;
	Deriv(CartPos, CartVel, PoleAngle, PoleAngVel, Xdd1, Thdd1);

	// k2
	float Xdd2, Thdd2;
	Deriv(CartPos  + 0.5f * Dt * CartVel,
	      CartVel  + 0.5f * Dt * Xdd1,
	      PoleAngle  + 0.5f * Dt * PoleAngVel,
	      PoleAngVel + 0.5f * Dt * Thdd1,
	      Xdd2, Thdd2);

	// k3
	float Xdd3, Thdd3;
	Deriv(CartPos  + 0.5f * Dt * (CartVel  + 0.5f * Dt * Xdd1),
	      CartVel  + 0.5f * Dt * Xdd2,
	      PoleAngle  + 0.5f * Dt * (PoleAngVel + 0.5f * Dt * Thdd1),
	      PoleAngVel + 0.5f * Dt * Thdd2,
	      Xdd3, Thdd3);

	// k4
	float Xdd4, Thdd4;
	Deriv(CartPos  + Dt * (CartVel  + 0.5f * Dt * Xdd2),
	      CartVel  + Dt * Xdd3,
	      PoleAngle  + Dt * (PoleAngVel + 0.5f * Dt * Thdd2),
	      PoleAngVel + Dt * Thdd3,
	      Xdd4, Thdd4);

	CartPos    += Dt / 6.f * (CartVel    + 2.f * (CartVel  + 0.5f*Dt*Xdd1)  + 2.f * (CartVel  + 0.5f*Dt*Xdd2)  + (CartVel  + Dt*Xdd3));
	CartVel    += Dt / 6.f * (Xdd1  + 2.f * Xdd2  + 2.f * Xdd3  + Xdd4);
	PoleAngle  += Dt / 6.f * (PoleAngVel + 2.f * (PoleAngVel + 0.5f*Dt*Thdd1) + 2.f * (PoleAngVel + 0.5f*Dt*Thdd2) + (PoleAngVel + Dt*Thdd3));
	PoleAngVel += Dt / 6.f * (Thdd1 + 2.f * Thdd2 + 2.f * Thdd3 + Thdd4);
}

void ABalanceCart::UpdateVisuals() const
{
	// UE5 は cm 単位。m → cm 変換 (×100)
	if (CartMesh)
	{
		CartMesh->SetRelativeLocation(FVector(CartPos * 100.f, 0.f, 0.f));
	}
	// PolePivot を Pitch 回転 → ポール下端（関節）を軸に傾く
	if (PolePivot)
	{
		PolePivot->SetRelativeRotation(
			FRotator(FMath::RadiansToDegrees(PoleAngle), 0.f, 0.f));
	}
}
