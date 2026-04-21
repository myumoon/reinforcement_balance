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

	// PolePivot1: カート天面（Z=20cm world）を関節とする中間コンポーネント
	// CartMesh スケール Z=0.4 のため parent-local Z = 20/0.4 = 50
	PolePivot = CreateDefaultSubobject<USceneComponent>(TEXT("PolePivot"));
	PolePivot->SetupAttachment(RootComponent);
	PolePivot->SetRelativeLocation(FVector(0.f, 0.f, 50.f));

	// ポール1: Cylinder を (径10cm, 高100cm) にスケール、pivot から 50cm 上にオフセット
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

	// PolePivot2: Y 方向 +15cm (parent-local +30) にオフセット
	PolePivot2 = CreateDefaultSubobject<USceneComponent>(TEXT("PolePivot2"));
	PolePivot2->SetupAttachment(RootComponent);
	PolePivot2->SetRelativeLocation(FVector(0.f, 30.f, 50.f));

	// ポール2: 全長 50cm（PoleHalfLength2=0.25m）に対応、スケール Z=0.5
	PoleMesh2 = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("PoleMesh2"));
	PoleMesh2->SetupAttachment(PolePivot2);
	PoleMesh2->SetSimulatePhysics(false);
	{
		static ConstructorHelpers::FObjectFinder<UStaticMesh> Asset(
			TEXT("/Engine/BasicShapes/Cylinder.Cylinder"));
		if (Asset.Succeeded())
		{
			PoleMesh2->SetStaticMesh(Asset.Object);
			PoleMesh2->SetRelativeScale3D(FVector(0.1f, 0.1f, 0.5f));
			PoleMesh2->SetRelativeLocation(FVector(0.f, 0.f, 25.f));
		}
	}
	PoleMesh2->SetVisibility(false);
}

void ABalanceCart::OnConstruction(const FTransform& Transform)
{
	Super::OnConstruction(Transform);
	ApplyPole2Visibility(NumPoles >= 2);
}

void ABalanceCart::BeginPlay()
{
	Super::BeginPlay();
	ApplyPole2Visibility(NumPoles >= 2);
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

	const float PrevCartVel = CartVel;
	Integrate(Force);

	if (NumPoles >= 2)
	{
		// カート加速度をポール1の積分結果から近似して取得
		const float CartAccel = (CartVel - PrevCartVel) / PhysicsDt;
		IntegratePole2(CartAccel);
	}

	const float MaxAngleRad = FMath::DegreesToRadians(MaxPoleAngleDegrees);
	bEpisodeDone = FMath::Abs(CartPos)  > MaxCartPositionMeters
	            || FMath::Abs(PoleAngle) > MaxAngleRad
	            || (NumPoles >= 2 && FMath::Abs(PoleAngle2) > MaxAngleRad);
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

	CartPos    = RandStream.FRandRange(-0.05f, 0.05f);
	CartVel    = RandStream.FRandRange(-0.05f, 0.05f);
	PoleAngle  = RandStream.FRandRange(-0.05f, 0.05f);
	PoleAngVel = RandStream.FRandRange(-0.05f, 0.05f);

	if (NumPoles >= 2)
	{
		PoleAngle2  = RandStream.FRandRange(-0.05f, 0.05f);
		PoleAngVel2 = RandStream.FRandRange(-0.05f, 0.05f);
	}
	else
	{
		PoleAngle2  = 0.f;
		PoleAngVel2 = 0.f;
	}

	bEpisodeDone = false;
}

TArray<float> ABalanceCart::GetObservation() const
{
	return { CartPos, CartVel, PoleAngle, PoleAngVel, PoleAngle2, PoleAngVel2 };
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
	// Barto et al. (1983) CartPole 運動方程式を RK4 で積分
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

	float Xdd1, Thdd1;
	Deriv(CartPos, CartVel, PoleAngle, PoleAngVel, Xdd1, Thdd1);

	float Xdd2, Thdd2;
	Deriv(CartPos  + 0.5f * Dt * CartVel,
	      CartVel  + 0.5f * Dt * Xdd1,
	      PoleAngle  + 0.5f * Dt * PoleAngVel,
	      PoleAngVel + 0.5f * Dt * Thdd1,
	      Xdd2, Thdd2);

	float Xdd3, Thdd3;
	Deriv(CartPos  + 0.5f * Dt * (CartVel  + 0.5f * Dt * Xdd1),
	      CartVel  + 0.5f * Dt * Xdd2,
	      PoleAngle  + 0.5f * Dt * (PoleAngVel + 0.5f * Dt * Thdd1),
	      PoleAngVel + 0.5f * Dt * Thdd2,
	      Xdd3, Thdd3);

	float Xdd4, Thdd4;
	Deriv(CartPos  + Dt * (CartVel  + 0.5f * Dt * Xdd2),
	      CartVel  + Dt * Xdd3,
	      PoleAngle  + Dt * (PoleAngVel + 0.5f * Dt * Thdd2),
	      PoleAngVel + Dt * Thdd3,
	      Xdd4, Thdd4);

	CartPos    += Dt / 6.f * (CartVel    + 2.f*(CartVel  + 0.5f*Dt*Xdd1)  + 2.f*(CartVel  + 0.5f*Dt*Xdd2)  + (CartVel  + Dt*Xdd3));
	CartVel    += Dt / 6.f * (Xdd1  + 2.f*Xdd2  + 2.f*Xdd3  + Xdd4);
	PoleAngle  += Dt / 6.f * (PoleAngVel + 2.f*(PoleAngVel + 0.5f*Dt*Thdd1) + 2.f*(PoleAngVel + 0.5f*Dt*Thdd2) + (PoleAngVel + Dt*Thdd3));
	PoleAngVel += Dt / 6.f * (Thdd1 + 2.f*Thdd2 + 2.f*Thdd3 + Thdd4);
}

void ABalanceCart::IntegratePole2(float CartAccel)
{
	// カートが外部加速度 CartAccel で駆動される場合のポール単体方程式
	// θ̈ = (g·sin(θ) - a·cos(θ)) / (L · 4/3)
	// ポール質量が CartMass に比べ十分小さい前提の近似式
	const float Dt = PhysicsDt;
	const float L  = PoleHalfLength2;

	auto F = [&](float Th) -> float
	{
		return (Gravity * FMath::Sin(Th) - CartAccel * FMath::Cos(Th)) / (L * 4.f / 3.f);
	};

	// RK4: 状態 [θ, θ̇]
	const float k1_pos = PoleAngVel2;
	const float k1_vel = F(PoleAngle2);

	const float k2_pos = PoleAngVel2 + 0.5f * Dt * k1_vel;
	const float k2_vel = F(PoleAngle2 + 0.5f * Dt * k1_pos);

	const float k3_pos = PoleAngVel2 + 0.5f * Dt * k2_vel;
	const float k3_vel = F(PoleAngle2 + 0.5f * Dt * k2_pos);

	const float k4_pos = PoleAngVel2 + Dt * k3_vel;
	const float k4_vel = F(PoleAngle2 + Dt * k3_pos);

	PoleAngle2  += Dt / 6.f * (k1_pos + 2.f * k2_pos + 2.f * k3_pos + k4_pos);
	PoleAngVel2 += Dt / 6.f * (k1_vel + 2.f * k2_vel + 2.f * k3_vel + k4_vel);
}

void ABalanceCart::UpdateVisuals() const
{
	if (CartMesh)
	{
		CartMesh->SetRelativeLocation(FVector(CartPos * 100.f, 0.f, 0.f));
	}
	if (PolePivot)
	{
		PolePivot->SetRelativeRotation(
			FRotator(FMath::RadiansToDegrees(PoleAngle), 0.f, 0.f));
	}
	if (PolePivot2 && NumPoles >= 2)
	{
		PolePivot2->SetRelativeRotation(
			FRotator(FMath::RadiansToDegrees(PoleAngle2), 0.f, 0.f));
	}
}

void ABalanceCart::ApplyPole2Visibility(bool bVisible)
{
	if (PoleMesh2)
	{
		PoleMesh2->SetVisibility(bVisible);
	}
}
