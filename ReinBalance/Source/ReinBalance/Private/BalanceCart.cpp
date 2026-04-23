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

	// 関節1: カート天面 (CartMesh スケール Z=0.4 → parent-local Z=50 が世界座標 Z=20cm)
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

	// 関節2: PolePivot の子として PolePivot ローカル Z=100cm に配置。
	// PolePivot が回転するとポール1先端へ自動追従する。
	PolePivot2 = CreateDefaultSubobject<USceneComponent>(TEXT("PolePivot2"));
	PolePivot2->SetupAttachment(PolePivot);
	PolePivot2->SetRelativeLocation(FVector(0.f, 0.f, 100.f));

	// ポール2: 全長 50cm に対応 (スケール Z=0.5)、pivot から 25cm 上にオフセット
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

	const float PrevCartVel    = CartVel;
	const float PrevPoleAngVel = PoleAngVel;
	Integrate(Force);

	if (NumPoles >= 2)
	{
		// ポール1積分結果から加速度を近似（差分）
		const float CartAccel    = (CartVel    - PrevCartVel)    / PhysicsDt;
		const float PoleAngAccel = (PoleAngVel - PrevPoleAngVel) / PhysicsDt;

		// ポール1先端の 2D 加速度 (X: 水平, Z: 垂直)
		// 先端位置 = (CartPos + 2L1*sin(θ1), 2L1*cos(θ1)) を 2 階微分
		const float CosT1 = FMath::Cos(PoleAngle);
		const float SinT1 = FMath::Sin(PoleAngle);
		const float TipAccelX = CartAccel
			+ 2.f * PoleHalfLength * (PoleAngAccel * CosT1 - PoleAngVel * PoleAngVel * SinT1);
		const float TipAccelZ = -2.f * PoleHalfLength
			* (PoleAngAccel * SinT1 + PoleAngVel * PoleAngVel * CosT1);

		IntegratePole2(TipAccelX, TipAccelZ);

		// 終了条件: ポール2先端がカート関節基準で MinTipHeightMeters を下回ったら終了
		const float TipZ = 2.f * PoleHalfLength  * FMath::Cos(PoleAngle)
		                 + 2.f * PoleHalfLength2 * FMath::Cos(PoleAngle2);
		bEpisodeDone = FMath::Abs(CartPos) > MaxCartPositionMeters
		            || TipZ < MinTipHeightMeters;
	}
	else
	{
		// NumPoles=1: 後方互換の角度閾値判定
		const float MaxAngleRad = FMath::DegreesToRadians(MaxPoleAngleDegrees);
		bEpisodeDone = FMath::Abs(CartPos)   > MaxCartPositionMeters
		            || FMath::Abs(PoleAngle) > MaxAngleRad;
	}
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
		// ポール2は ポール1に対する相対角度で初期化（直列構造の自然な初期状態）
		PoleAngle2  = PoleAngle  + RandStream.FRandRange(-0.05f, 0.05f);
		PoleAngVel2 = PoleAngVel + RandStream.FRandRange(-0.05f, 0.05f);
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
	// obs[4], obs[5] は関節角度・関節角速度（ポール1に対する相対値）
	return { CartPos, CartVel,
	         PoleAngle,  PoleAngVel,
	         PoleAngle2  - PoleAngle,   // θ2_rel
	         PoleAngVel2 - PoleAngVel }; // θ̇2_rel
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

void ABalanceCart::IntegratePole2(float TipAccelX, float TipAccelZ)
{
	// ポール2: 外部加速度 (TipAccelX, TipAccelZ) で駆動されるピボット上の倒立振子
	// ラグランジュ方程式から導出:
	//   (4/3)·L·θ̈ = (g + az)·sin(θ) - ax·cos(θ)
	// ここで θ は鉛直からの絶対角度、az は垂直加速度（上向き正）
	const float Dt = PhysicsDt;
	const float L  = PoleHalfLength2;

	auto F = [&](float Th) -> float
	{
		return ((Gravity + TipAccelZ) * FMath::Sin(Th) - TipAccelX * FMath::Cos(Th))
		       / (L * 4.f / 3.f);
	};

	// RK4: 状態 [θ2, θ̇2]
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
		// PolePivot2 は PolePivot の子なので、相対角度を設定すると世界角度 = θ2_abs になる
		PolePivot2->SetRelativeRotation(
			FRotator(FMath::RadiansToDegrees(PoleAngle2 - PoleAngle), 0.f, 0.f));
	}
}

void ABalanceCart::ApplyPole2Visibility(bool bVisible)
{
	if (PoleMesh2)
	{
		PoleMesh2->SetVisibility(bVisible);
	}
}
