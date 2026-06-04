#include "Survivors/Logic/SurvivorsCollisionComponent.h"

#include "Survivors/Logic/SurvivorsGame.h"
#include "Survivors/View/WallActor.h"
#include "Kismet/GameplayStatics.h"

// ============================================================================
// FSurvivorsTargetGrid 実装
// ============================================================================

void FSurvivorsTargetGrid::Clear()
{
	for (FSurvivorsCollisionCell& Cell : Cells)
	{
		Cell.TargetIndices.Reset();
	}
	Targets.Reset();
	MaxTargetRadius = 0.f;
}

void FSurvivorsTargetGrid::Rebuild(FVector2D Center, float HalfExtent, float InCellSize)
{
	CellSize = FMath::Max(InCellSize, 1.f);
	NumX = FMath::CeilToInt(2.f * HalfExtent / CellSize);
	NumY = NumX;
	Origin = Center - FVector2D(HalfExtent, HalfExtent);

	Cells.SetNum(NumX * NumY);
	Clear();
}

FIntPoint FSurvivorsTargetGrid::WorldToCell(FVector2D Pos) const
{
	const FVector2D Local = Pos - Origin;
	return FIntPoint(
		FMath::FloorToInt(Local.X / CellSize),
		FMath::FloorToInt(Local.Y / CellSize)
	);
}

bool FSurvivorsTargetGrid::AddTarget(FSurvivorsTargetProxy Proxy)
{
	const FIntPoint Cell = WorldToCell(Proxy.Pos);
	if (Cell.X < 0 || Cell.X >= NumX || Cell.Y < 0 || Cell.Y >= NumY)
	{
		return false;
	}

	const int32 TargetIndex = Targets.Add(Proxy);
	Cells[Cell.Y * NumX + Cell.X].TargetIndices.Add(TargetIndex);
	MaxTargetRadius = FMath::Max(MaxTargetRadius, Proxy.Radius);
	return true;
}

void FSurvivorsTargetGrid::QueryContacts(FVector2D Pos, float QueryRadius, TArray<int32>& OutIndices) const
{
	if (NumX == 0 || NumY == 0) return;

	const FVector2D MinPos = Pos - FVector2D(QueryRadius, QueryRadius);
	const FVector2D MaxPos = Pos + FVector2D(QueryRadius, QueryRadius);

	const FIntPoint MinCell(
		FMath::Clamp(FMath::FloorToInt((MinPos.X - Origin.X) / CellSize), 0, NumX - 1),
		FMath::Clamp(FMath::FloorToInt((MinPos.Y - Origin.Y) / CellSize), 0, NumY - 1)
	);
	const FIntPoint MaxCell(
		FMath::Clamp(FMath::FloorToInt((MaxPos.X - Origin.X) / CellSize), 0, NumX - 1),
		FMath::Clamp(FMath::FloorToInt((MaxPos.Y - Origin.Y) / CellSize), 0, NumY - 1)
	);

	for (int32 CY = MinCell.Y; CY <= MaxCell.Y; ++CY)
	{
		for (int32 CX = MinCell.X; CX <= MaxCell.X; ++CX)
		{
			for (int32 Idx : Cells[CY * NumX + CX].TargetIndices)
			{
				OutIndices.AddUnique(Idx);
			}
		}
	}
}

USurvivorsCollisionComponent::USurvivorsCollisionComponent()
{
	PrimaryComponentTick.bCanEverTick = false;
}

void USurvivorsCollisionComponent::Initialize(ASurvivorsGame* InGame)
{
	Game = InGame;
}

void USurvivorsCollisionComponent::CollectWallActors()
{
	if (!Game) return;

	Game->WallActors.Empty();
	TArray<AActor*> Found;
	UGameplayStatics::GetAllActorsOfClass(Game->GetWorld(), AWallActor::StaticClass(), Found);
	for (AActor* A : Found)
	{
		if (AWallActor* W = Cast<AWallActor>(A))
		{
			Game->WallActors.Add(W);
		}
	}

	UE_LOG(LogTemp, Log, TEXT("[SurvivorsGame] WallActors found: %d"), Game->WallActors.Num());
}

void USurvivorsCollisionComponent::ResolveWallCollisions()
{
	if (!Game) return;

	for (const TObjectPtr<AWallActor>& Wall : Game->WallActors)
	{
		if (!Wall) continue;
		const FBox2D Box = Wall->GetSimBounds(Game->SimToUE);

		const FVector2D Closest(
			FMath::Clamp(Game->PlayerPos.X, Box.Min.X, Box.Max.X),
			FMath::Clamp(Game->PlayerPos.Y, Box.Min.Y, Box.Max.Y));
		const FVector2D Delta = Game->PlayerPos - Closest;
		const float DistSq = Delta.SizeSquared();

		if (DistSq < Game->PlayerRadius * Game->PlayerRadius && DistSq > KINDA_SMALL_NUMBER)
		{
			const float Dist = FMath::Sqrt(DistSq);
			const FVector2D N = Delta / Dist;
			Game->PlayerPos = Closest + N * Game->PlayerRadius;
			const float VdotN = FVector2D::DotProduct(Game->PlayerVel, N);
			if (VdotN < 0.f) Game->PlayerVel -= N * VdotN;
		}
		else if (DistSq <= KINDA_SMALL_NUMBER)
		{
			const float px1 = Game->PlayerPos.X - Box.Min.X;
			const float px2 = Box.Max.X - Game->PlayerPos.X;
			const float py1 = Game->PlayerPos.Y - Box.Min.Y;
			const float py2 = Box.Max.Y - Game->PlayerPos.Y;
			const float m = FMath::Min(FMath::Min(px1, px2), FMath::Min(py1, py2));
			if      (m == px1) { Game->PlayerPos.X = Box.Min.X - Game->PlayerRadius; Game->PlayerVel.X = FMath::Min(Game->PlayerVel.X, 0.f); }
			else if (m == px2) { Game->PlayerPos.X = Box.Max.X + Game->PlayerRadius; Game->PlayerVel.X = FMath::Max(Game->PlayerVel.X, 0.f); }
			else if (m == py1) { Game->PlayerPos.Y = Box.Min.Y - Game->PlayerRadius; Game->PlayerVel.Y = FMath::Min(Game->PlayerVel.Y, 0.f); }
			else               { Game->PlayerPos.Y = Box.Max.Y + Game->PlayerRadius; Game->PlayerVel.Y = FMath::Max(Game->PlayerVel.Y, 0.f); }
		}
	}
}

// ============================================================================
// LocalUniformGrid API
// ============================================================================

float USurvivorsCollisionComponent::GetEffectiveHalfExtent() const
{
	if (!Game) return CollisionHalfExtent;
	return bUseFullFieldCollision ? Game->FieldHalfSize : CollisionHalfExtent;
}

void USurvivorsCollisionComponent::RegisterEnemyTargets()
{
	if (!Game) return;
	for (int32 i = 0; i < Game->Enemies.Num(); ++i)
	{
		const FEnemyState& E = Game->Enemies[i];
		FSurvivorsTargetProxy P;
		P.Ref = { ESurvivorsCollisionOwnerKind::Enemy, E.UniqueId, i };
		P.Pos = E.Pos;
		P.Radius = E.CollisionRadius;
		EnemyGrid.AddTarget(P);
	}
}

void USurvivorsCollisionComponent::RegisterPickupTargets()
{
	if (!Game) return;
	for (int32 i = 0; i < Game->Gems.Num(); ++i)
	{
		const FGemState& G = Game->Gems[i];
		FSurvivorsTargetProxy P;
		P.Ref = { ESurvivorsCollisionOwnerKind::Gem, G.UniqueId, i };
		P.Pos = G.Pos;
		P.Radius = 0.f;
		PickupGrid.AddTarget(P);
	}
}

void USurvivorsCollisionComponent::BuildEnemyGrid()
{
	if (!Game) return;
	EnemyGrid.Rebuild(Game->PlayerPos, GetEffectiveHalfExtent(), CollisionCellSize);
	RegisterEnemyTargets();
}

void USurvivorsCollisionComponent::BuildPickupGrid()
{
	if (!Game) return;
	PickupGrid.Rebuild(Game->PlayerPos, GetEffectiveHalfExtent(), CollisionCellSize);
	RegisterPickupTargets();
}

void USurvivorsCollisionComponent::QueryEnemyContacts(FVector2D Pos, float Radius, TArray<FSurvivorsTargetProxy const*>& Out) const
{
	TArray<int32> Indices;
	EnemyGrid.QueryContacts(Pos, Radius + EnemyGrid.MaxTargetRadius, Indices);
	for (int32 Idx : Indices)
	{
		Out.Add(&EnemyGrid.Targets[Idx]);
	}
}

void USurvivorsCollisionComponent::QueryPickupContacts(FVector2D Pos, float Radius, TArray<FSurvivorsTargetProxy const*>& Out) const
{
	TArray<int32> Indices;
	PickupGrid.QueryContacts(Pos, Radius + PickupGrid.MaxTargetRadius, Indices);
	for (int32 Idx : Indices)
	{
		Out.Add(&PickupGrid.Targets[Idx]);
	}
}

float USurvivorsCollisionComponent::CastRayToObstacles(FVector2D Origin, FVector2D Dir) const
{
	if (!Game) return 0.f;

	float tMin = TNumericLimits<float>::Max();
	if (Dir.X >  1e-6f) tMin = FMath::Min(tMin, ( Game->FieldHalfSize - Origin.X) / Dir.X);
	if (Dir.X < -1e-6f) tMin = FMath::Min(tMin, (-Game->FieldHalfSize - Origin.X) / Dir.X);
	if (Dir.Y >  1e-6f) tMin = FMath::Min(tMin, ( Game->FieldHalfSize - Origin.Y) / Dir.Y);
	if (Dir.Y < -1e-6f) tMin = FMath::Min(tMin, (-Game->FieldHalfSize - Origin.Y) / Dir.Y);

	for (const TObjectPtr<AWallActor>& Wall : Game->WallActors)
	{
		if (!Wall) continue;
		const FBox2D Box = Wall->GetSimBounds(Game->SimToUE);

		float tNear = 0.f;
		float tFar = TNumericLimits<float>::Max();
		if (FMath::Abs(Dir.X) > 1e-6f)
		{
			float t1 = (Box.Min.X - Origin.X) / Dir.X;
			float t2 = (Box.Max.X - Origin.X) / Dir.X;
			if (t1 > t2) Swap(t1, t2);
			tNear = FMath::Max(tNear, t1);
			tFar = FMath::Min(tFar, t2);
		}
		else if (Origin.X < Box.Min.X || Origin.X > Box.Max.X) continue;

		if (FMath::Abs(Dir.Y) > 1e-6f)
		{
			float t1 = (Box.Min.Y - Origin.Y) / Dir.Y;
			float t2 = (Box.Max.Y - Origin.Y) / Dir.Y;
			if (t1 > t2) Swap(t1, t2);
			tNear = FMath::Max(tNear, t1);
			tFar = FMath::Min(tFar, t2);
		}
		else if (Origin.Y < Box.Min.Y || Origin.Y > Box.Max.Y) continue;

		if (tNear < tFar && tNear > 0.f)
		{
			tMin = FMath::Min(tMin, tNear);
		}
	}

	return tMin < TNumericLimits<float>::Max() ? tMin : 0.f;
}

bool USurvivorsCollisionComponent::ReflectOffWall(FVector2D& InOutPos, FVector2D& InOutVel, float Radius) const
{
	if (!Game) return false;

	bool bReflected = false;

	// フィールド境界での反射（AABB 法線）
	const float HS = Game->FieldHalfSize - Radius;
	if (InOutPos.X > HS)
	{
		InOutPos.X = HS;
		InOutVel.X = -FMath::Abs(InOutVel.X);
		bReflected = true;
	}
	else if (InOutPos.X < -HS)
	{
		InOutPos.X = -HS;
		InOutVel.X = FMath::Abs(InOutVel.X);
		bReflected = true;
	}
	if (InOutPos.Y > HS)
	{
		InOutPos.Y = HS;
		InOutVel.Y = -FMath::Abs(InOutVel.Y);
		bReflected = true;
	}
	else if (InOutPos.Y < -HS)
	{
		InOutPos.Y = -HS;
		InOutVel.Y = FMath::Abs(InOutVel.Y);
		bReflected = true;
	}

	return bReflected;
}
