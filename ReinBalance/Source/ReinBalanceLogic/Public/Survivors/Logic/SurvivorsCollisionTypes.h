#pragma once

#include "CoreMinimal.h"
#include "SurvivorsCollisionTypes.generated.h"

UENUM()
enum class ESurvivorsCollisionOwnerKind : uint8
{
	Player = 0,
	Enemy = 1,
	Gem = 2,
	Projectile = 3,
	GroundZone = 4,
};

UENUM()
enum class ESurvivorsCollisionLayer : uint8
{
	Enemy = 0,
	Pickup = 1,
};

struct FSurvivorsCollisionRef
{
	ESurvivorsCollisionOwnerKind Kind = ESurvivorsCollisionOwnerKind::Enemy;
	int32 UniqueId = 0;
	int32 IndexAtBuildTime = 0;
};

struct FSurvivorsTargetProxy
{
	FSurvivorsCollisionRef Ref;
	FVector2D Pos;
	float Radius = 0.f;
};

struct FSurvivorsCollisionCell
{
	TArray<int32> TargetIndices;
};

struct REINBALANCELOGIC_API FSurvivorsTargetGrid
{
	FVector2D Origin;
	float CellSize = 128.f;
	int32 NumX = 0;
	int32 NumY = 0;
	float MaxTargetRadius = 0.f;

	TArray<FSurvivorsCollisionCell> Cells;
	TArray<FSurvivorsTargetProxy> Targets;

	void Clear();
	void Rebuild(FVector2D Center, float HalfExtent, float InCellSize);
	bool AddTarget(FSurvivorsTargetProxy Proxy);
	void QueryContacts(FVector2D Pos, float QueryRadius, TArray<int32>& OutIndices) const;

private:
	FIntPoint WorldToCell(FVector2D Pos) const;
};
