#include "Survivors/SurvivorsCollisionTypes.h"

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
	if (NumX == 0 || NumY == 0)
	{
		return;
	}

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
