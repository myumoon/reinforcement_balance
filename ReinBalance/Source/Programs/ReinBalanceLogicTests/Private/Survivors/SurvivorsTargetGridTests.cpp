#include "TestHarness.h"

#include "Survivors/SurvivorsCollisionTypes.h"

namespace
{
FSurvivorsTargetProxy MakeTarget(int32 UniqueId, FVector2D Pos, float Radius = 0.f, int32 IndexAtBuildTime = 0)
{
	FSurvivorsTargetProxy Target;
	Target.Ref = { ESurvivorsCollisionOwnerKind::Enemy, UniqueId, IndexAtBuildTime };
	Target.Pos = Pos;
	Target.Radius = Radius;
	return Target;
}

bool QueryContainsUniqueId(const FSurvivorsTargetGrid& Grid, const TArray<int32>& Indices, int32 UniqueId)
{
	for (const int32 Index : Indices)
	{
		if (Grid.Targets.IsValidIndex(Index) && Grid.Targets[Index].Ref.UniqueId == UniqueId)
		{
			return true;
		}
	}
	return false;
}
}

TEST_CASE("Survivors target grid returns a boundary target when the query reaches its radius", "[unit][survivors][logic][target-grid]")
{
	FSurvivorsTargetGrid Grid;
	Grid.Rebuild(FVector2D::ZeroVector, 512.f, 128.f);

	CHECK(Grid.AddTarget(MakeTarget(1, FVector2D(128.f, 0.f), 20.f)));

	TArray<int32> Results;
	Grid.QueryContacts(FVector2D::ZeroVector, 148.f, Results);

	CHECK(QueryContainsUniqueId(Grid, Results, 1));
}

TEST_CASE("Survivors target grid rejects targets outside the indexed field", "[unit][survivors][logic][target-grid]")
{
	FSurvivorsTargetGrid Grid;
	Grid.Rebuild(FVector2D::ZeroVector, 512.f, 128.f);

	CHECK(!Grid.AddTarget(MakeTarget(2, FVector2D(2000.f, 0.f), 10.f)));
	CHECK(Grid.Targets.IsEmpty());
	CHECK(Grid.MaxTargetRadius == 0.f);
}

TEST_CASE("Survivors target grid centered on the full field covers origin and field edge", "[unit][survivors][logic][target-grid]")
{
	FSurvivorsTargetGrid LocalGrid;
	LocalGrid.Rebuild(FVector2D(800.f, 0.f), 200.f, 128.f);
	CHECK(!LocalGrid.AddTarget(MakeTarget(1, FVector2D::ZeroVector, 5.f)));

	FSurvivorsTargetGrid FullFieldGrid;
	FullFieldGrid.Rebuild(FVector2D::ZeroVector, 1000.f, 128.f);

	CHECK(FullFieldGrid.AddTarget(MakeTarget(1, FVector2D::ZeroVector, 5.f)));
	CHECK(FullFieldGrid.AddTarget(MakeTarget(2, FVector2D(900.f, 0.f), 5.f, 1)));

	TArray<int32> OriginResults;
	FullFieldGrid.QueryContacts(FVector2D::ZeroVector, 10.f, OriginResults);
	CHECK(QueryContainsUniqueId(FullFieldGrid, OriginResults, 1));
	CHECK(!QueryContainsUniqueId(FullFieldGrid, OriginResults, 2));

	TArray<int32> EdgeResults;
	FullFieldGrid.QueryContacts(FVector2D(900.f, 0.f), 10.f, EdgeResults);
	CHECK(QueryContainsUniqueId(FullFieldGrid, EdgeResults, 2));
}

TEST_CASE("Survivors target grid rebuild clears stale targets and radius", "[unit][survivors][logic][target-grid]")
{
	FSurvivorsTargetGrid Grid;
	Grid.Rebuild(FVector2D::ZeroVector, 256.f, 64.f);
	CHECK(Grid.AddTarget(MakeTarget(1, FVector2D::ZeroVector, 40.f)));

	Grid.Rebuild(FVector2D(100.f, 100.f), 128.f, 32.f);

	CHECK(Grid.Targets.IsEmpty());
	CHECK(Grid.MaxTargetRadius == 0.f);
	CHECK(Grid.Origin == FVector2D(-28.f, -28.f));
	CHECK(Grid.CellSize == 32.f);
}

TEST_CASE("Survivors target grid clamps invalid cell size to one unit", "[unit][survivors][logic][target-grid]")
{
	FSurvivorsTargetGrid Grid;
	Grid.Rebuild(FVector2D::ZeroVector, 2.f, 0.f);

	CHECK(Grid.CellSize == 1.f);
	CHECK(Grid.NumX == 4);
	CHECK(Grid.NumY == 4);
	CHECK(Grid.Cells.Num() == 16);
}

TEST_CASE("Survivors target grid query before rebuild is empty", "[unit][survivors][logic][target-grid]")
{
	FSurvivorsTargetGrid Grid;
	TArray<int32> Results;

	Grid.QueryContacts(FVector2D::ZeroVector, 100.f, Results);

	CHECK(Results.IsEmpty());
}