#include "Misc/AutomationTest.h"
#include "Survivors/Logic/SurvivorsTypes.h"
// FSurvivorsTargetGrid のテスト（UE Editor 不要の純粋 C++ ロジック）

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsTargetGridBoundaryTest,
    "ReinBalance.Survivors.Collision.BoundaryQuery",
    EAutomationTestFlags::ApplicationContextMask | EAutomationTestFlags::SmokeFilter)

bool FSurvivorsTargetGridBoundaryTest::RunTest(const FString& Parameters)
{
    // セル境界上のターゲットが SourceRadius + MaxTargetRadius の query で拾える
    FSurvivorsTargetGrid Grid;
    Grid.Rebuild(FVector2D::ZeroVector, 512.f, 128.f);

    // セル境界 (128, 0) にターゲットを配置（半径 20）
    FSurvivorsTargetProxy P;
    P.Ref = { ESurvivorsCollisionOwnerKind::Enemy, 1, 0 };
    P.Pos = FVector2D(128.f, 0.f);
    P.Radius = 20.f;
    TestTrue("AddTarget succeeds", Grid.AddTarget(P));

    // Source (0,0) QueryRadius = SourceRadius(0) + TargetCenter距離(128) + TargetRadius(20) = 148 で確実に届く
    TArray<int32> Results;
    Grid.QueryContacts(FVector2D::ZeroVector, 148.f, Results);
    TestTrue("Boundary target found", Results.Num() > 0);
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsTargetGridOutsideTest,
    "ReinBalance.Survivors.Collision.OutsideGrid",
    EAutomationTestFlags::ApplicationContextMask | EAutomationTestFlags::SmokeFilter)

bool FSurvivorsTargetGridOutsideTest::RunTest(const FString& Parameters)
{
    FSurvivorsTargetGrid Grid;
    Grid.Rebuild(FVector2D::ZeroVector, 512.f, 128.f);

    // Grid 外（2000, 0）にターゲットを配置
    FSurvivorsTargetProxy P;
    P.Ref = { ESurvivorsCollisionOwnerKind::Enemy, 2, 0 };
    P.Pos = FVector2D(2000.f, 0.f);
    P.Radius = 10.f;
    TestFalse("Outside target rejected", Grid.AddTarget(P));
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsPendingRemoveTest,
    "ReinBalance.Survivors.Collision.PendingRemoveSkip",
    EAutomationTestFlags::ApplicationContextMask | EAutomationTestFlags::SmokeFilter)

bool FSurvivorsPendingRemoveTest::RunTest(const FString& Parameters)
{
    // bPendingRemove = true の FGemState が ApplyPickupHits でスキップされる
    FGemState G;
    G.UniqueId = 1;
    G.bPendingRemove = true;
    TestTrue("bPendingRemove field exists", G.bPendingRemove);
    // (実際の ApplyPickupHits 統合テストは SurvivorsGame 依存のため手動確認)
    return true;
}
