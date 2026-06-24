#include "Misc/AutomationTest.h"
#include "SurvivorsTestHelpers.h"

// 遠くにいる通常敵がテレポートされること
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsEnemyRecycleFarEnemyTeleports,
	"ReinBalance.Survivors.EnemyRecycle.FarEnemy_Teleports",
	EAutomationTestFlags::ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FSurvivorsEnemyRecycleFarEnemyTeleports::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	// EnemyRecycleDistance=1000, SpawnMinDistance=500, SpawnMaxDistance=700 に設定
	FSurvivorsGameLogic* Logic = FSurvivorsGameTestAccess::GetLogic(S.Game);
	Logic->CurrentConfig.EnemyRecycleDistance = 1000.f;
	Logic->CurrentConfig.SpawnMinDistance     = 500.f;
	Logic->CurrentConfig.SpawnMaxDistance     = 700.f;

	// プレイヤーを (0,0)、敵を (1200, 0) に配置（RecycleDistance を超える）
	FSurvivorsGameTestAccess::PlayerPos(S.Game) = FVector2D(0.f, 0.f);
	S.AddEnemyAt(FVector2D(1200.f, 0.f), /*HP=*/100.f);

	// PhysicsStep を 1 回実行
	S.Game->PhysicsStep(8); // 8 = 静止

	const TArray<FEnemyState>& Enemies = FSurvivorsGameTestAccess::Enemies(S.Game);
	if (!TestTrue("Enemy still exists", Enemies.Num() > 0)) { S.Destroy(); return false; }

	const FVector2D NewPos = Enemies[0].Pos;

	// 敵の位置が (1200, 0) から変化していること（テレポートされた）
	TestTrue("Enemy teleported away from (1200,0)",
		!FMath::IsNearlyEqual(NewPos.X, 1200.f, 10.f) || !FMath::IsNearlyEqual(NewPos.Y, 0.f, 10.f));

	// 敵とプレイヤーの距離が SpawnMinDistance〜SpawnMaxDistance の範囲内（クランプを考慮して 490〜710 程度）
	const float Dist = NewPos.Size();
	TestTrue("Enemy teleported to spawn ring (490~710)",
		Dist >= 490.f && Dist <= 710.f);

	S.Destroy();
	return true;
}

// 近くにいる通常敵がテレポートされないこと
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsEnemyRecycleNearEnemyNoTeleport,
	"ReinBalance.Survivors.EnemyRecycle.NearEnemy_NoTeleport",
	EAutomationTestFlags::ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FSurvivorsEnemyRecycleNearEnemyNoTeleport::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	FSurvivorsGameLogic* Logic = FSurvivorsGameTestAccess::GetLogic(S.Game);
	Logic->CurrentConfig.EnemyRecycleDistance = 1000.f;
	Logic->CurrentConfig.SpawnMinDistance     = 500.f;
	Logic->CurrentConfig.SpawnMaxDistance     = 700.f;

	// プレイヤーを (0,0)、敵を (600, 0) に配置（RecycleDistance 以内）
	FSurvivorsGameTestAccess::PlayerPos(S.Game) = FVector2D(0.f, 0.f);
	S.AddEnemyAt(FVector2D(600.f, 0.f), /*HP=*/100.f);

	// PhysicsStep を 1 回実行
	S.Game->PhysicsStep(8); // 8 = 静止

	const TArray<FEnemyState>& Enemies = FSurvivorsGameTestAccess::Enemies(S.Game);
	if (!TestTrue("Enemy still exists", Enemies.Num() > 0)) { S.Destroy(); return false; }

	const FVector2D NewPos = Enemies[0].Pos;

	// 敵は1ステップで約0.583u移動するだけなので EnemyRecycleDistance(1000u) 以内に留まるはず
	// テレポートされていないことを確認（元の600u付近、EnemyRecycleDistance未満）
	const float DistSq = NewPos.SizeSquared();
	TestTrue("Near enemy not teleported (dist < EnemyRecycleDistance=1000)",
		DistSq < FMath::Square(1000.f) && DistSq > 0.f);

	S.Destroy();
	return true;
}

// ボス敵はリサイクルされないこと
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsEnemyRecycleBossNotRecycled,
	"ReinBalance.Survivors.EnemyRecycle.Boss_NotRecycled",
	EAutomationTestFlags::ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FSurvivorsEnemyRecycleBossNotRecycled::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	FSurvivorsGameLogic* Logic = FSurvivorsGameTestAccess::GetLogic(S.Game);
	Logic->CurrentConfig.EnemyRecycleDistance = 1000.f;

	// TypeId=10 のエントリを bIsBoss=true に設定
	TArray<FEnemyTypeParams>& EnemyTypeTable = Logic->CurrentConfig.EnemyTypeTable;
	// テーブルが TypeId=10 を持つように拡張
	while (EnemyTypeTable.Num() <= 10)
	{
		EnemyTypeTable.Add(FEnemyTypeParams());
	}
	EnemyTypeTable[10].bIsBoss = true;

	// ボス敵をプレイヤーから (1500, 0) に配置
	FSurvivorsGameTestAccess::PlayerPos(S.Game) = FVector2D(0.f, 0.f);
	FEnemyState BossEnemy;
	BossEnemy.UniqueId        = ++FSurvivorsGameTestAccess::NextEnemyId(S.Game);
	BossEnemy.Pos             = FVector2D(1500.f, 0.f);
	BossEnemy.CollisionRadius = 10.f;
	BossEnemy.HP              = 1000.f;
	BossEnemy.MaxHP           = 1000.f;
	BossEnemy.ContactDamage   = 10.f;
	BossEnemy.TypeId          = 10;
	FSurvivorsGameTestAccess::Enemies(S.Game).Add(BossEnemy);

	// PhysicsStep を 1 回実行
	S.Game->PhysicsStep(8); // 8 = 静止

	const TArray<FEnemyState>& Enemies = FSurvivorsGameTestAccess::Enemies(S.Game);
	if (!TestTrue("Boss enemy still exists", Enemies.Num() > 0)) { S.Destroy(); return false; }

	// ボス敵が 1000u 以上の距離にまだいること（再配置されていない）
	const float Dist = Enemies[0].Pos.Size();
	TestTrue("Boss not recycled (dist > 1000)",
		Dist > 1000.f);

	S.Destroy();
	return true;
}
