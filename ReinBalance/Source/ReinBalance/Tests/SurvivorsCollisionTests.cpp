#include "Misc/AutomationTest.h"
#include "Engine/World.h"
#include "Survivors/Logic/SurvivorsGame.h"
#include "Survivors/Logic/SurvivorsTypes.h"
#include "Survivors/Logic/SurvivorsGameConstants.h"
#include "Survivors/Logic/SurvivorsCollisionComponent.h"
#include "Survivors/Logic/SurvivorsEnemyComponent.h"
#include "Survivors/Logic/SurvivorsGemComponent.h"
#include "Survivors/Logic/Weapons/SurvivorsWeaponComponent.h"

// ============================================================
// テストヘルパー: ASurvivorsGame の private 状態へのアクセサ
// (SurvivorsGame.h の WITH_AUTOMATION_TESTS friend 宣言が必要)
// ============================================================
struct FSurvivorsGameTestAccess
{
	static TArray<FEnemyState>& Enemies(ASurvivorsGame* G) { return G->Enemies; }
	static TArray<FGemState>&   Gems(ASurvivorsGame* G)    { return G->Gems; }
	static float& PlayerHP(ASurvivorsGame* G)              { return G->PlayerHP; }
	static FVector2D& PlayerPos(ASurvivorsGame* G)         { return G->PlayerPos; }
	static bool& bShieldActive(ASurvivorsGame* G)          { return G->bShieldActive; }
	static float& LastReward(ASurvivorsGame* G)            { return G->LastReward; }
	static float& ElapsedTime(ASurvivorsGame* G)           { return G->ElapsedTime; }
	static float& GemPickupRadius(ASurvivorsGame* G)       { return G->GemPickupRadius; }
	static int32& NextEnemyId(ASurvivorsGame* G)           { return G->NextEnemyId; }
	static int32& NextGemId(ASurvivorsGame* G)             { return G->NextGemId; }

	static USurvivorsCollisionComponent* CollComp(ASurvivorsGame* G) { return G->CollisionComponent; }
	static USurvivorsEnemyComponent*     EnemyComp(ASurvivorsGame* G){ return G->EnemyComponent; }
	static USurvivorsGemComponent*       GemComp(ASurvivorsGame* G)  { return G->GemComponent; }
	static USurvivorsWeaponComponent*    WeaponComp(ASurvivorsGame* G){ return G->WeaponComponent; }

	static void FinalizePendingEnemies(ASurvivorsGame* G)  { G->FinalizePendingEnemies(); }
	static void FinalizePickupRemovals(ASurvivorsGame* G)  { G->FinalizePickupRemovals(); }
};

// ============================================================
// テストワールドライフサイクル
// ============================================================
struct FSurvivorsTestWorld
{
	UWorld*       World = nullptr;
	ASurvivorsGame* Game = nullptr;

	bool Create()
	{
		if (!GEngine) return false;
		World = UWorld::CreateWorld(EWorldType::Game, false);
		if (!World) return false;
		FWorldContext& Ctx = GEngine->CreateNewWorldContext(EWorldType::Game);
		Ctx.SetCurrentWorld(World);
		World->InitializeActorsForPlay(FURL(), true);
		FActorSpawnParameters P;
		P.bNoFail = true;
		Game = World->SpawnActor<ASurvivorsGame>(P);
		if (!IsValid(Game)) return false;
		// BeginPlay が呼ばれない場合のフォールバック（テスト環境ではタイミングが異なる）
		// ResetState は public なので常に明示的に呼び出して初期状態を保証する
		Game->ResetState(TOptional<int32>());
		return true;
	}

	void Destroy()
	{
		if (World && GEngine)
		{
			GEngine->DestroyWorldContext(World);
			World->DestroyWorld(false);
		}
		World = nullptr;
		Game = nullptr;
	}

	// プレイヤー位置に近い敵を追加する
	int32 AddEnemyAt(FVector2D Pos, float HP = 100.f, float ContactDamage = 5.f, float Radius = 5.f)
	{
		FEnemyState E;
		E.UniqueId        = ++FSurvivorsGameTestAccess::NextEnemyId(Game);
		E.Pos             = Pos;
		E.CollisionRadius = Radius;
		E.HP              = HP;
		E.MaxHP           = HP;
		E.ContactDamage   = ContactDamage;
		E.TypeId          = 0;
		FSurvivorsGameTestAccess::Enemies(Game).Add(E);
		return E.UniqueId;
	}

	int32 AddEnemyNearPlayer(float HP = 100.f, float ContactDamage = 5.f, float Radius = 5.f)
	{
		return AddEnemyAt(FSurvivorsGameTestAccess::PlayerPos(Game), HP, ContactDamage, Radius);
	}

	// Weapon/Enemy/Gem の一連の当たり判定ステップを実行するヘルパー
	void RunWeaponHits()
	{
		auto* CC = FSurvivorsGameTestAccess::CollComp(Game);
		auto* WC = FSurvivorsGameTestAccess::WeaponComp(Game);
		CC->BuildEnemyGrid();
		FSurvivorsHitFrame HF;
		WC->ComputeAllWeaponHits(CC, HF);
		WC->ApplyWeaponHits(HF);
	}

	void RunContactHits()
	{
		auto* CC = FSurvivorsGameTestAccess::CollComp(Game);
		auto* EC = FSurvivorsGameTestAccess::EnemyComp(Game);
		CC->BuildEnemyGrid();
		FSurvivorsHitFrame HF;
		EC->ComputeContactHits(CC, HF);
		EC->ApplyContactHits(HF);
	}

	void RunPickupHits()
	{
		auto* CC = FSurvivorsGameTestAccess::CollComp(Game);
		auto* GC = FSurvivorsGameTestAccess::GemComp(Game);
		CC->BuildPickupGrid();
		FSurvivorsHitFrame HF;
		GC->ComputePickupHits(CC, HF);
		GC->ApplyPickupHits(HF);
	}
};

// ============================================================
// ① FSurvivorsTargetGrid: 純粋構造体テスト（ワールド不要）
// ============================================================

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsGridBoundaryQuery,
	"ReinBalance.Survivors.Grid.BoundaryQuery",
	EAutomationTestFlags::ApplicationContextMask | EAutomationTestFlags::SmokeFilter)

bool FSurvivorsGridBoundaryQuery::RunTest(const FString& Parameters)
{
	FSurvivorsTargetGrid Grid;
	Grid.Rebuild(FVector2D::ZeroVector, 512.f, 128.f);

	FSurvivorsTargetProxy P;
	P.Ref    = { ESurvivorsCollisionOwnerKind::Enemy, 1, 0 };
	P.Pos    = FVector2D(128.f, 0.f);
	P.Radius = 20.f;
	TestTrue("AddTarget succeeds", Grid.AddTarget(P));

	TArray<int32> Results;
	Grid.QueryContacts(FVector2D::ZeroVector, 148.f, Results); // 128 + 20 = 148
	TestTrue("Boundary target found via QueryRadius=148", Results.Num() > 0);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsGridOutsideReject,
	"ReinBalance.Survivors.Grid.OutsideGridRejected",
	EAutomationTestFlags::ApplicationContextMask | EAutomationTestFlags::SmokeFilter)

bool FSurvivorsGridOutsideReject::RunTest(const FString& Parameters)
{
	FSurvivorsTargetGrid Grid;
	Grid.Rebuild(FVector2D::ZeroVector, 512.f, 128.f);

	FSurvivorsTargetProxy P;
	P.Ref = { ESurvivorsCollisionOwnerKind::Enemy, 2, 0 };
	P.Pos = FVector2D(2000.f, 0.f);
	P.Radius = 10.f;
	TestFalse("Target outside HalfExtent is rejected", Grid.AddTarget(P));
	return true;
}

// ② bUseFullFieldCollision: グリッド中心が ZeroVector になること（純粋構造体で検証）
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsGridFullFieldCoversWorldBounds,
	"ReinBalance.Survivors.Grid.FullFieldCoversWorldBounds",
	EAutomationTestFlags::ApplicationContextMask | EAutomationTestFlags::SmokeFilter)

bool FSurvivorsGridFullFieldCoversWorldBounds::RunTest(const FString& Parameters)
{
	const float FieldHalfSize = 1000.f;

	// 通常モード: PlayerPos=(800,0) 中心、HalfExtent=200 → 原点付近の敵は範囲外
	{
		FSurvivorsTargetGrid Grid;
		Grid.Rebuild(FVector2D(800.f, 0.f), 200.f, 128.f);

		FSurvivorsTargetProxy P;
		P.Ref = { ESurvivorsCollisionOwnerKind::Enemy, 1, 0 };
		P.Pos = FVector2D::ZeroVector;
		P.Radius = 5.f;
		TestFalse("Normal mode: enemy at origin outside PlayerPos(800)+HalfExtent(200) grid", Grid.AddTarget(P));
	}

	// FullField モード: ZeroVector 中心、HalfExtent=FieldHalfSize → 原点の敵が範囲内
	{
		FSurvivorsTargetGrid Grid;
		Grid.Rebuild(FVector2D::ZeroVector, FieldHalfSize, 128.f);

		FSurvivorsTargetProxy P;
		P.Ref = { ESurvivorsCollisionOwnerKind::Enemy, 1, 0 };
		P.Pos = FVector2D::ZeroVector;
		P.Radius = 5.f;
		TestTrue("FullField mode: enemy at origin inside ZeroVector+FieldHalfSize grid", Grid.AddTarget(P));

		// 原点付近を query すれば拾える（FullField でもクエリ半径内にある必要がある）
		TArray<int32> Results;
		Grid.QueryContacts(FVector2D::ZeroVector, 10.f, Results);
		TestEqual("FullField query at origin finds enemy", Results.Num(), 1);

		// フィールド端 (900,0) の敵も登録できる（通常モードでは PlayerPos±200 外になる）
		FSurvivorsTargetProxy P2;
		P2.Ref = { ESurvivorsCollisionOwnerKind::Enemy, 2, 1 };
		P2.Pos = FVector2D(900.f, 0.f);
		P2.Radius = 5.f;
		TestTrue("FullField mode: enemy at field edge (900,0) also inside grid", Grid.AddTarget(P2));
	}
	return true;
}

// ============================================================
// ③ 統合テスト: HitFrame / 報酬 / 削除（ワールド必要）
// ============================================================

// ContactDamage に EnemyDamageScale が二重適用されないこと
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsContactDamageNoDoubleScale,
	"ReinBalance.Survivors.HitFrame.ContactDamage_NoDoubleEnemyDamageScale",
	EAutomationTestFlags::ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FSurvivorsContactDamageNoDoubleScale::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	// EnemyDamageScale=2 でも ContactDamage はスポーン時に適用済みなので再乗算しない
	S.Game->EnemyDamageScale = 2.f;
	FSurvivorsGameTestAccess::PlayerHP(S.Game) = 100.f;
	S.AddEnemyNearPlayer(/*HP=*/100.f, /*ContactDamage=*/5.f);

	S.RunContactHits();

	// 期待: 100 - 5 = 95（5*2=10 ではない）
	TestEqual("PlayerHP reduced by Damage(5) only, not Damage*Scale(10)",
		FSurvivorsGameTestAccess::PlayerHP(S.Game), 95.f);

	S.Destroy();
	return true;
}

// Laurel シールド中は PlayerHP も PlayerLastHitTime も変化しないこと
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsShieldBlocksContact,
	"ReinBalance.Survivors.HitFrame.Shield_DoesNotUpdatePlayerLastHitTime",
	EAutomationTestFlags::ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FSurvivorsShieldBlocksContact::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	const float InitialHP = FSurvivorsGameTestAccess::PlayerHP(S.Game);
	S.AddEnemyNearPlayer();

	// シールド有効: ダメージなし、PlayerLastHitTime 更新なし
	FSurvivorsGameTestAccess::bShieldActive(S.Game) = true;
	S.RunContactHits();

	TestEqual("PlayerHP unchanged with shield", FSurvivorsGameTestAccess::PlayerHP(S.Game), InitialHP);
	const float LastHitAfterShield = FSurvivorsGameTestAccess::Enemies(S.Game)[0].PlayerLastHitTime;
	TestTrue("PlayerLastHitTime not updated while shielded",
		LastHitAfterShield < -100.f); // 初期値は -1000.f

	// シールド解除後: 即ダメージ（cooldown 未消費のため）
	FSurvivorsGameTestAccess::bShieldActive(S.Game) = false;
	S.RunContactHits();

	TestTrue("PlayerHP decreased after shield dropped",
		FSurvivorsGameTestAccess::PlayerHP(S.Game) < InitialHP);

	S.Destroy();
	return true;
}

// Garlic が敵を倒し、ドロップしたジェムが同 tick 内に回収されること
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsGarlicKillDropCollectSameTick,
	"ReinBalance.Survivors.HitFrame.Garlic_KillsEnemy_DropGemCollectedSameTick",
	EAutomationTestFlags::ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FSurvivorsGarlicKillDropCollectSameTick::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	// Garlic Lv1: AreaRadius=25u, Damage=5
	// 敵を HP=1 でプレイヤー位置(0,0)に置く → 1発で死ぬ
	S.AddEnemyNearPlayer(/*HP=*/1.f);
	FSurvivorsGameTestAccess::GemPickupRadius(S.Game) = 50.f; // 十分大きい
	FSurvivorsGameTestAccess::LastReward(S.Game) = 0.f;

	const float RewardBefore = FSurvivorsGameTestAccess::LastReward(S.Game);

	// WeaponHits → FinalizePendingEnemies(DropGem) → BuildPickupGrid → PickupHits
	S.RunWeaponHits();
	FSurvivorsGameTestAccess::FinalizePendingEnemies(S.Game);
	S.RunPickupHits();
	FSurvivorsGameTestAccess::FinalizePickupRemovals(S.Game);

	TestEqual("Enemy array empty after kill",    FSurvivorsGameTestAccess::Enemies(S.Game).Num(), 0);
	TestEqual("Gem array empty after collection", FSurvivorsGameTestAccess::Gems(S.Game).Num(), 0);
	TestTrue("KillReward + ItemReward accumulated",
		FSurvivorsGameTestAccess::LastReward(S.Game) > RewardBefore);

	S.Destroy();
	return true;
}

// StepSpawn 後に追加された敵は同 tick の WeaponHits に参加しないこと
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsSpawnedEnemyNotHitSameTick,
	"ReinBalance.Survivors.HitFrame.SpawnedEnemy_NotHitByWeaponOnSpawnTick",
	EAutomationTestFlags::ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FSurvivorsSpawnedEnemyNotHitSameTick::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	// 初期状態: 敵なし。BuildEnemyGrid でスナップショットを取る（空）
	FSurvivorsGameTestAccess::CollComp(S.Game)->BuildEnemyGrid();

	// StepSpawn 相当: 敵を後から追加（Garlic 範囲内）
	S.AddEnemyNearPlayer(/*HP=*/1.f);

	// WeaponHits はすでに BuildEnemyGrid 済みのグリッドを使うので新規敵はヒット対象外
	auto* WC = FSurvivorsGameTestAccess::WeaponComp(S.Game);
	auto* CC = FSurvivorsGameTestAccess::CollComp(S.Game);
	FSurvivorsHitFrame HF;
	WC->ComputeAllWeaponHits(CC, HF);
	WC->ApplyWeaponHits(HF);

	// スポーン後に追加した敵のHPは変わっていないはず
	TestTrue("Newly added enemy HP unchanged (not in pre-spawn grid)",
		FSurvivorsGameTestAccess::Enemies(S.Game).Num() == 1 &&
		FSurvivorsGameTestAccess::Enemies(S.Game)[0].HP > 0.f);

	S.Destroy();
	return true;
}

// UniqueId 不一致の場合は Apply がスキップされること
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsUniqueIdMismatchSkipped,
	"ReinBalance.Survivors.HitFrame.UniqueId_Mismatch_NotApplied",
	EAutomationTestFlags::ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FSurvivorsUniqueIdMismatchSkipped::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	S.AddEnemyNearPlayer(/*HP=*/100.f);

	// HitFrame を生成（この時点では UniqueId 一致）
	auto* CC = FSurvivorsGameTestAccess::CollComp(S.Game);
	auto* WC = FSurvivorsGameTestAccess::WeaponComp(S.Game);
	CC->BuildEnemyGrid();
	FSurvivorsHitFrame HF;
	WC->ComputeAllWeaponHits(CC, HF);
	TestTrue("Has hit events before corruption", HF.Events.Num() > 0);

	// 配列変化を模倣: UniqueId を変える
	FSurvivorsGameTestAccess::Enemies(S.Game)[0].UniqueId = 9999;
	const float HPBefore = FSurvivorsGameTestAccess::Enemies(S.Game)[0].HP;

	WC->ApplyWeaponHits(HF);

	TestEqual("Damage not applied when UniqueId mismatches",
		FSurvivorsGameTestAccess::Enemies(S.Game)[0].HP, HPBefore);

	S.Destroy();
	return true;
}

// Garlic hit interval: 同一 tick 2回呼んでも1回分しかヒットしないこと
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsGarlicHitInterval,
	"ReinBalance.Survivors.HitFrame.Garlic_HitInterval_PreventsDuplicateHit",
	EAutomationTestFlags::ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FSurvivorsGarlicHitInterval::RunTest(const FString& Parameters)
{
	using namespace SurvivorsGameConstants;
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	const float GarlicDmg = GarlicTable[0].Damage.Value; // Lv1: 5
	S.AddEnemyNearPlayer(/*HP=*/200.f);

	// 1回目ヒット
	S.RunWeaponHits();
	const float HPAfterFirst = FSurvivorsGameTestAccess::Enemies(S.Game)[0].HP;
	TestTrue("First hit deals damage", FMath::IsNearlyEqual(HPAfterFirst, 200.f - GarlicDmg, 0.01f));

	// HitInterval 未満で再度呼ぶ → ヒットなし
	S.RunWeaponHits();
	TestEqual("Second hit within interval has no effect",
		FSurvivorsGameTestAccess::Enemies(S.Game)[0].HP, HPAfterFirst);

	// HitInterval を超えて経過 → 再ヒット
	FSurvivorsGameTestAccess::ElapsedTime(S.Game) += GarlicTable[0].HitInterval + 0.01f;
	S.RunWeaponHits();
	TestTrue("Hit after interval deals damage again",
		FMath::IsNearlyEqual(FSurvivorsGameTestAccess::Enemies(S.Game)[0].HP, HPAfterFirst - GarlicDmg, 0.01f));

	S.Destroy();
	return true;
}

// Garlic knockback: 命中後に敵が KnockbackDir * Strength * (1-Resistance) だけ移動すること
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsGarlicKnockback,
	"ReinBalance.Survivors.HitFrame.Garlic_Knockback",
	EAutomationTestFlags::ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FSurvivorsGarlicKnockback::RunTest(const FString& Parameters)
{
	using namespace SurvivorsGameConstants;
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	// 敵を X+10 位置に置く → knockback 方向 = (+1, 0)
	S.AddEnemyAt(FVector2D(10.f, 0.f), /*HP=*/200.f);
	const float ExpectedDelta = GarlicKnockbackStrength * (1.f - 0.f); // Resistance=0

	S.RunWeaponHits();

	const FVector2D EnemyPos = FSurvivorsGameTestAccess::Enemies(S.Game)[0].Pos;
	TestTrue("Enemy knocked back along X", FMath::IsNearlyEqual(EnemyPos.X, 10.f + ExpectedDelta, 0.5f));
	TestTrue("No Y knockback", FMath::IsNearlyEqual(EnemyPos.Y, 0.f, 0.01f));

	S.Destroy();
	return true;
}

// KnockbackResistance=1 なら位置が変わらないこと
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsGarlicKnockbackResistance,
	"ReinBalance.Survivors.HitFrame.Garlic_Knockback_FullResistance",
	EAutomationTestFlags::ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FSurvivorsGarlicKnockbackResistance::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	S.AddEnemyAt(FVector2D(10.f, 0.f), /*HP=*/200.f);
	// EnemyTypeTable[0] に KnockbackResistance=1 を設定
	if (S.Game->EnemyTypeTable.IsValidIndex(0))
		S.Game->EnemyTypeTable[0].KnockbackResistance = 1.f;

	S.RunWeaponHits();

	TestTrue("Enemy position unchanged with full resistance",
		FMath::IsNearlyEqual(FSurvivorsGameTestAccess::Enemies(S.Game)[0].Pos.X, 10.f, 0.01f));

	S.Destroy();
	return true;
}

// 非 piercing 弾: 複数の敵がいても1体だけヒットすること
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsProjectileNonPiercingSingleHit,
	"ReinBalance.Survivors.HitFrame.Projectile_NonPiercing_SingleHit",
	EAutomationTestFlags::ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FSurvivorsProjectileNonPiercingSingleHit::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	// 敵2体を同じ位置に配置
	S.AddEnemyAt(FVector2D(50.f, 0.f), 100.f);
	S.AddEnemyAt(FVector2D(50.f, 0.f), 100.f);

	// 非 piercing プロジェクタイルを追加
	FProjectileState Proj;
	Proj.Pos          = FVector2D(50.f, 0.f);
	Proj.Radius       = FSimRadius(10.f);
	Proj.Damage       = FDamage(20.f);
	Proj.bPiercing    = false;
	Proj.WeaponType   = EWeaponType::MagicWand;
	Proj.WeaponSlotIdx= 1;
	Proj.LifeTime     = FProjectileLifeTime(5.f);
	FSurvivorsGameTestAccess::WeaponComp(S.Game)->SpawnProjectile(Proj);

	auto* CC = FSurvivorsGameTestAccess::CollComp(S.Game);
	auto* WC = FSurvivorsGameTestAccess::WeaponComp(S.Game);
	CC->BuildEnemyGrid();
	FSurvivorsHitFrame HF;
	WC->ComputeAllWeaponHits(CC, HF);

	// 非 piercing → HitFrame に ProjectileDamage は 1件のみ
	int32 ProjDmgCount = 0;
	for (const auto& Ev : HF.Events)
		if (Ev.Type == ESurvivorsHitType::ProjectileDamage) ++ProjDmgCount;
	TestEqual("Non-piercing projectile generates exactly 1 damage event", ProjDmgCount, 1);

	WC->ApplyWeaponHits(HF);

	// 2体のうち1体だけダメージ
	int32 DamagedCount = 0;
	for (const auto& E : FSurvivorsGameTestAccess::Enemies(S.Game))
		if (E.HP < 100.f) ++DamagedCount;
	TestEqual("Only 1 enemy takes damage from non-piercing projectile", DamagedCount, 1);

	S.Destroy();
	return true;
}

// GroundZone HitCooldown: cooldown 内は再ヒットしないこと
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsGroundZoneHitCooldown,
	"ReinBalance.Survivors.HitFrame.GroundZone_HitCooldown",
	EAutomationTestFlags::ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FSurvivorsGroundZoneHitCooldown::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	S.AddEnemyAt(FVector2D(50.f, 0.f), 200.f);

	// GroundZone を追加
	FGroundZoneState Zone;
	Zone.Pos          = FVector2D(50.f, 0.f);
	Zone.Radius       = 30.f;
	Zone.Damage       = 10.f;
	Zone.LifeTime     = 10.f;
	Zone.HitCooldown  = 0.5f;
	Zone.WeaponType   = EWeaponType::SantaWater;
	Zone.WeaponSlotIdx= 2;
	FSurvivorsGameTestAccess::WeaponComp(S.Game)->SpawnGroundZone(Zone);

	// 1回目ヒット
	S.RunWeaponHits();
	const float HPAfterFirst = FSurvivorsGameTestAccess::Enemies(S.Game)[0].HP;
	TestTrue("First GroundZone hit deals damage", FMath::IsNearlyEqual(HPAfterFirst, 190.f, 0.01f));

	// cooldown 内で再実行 → ヒットなし
	S.RunWeaponHits();
	TestEqual("No second hit within cooldown", FSurvivorsGameTestAccess::Enemies(S.Game)[0].HP, HPAfterFirst);

	// cooldown 経過後 → 再ヒット
	FSurvivorsGameTestAccess::ElapsedTime(S.Game) += 0.6f;
	S.RunWeaponHits();
	TestTrue("Hit resumes after cooldown",
		FMath::IsNearlyEqual(FSurvivorsGameTestAccess::Enemies(S.Game)[0].HP, HPAfterFirst - 10.f, 0.01f));

	S.Destroy();
	return true;
}
