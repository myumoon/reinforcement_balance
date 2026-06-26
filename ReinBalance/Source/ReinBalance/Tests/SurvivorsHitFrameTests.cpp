#include "Misc/AutomationTest.h"
#include "SurvivorsTestHelpers.h"

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
	FSurvivorsHitFrame HF;
	FSurvivorsGameTestAccess::ComputeAllWeaponHits(S.Game, HF);
	FSurvivorsGameTestAccess::ApplyWeaponHits(S.Game, HF);

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
	FSurvivorsGameTestAccess::BuildEnemyGrid(S.Game);
	FSurvivorsHitFrame HF;
	FSurvivorsGameTestAccess::ComputeAllWeaponHits(S.Game, HF);
	TestTrue("Has hit events before corruption", HF.Events.Num() > 0);

	// 配列変化を模倣: UniqueId を変える
	FSurvivorsGameTestAccess::Enemies(S.Game)[0].UniqueId = 9999;
	const float HPBefore = FSurvivorsGameTestAccess::Enemies(S.Game)[0].HP;

	FSurvivorsGameTestAccess::ApplyWeaponHits(S.Game, HF);

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

// Garlic knockback: 命中後に KnockbackFramesLeft=7 が設定され、7回 UpdateEnemies 後に期待位置に到達すること
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

	// RunWeaponHits 直後: 位置変化なし、KnockbackFramesLeft == 7
	const FEnemyState& EAfterHit = FSurvivorsGameTestAccess::Enemies(S.Game)[0];
	TestTrue("Enemy position unchanged immediately after hit",
		FMath::IsNearlyEqual(EAfterHit.Pos.X, 10.f, 0.01f));
	TestEqual("KnockbackFramesLeft == 7 after hit",
		EAfterHit.KnockbackFramesLeft, KnockbackFrames);

	// 7 回 UpdateEnemies 後: 期待位置に到達
	for (int32 i = 0; i < KnockbackFrames; ++i)
		S.RunUpdateEnemies();

	const FVector2D EnemyPos = FSurvivorsGameTestAccess::Enemies(S.Game)[0].Pos;
	TestTrue("Enemy knocked back along X after 7 frames",
		FMath::IsNearlyEqual(EnemyPos.X, 10.f + ExpectedDelta, 0.5f));
	TestTrue("No Y knockback", FMath::IsNearlyEqual(EnemyPos.Y, 0.f, 0.01f));

	S.Destroy();
	return true;
}

// KnockbackResistance=1 なら KnockbackFramesLeft=7 が設定されるが、7f後も位置変化なし（EffKB=0 のため）
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsGarlicKnockbackResistance,
	"ReinBalance.Survivors.HitFrame.Garlic_Knockback_FullResistance",
	EAutomationTestFlags::ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FSurvivorsGarlicKnockbackResistance::RunTest(const FString& Parameters)
{
	using namespace SurvivorsGameConstants;
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	S.AddEnemyAt(FVector2D(10.f, 0.f), /*HP=*/200.f);
	// EnemyTypeTable[0] に KnockbackResistance=1 を設定
	FSurvivorsGameLogic* Logic = FSurvivorsGameTestAccess::GetLogic(S.Game);
	TArray<FEnemyTypeParams>& EnemyTypeTable = Logic->CurrentConfig.EnemyTypeTable;
	if (EnemyTypeTable.IsValidIndex(0))
		EnemyTypeTable[0].KnockbackResistance = 1.f;

	S.RunWeaponHits();

	// Resistance=1 でも KnockbackFramesLeft=7 が設定される（7f停止は発生する）
	TestEqual("KnockbackFramesLeft == 7 even with full resistance",
		FSurvivorsGameTestAccess::Enemies(S.Game)[0].KnockbackFramesLeft, KnockbackFrames);

	// 7 回 UpdateEnemies 後: 位置は変化なし（EffKB=0 のため KnockbackVelPerFrame=zero）
	for (int32 i = 0; i < KnockbackFrames; ++i)
		S.RunUpdateEnemies();

	TestTrue("Enemy position unchanged with full resistance after 7 frames",
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
	S.Game->GetLogic()->SpawnProjectile(Proj);

	FSurvivorsGameTestAccess::BuildEnemyGrid(S.Game);
	FSurvivorsHitFrame HF;
	FSurvivorsGameTestAccess::ComputeAllWeaponHits(S.Game, HF);

	// 非 piercing → HitFrame に ProjectileDamage は 1件のみ
	int32 ProjDmgCount = 0;
	for (const auto& Ev : HF.Events)
		if (Ev.Type == ESurvivorsHitType::ProjectileDamage) ++ProjDmgCount;
	TestEqual("Non-piercing projectile generates exactly 1 damage event", ProjDmgCount, 1);

	FSurvivorsGameTestAccess::ApplyWeaponHits(S.Game, HF);

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
	S.Game->GetLogic()->SpawnGroundZone(Zone);

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

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsGroundZoneWarningNoDamage,
	"ReinBalance.Survivors.HitFrame.GroundZone_WarningNoDamage",
	EAutomationTestFlags::ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FSurvivorsGroundZoneWarningNoDamage::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	S.AddEnemyAt(FVector2D(50.f, 0.f), 200.f);

	FGroundZoneState Zone;
	Zone.Pos          = FVector2D(50.f, 0.f);
	Zone.Radius       = 30.f;
	Zone.Damage       = 10.f;
	Zone.LifeTime     = 2.f;
	Zone.WarningTime  = 1.f;
	Zone.HitCooldown  = 0.5f;
	Zone.WeaponType   = EWeaponType::SantaWater;
	Zone.WeaponSlotIdx= 2;
	Zone.bIsWarning   = true;
	S.Game->GetLogic()->SpawnGroundZone(Zone);

	S.RunWeaponHits();
	TestEqual("Warning GroundZone does not damage", FSurvivorsGameTestAccess::Enemies(S.Game)[0].HP, 200.f);

	FSurvivorsGameTestAccess::TickWeapons(S.Game, 1.01f);
	S.RunWeaponHits();
	TestTrue("GroundZone damages after warning expires",
		FMath::IsNearlyEqual(FSurvivorsGameTestAccess::Enemies(S.Game)[0].HP, 190.f, 0.01f));

	S.Destroy();
	return true;
}

// KnockbackStrength=0 のヒットでも KnockbackFramesLeft=7 が設定され、7f後も位置変化なし
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsZeroKnockbackStop,
	"ReinBalance.Survivors.HitFrame.Zero_Knockback_Stop",
	EAutomationTestFlags::ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FSurvivorsZeroKnockbackStop::RunTest(const FString& Parameters)
{
	using namespace SurvivorsGameConstants;
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	// KnockbackStrength=0 のプロジェクタイルで敵をヒット
	S.AddEnemyAt(FVector2D(50.f, 0.f), /*HP=*/200.f);

	FProjectileState Proj;
	Proj.Pos              = FVector2D(50.f, 0.f);
	Proj.Radius           = FSimRadius(10.f);
	Proj.Damage           = FDamage(5.f);
	Proj.bPiercing        = true;
	Proj.WeaponType       = EWeaponType::MagicWand;
	Proj.WeaponSlotIdx    = 1;
	Proj.LifeTime         = FProjectileLifeTime(5.f);
	Proj.KnockbackStrength= 0.f;  // KnockbackStrength=0
	S.Game->GetLogic()->SpawnProjectile(Proj);

	FSurvivorsGameTestAccess::BuildEnemyGrid(S.Game);
	FSurvivorsHitFrame HF;
	FSurvivorsGameTestAccess::ComputeAllWeaponHits(S.Game, HF);
	// KnockbackStrength=0 のイベントを手動生成してApply
	FSurvivorsHitEvent Ev;
	Ev.Type              = ESurvivorsHitType::WeaponAreaDamage;
	Ev.Target.IndexAtBuildTime = 0;
	Ev.Target.UniqueId   = FSurvivorsGameTestAccess::Enemies(S.Game)[0].UniqueId;
	Ev.Damage            = 0.f;
	Ev.WeaponSlot        = 0;
	Ev.KnockbackDir      = FVector2D(1.f, 0.f);
	Ev.KnockbackStrength = 0.f;
	Ev.KnockbackResistance = 0.f;
	HF.Events.Add(Ev);
	FSurvivorsGameTestAccess::ApplyWeaponHits(S.Game, HF);

	// KnockbackFramesLeft == 7 が設定されること（EffKB=0 でも停止フレームは発生）
	TestEqual("KnockbackFramesLeft == 7 even with zero knockback strength",
		FSurvivorsGameTestAccess::Enemies(S.Game)[0].KnockbackFramesLeft, KnockbackFrames);

	const float PosXBefore = FSurvivorsGameTestAccess::Enemies(S.Game)[0].Pos.X;

	// 7 回 UpdateEnemies 後: 位置変化なし（KnockbackVelPerFrame = zero）
	for (int32 i = 0; i < KnockbackFrames; ++i)
		S.RunUpdateEnemies();

	TestTrue("Position unchanged after 7 frames with zero knockback velocity",
		FMath::IsNearlyEqual(FSurvivorsGameTestAccess::Enemies(S.Game)[0].Pos.X, PosXBefore, 0.01f));

	S.Destroy();
	return true;
}

// GlobalFreeze 中はノックバック（7f停止）が発生しないこと
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsGlobalFreezeNoKnockback,
	"ReinBalance.Survivors.HitFrame.GlobalFreeze_No_Knockback",
	EAutomationTestFlags::ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FSurvivorsGlobalFreezeNoKnockback::RunTest(const FString& Parameters)
{
	using namespace SurvivorsGameConstants;
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	S.AddEnemyAt(FVector2D(10.f, 0.f), /*HP=*/200.f);

	// GlobalFreezeUntilTime を ElapsedTime 以降に設定（全体フリーズ中）
	FSurvivorsGameTestAccess::ElapsedTime(S.Game)           = 5.f;
	FSurvivorsGameTestAccess::GlobalFreezeUntilTime(S.Game) = 100.f;

	S.RunWeaponHits();

	// フリーズ中はノックバックが免除される → KnockbackFramesLeft == 0
	TestEqual("KnockbackFramesLeft == 0 during global freeze",
		FSurvivorsGameTestAccess::Enemies(S.Game)[0].KnockbackFramesLeft, 0);

	S.Destroy();
	return true;
}

// 同フレームに2回ヒット → KnockbackVelPerFrame が合算されること
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsMultiHitAccumulation,
	"ReinBalance.Survivors.HitFrame.Multi_Hit_Accumulation",
	EAutomationTestFlags::ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FSurvivorsMultiHitAccumulation::RunTest(const FString& Parameters)
{
	using namespace SurvivorsGameConstants;
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	S.AddEnemyAt(FVector2D(10.f, 0.f), /*HP=*/200.f);
	const int32 EnemyUID = FSurvivorsGameTestAccess::Enemies(S.Game)[0].UniqueId;

	// 2つの WeaponAreaDamage イベントを同フレームに送信（異なる方向）
	FSurvivorsHitFrame HF;
	// ヒット1: X方向
	FSurvivorsHitEvent Ev1;
	Ev1.Type               = ESurvivorsHitType::WeaponAreaDamage;
	Ev1.Target.IndexAtBuildTime = 0;
	Ev1.Target.UniqueId    = EnemyUID;
	Ev1.Damage             = 0.f;
	Ev1.WeaponSlot         = 0;
	Ev1.KnockbackDir       = FVector2D(1.f, 0.f);
	Ev1.KnockbackStrength  = GarlicKnockbackStrength;  // 10u
	Ev1.KnockbackResistance= 0.f;
	HF.Events.Add(Ev1);
	// ヒット2: Y方向（スロット1）
	FSurvivorsHitEvent Ev2;
	Ev2.Type               = ESurvivorsHitType::WeaponAreaDamage;
	Ev2.Target.IndexAtBuildTime = 0;
	Ev2.Target.UniqueId    = EnemyUID;
	Ev2.Damage             = 0.f;
	Ev2.WeaponSlot         = 1;
	Ev2.KnockbackDir       = FVector2D(0.f, 1.f);
	Ev2.KnockbackStrength  = GarlicKnockbackStrength;  // 10u
	Ev2.KnockbackResistance= 0.f;
	HF.Events.Add(Ev2);

	FSurvivorsGameTestAccess::ApplyWeaponHits(S.Game, HF);

	const FEnemyState& E = FSurvivorsGameTestAccess::Enemies(S.Game)[0];
	TestEqual("KnockbackFramesLeft == 7 after multi-hit", E.KnockbackFramesLeft, KnockbackFrames);

	// KnockbackVelPerFrame は X と Y の合算
	const float ExpVelPerFrame = GarlicKnockbackStrength / static_cast<float>(KnockbackFrames);
	TestTrue("KnockbackVelPerFrame.X accumulated",
		FMath::IsNearlyEqual(E.KnockbackVelPerFrame.X, ExpVelPerFrame, 0.01f));
	TestTrue("KnockbackVelPerFrame.Y accumulated",
		FMath::IsNearlyEqual(E.KnockbackVelPerFrame.Y, ExpVelPerFrame, 0.01f));

	// 7 回 UpdateEnemies 後: 合算後の総移動量に到達
	for (int32 i = 0; i < KnockbackFrames; ++i)
		S.RunUpdateEnemies();

	const FVector2D FinalPos = FSurvivorsGameTestAccess::Enemies(S.Game)[0].Pos;
	TestTrue("Final X position after multi-hit accumulation",
		FMath::IsNearlyEqual(FinalPos.X, 10.f + GarlicKnockbackStrength, 0.5f));
	TestTrue("Final Y position after multi-hit accumulation",
		FMath::IsNearlyEqual(FinalPos.Y, 0.f + GarlicKnockbackStrength, 0.5f));

	S.Destroy();
	return true;
}

// ヒット後 3f 経過後に再ヒット → KnockbackFramesLeft が 7 にリセットされること
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsKnockbackRehitReset,
	"ReinBalance.Survivors.HitFrame.Knockback_Rehit_Reset",
	EAutomationTestFlags::ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FSurvivorsKnockbackRehitReset::RunTest(const FString& Parameters)
{
	using namespace SurvivorsGameConstants;
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	S.AddEnemyAt(FVector2D(10.f, 0.f), /*HP=*/200.f);

	// 1回目ヒット
	S.RunWeaponHits();
	TestEqual("KnockbackFramesLeft == 7 after first hit",
		FSurvivorsGameTestAccess::Enemies(S.Game)[0].KnockbackFramesLeft, KnockbackFrames);

	// 3 回 UpdateEnemies（残り 4f）
	for (int32 i = 0; i < 3; ++i)
		S.RunUpdateEnemies();
	TestEqual("KnockbackFramesLeft == 4 after 3 UpdateEnemies",
		FSurvivorsGameTestAccess::Enemies(S.Game)[0].KnockbackFramesLeft, KnockbackFrames - 3);

	// Garlic HitInterval を経過させて再ヒット可能にする
	FSurvivorsGameTestAccess::ElapsedTime(S.Game) += GarlicTable[0].HitInterval + 0.01f;

	// 2回目ヒット → KnockbackFramesLeft が 7 にリセット
	S.RunWeaponHits();
	TestEqual("KnockbackFramesLeft reset to 7 on rehit",
		FSurvivorsGameTestAccess::Enemies(S.Game)[0].KnockbackFramesLeft, KnockbackFrames);

	S.Destroy();
	return true;
}

// GlobalFreeze 中も進行中ノックバックは継続すること（ノックバック優先設計の確認）
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsKnockbackContinuesDuringFreeze,
	"ReinBalance.Survivors.HitFrame.Knockback_Continues_During_Freeze",
	EAutomationTestFlags::ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FSurvivorsKnockbackContinuesDuringFreeze::RunTest(const FString& Parameters)
{
	using namespace SurvivorsGameConstants;
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	// 敵を X+10 位置に置いてヒット → KnockbackFramesLeft=7 を確認
	S.AddEnemyAt(FVector2D(10.f, 0.f), /*HP=*/200.f);
	S.RunWeaponHits();
	TestEqual("KnockbackFramesLeft == 7 after hit",
		FSurvivorsGameTestAccess::Enemies(S.Game)[0].KnockbackFramesLeft, KnockbackFrames);

	// GlobalFreeze を設定（Orologion/Freeze アイテム相当）
	FSurvivorsGameTestAccess::GlobalFreezeUntilTime(S.Game) = 100.f;

	// GlobalFreeze 中に UpdateEnemies を 3 回実行
	const FVector2D PosBeforeFreeze = FSurvivorsGameTestAccess::Enemies(S.Game)[0].Pos;
	for (int32 i = 0; i < 3; ++i)
		S.RunUpdateEnemies();

	// KnockbackFramesLeft が減少していること（フリーズ中もノックバック継続）
	TestEqual("KnockbackFramesLeft decreased during GlobalFreeze (knockback takes priority)",
		FSurvivorsGameTestAccess::Enemies(S.Game)[0].KnockbackFramesLeft, KnockbackFrames - 3);

	// 位置が変化していること（ノックバックが継続された）
	const FVector2D PosAfterFreeze = FSurvivorsGameTestAccess::Enemies(S.Game)[0].Pos;
	TestTrue("Enemy position changed during GlobalFreeze (knockback continued)",
		!FMath::IsNearlyEqual(PosAfterFreeze.X, PosBeforeFreeze.X, 0.01f));

	S.Destroy();
	return true;
}

// Garlic が同 tick で敵Aを倒し、非 piercing 弾も敵Aを狙っていた場合の挙動
// 期待: 弾は死亡済み敵Aに当たって消費され、敵Bはダメージを受けない
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsMixedHitGarlicKillsProjectileConsumed,
	"ReinBalance.Survivors.HitFrame.MixedHit_GarlicKills_ProjectileConsumedOnDeadEnemy",
	EAutomationTestFlags::ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FSurvivorsMixedHitGarlicKillsProjectileConsumed::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	// 敵A: PlayerPos(0,0) — Garlic 1発で死ぬ
	// 敵B: (50,0) — Garlic 範囲外（Garlic range = 25+5 = 30 < 50）
	S.AddEnemyAt(FVector2D(0.f, 0.f), /*HP=*/1.f);
	S.AddEnemyAt(FVector2D(50.f, 0.f), /*HP=*/100.f);

	// 非 piercing 弾を敵A位置(0,0)に配置（敵Bには届かない半径）
	FProjectileState Proj;
	Proj.Pos          = FVector2D(0.f, 0.f);
	Proj.Radius       = FSimRadius(10.f);
	Proj.Damage       = FDamage(20.f);
	Proj.bPiercing    = false;
	Proj.WeaponType   = EWeaponType::MagicWand;
	Proj.WeaponSlotIdx= 1;
	Proj.LifeTime     = FProjectileLifeTime(5.f);
	S.Game->GetLogic()->SpawnProjectile(Proj);

	S.RunWeaponHits();

	// 敵A は Garlic で死亡 pending
	TestTrue("Enemy A is pending remove after Garlic hit",
		FSurvivorsGameTestAccess::Enemies(S.Game)[0].bPendingRemove);

	// 非 piercing 弾は死亡済み敵Aで消費される → 敵Bはダメージなし
	TestEqual("Enemy B HP unchanged (projectile consumed on dead enemy A)",
		FSurvivorsGameTestAccess::Enemies(S.Game)[1].HP, 100.f);

	// プロジェクタイルは消費済み（ApplyWeaponHits 内で削除される）
	TestEqual("Projectile removed after hitting dead enemy A",
		S.Game->GetProjectileCount(), 0);

	S.Destroy();
	return true;
}
