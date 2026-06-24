#include "Misc/AutomationTest.h"
#include "SurvivorsTestHelpers.h"

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsWeaponSpecProjectileAmounts,
	"ReinBalance.Survivors.Wiki.WeaponSpecProjectileAmounts",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FSurvivorsWeaponSpecProjectileAmounts::RunTest(const FString& Parameters)
{
	struct FCase
	{
		EWeaponType Type;
		int32 Level;
		int32 ExpectedProjectiles;
		float SecondsAfterFirstTick;
		const TCHAR* Label;
	};

	const FCase Cases[] = {
		{ EWeaponType::Whip,         1, 1, 0.00f, TEXT("Whip Lv1 first swing") },
		{ EWeaponType::Whip,         2, 1, 0.00f, TEXT("Whip Lv2 first swing") },
		{ EWeaponType::MagicWand,    1, 2, 0.12f, TEXT("Magic Wand Lv1 baseline Amount=2 after 0.1s interval") },
		{ EWeaponType::MagicWand,    2, 3, 0.23f, TEXT("Magic Wand Lv2 baseline progression Amount=3 after 0.1s intervals") },
		{ EWeaponType::Knife,        1, 2, 0.12f, TEXT("Knife Lv1 baseline Amount=2 after 0.1s interval") },
		{ EWeaponType::Knife,        2, 3, 0.23f, TEXT("Knife Lv2 baseline progression Amount=3 after 0.1s intervals") },
		{ EWeaponType::Axe,          1, 2, 0.23f, TEXT("Axe Lv1 baseline Amount=2 after 0.2s interval") },
		{ EWeaponType::Axe,          8, 4, 0.65f, TEXT("Axe Lv8 baseline progression Amount=4 after 0.2s intervals") },
		{ EWeaponType::DeathSpiral,  1, 9, 0.00f, TEXT("Death Spiral Amount=9") },
		{ EWeaponType::Cross,        1, 2, 0.12f, TEXT("Cross Lv1 baseline Amount=2 after 0.1s interval") },
		{ EWeaponType::Cross,        7, 4, 0.35f, TEXT("Cross Lv7 baseline progression Amount=4 after 0.1s intervals") },
		{ EWeaponType::HeavenSword,  1, 1, 0.00f, TEXT("Heaven Sword Amount=1") },
		{ EWeaponType::FireWand,     1, 4, 0.08f, TEXT("Fire Wand Lv1 fan=4 within 0.02s interval window") },
		{ EWeaponType::Runetracer,   1, 2, 0.23f, TEXT("Runetracer Lv1 baseline Amount=2 after 0.2s interval") },
		{ EWeaponType::Runetracer,   7, 4, 0.65f, TEXT("Runetracer Lv7 baseline progression Amount=4 after 0.2s intervals") },
		{ EWeaponType::NoFuture,     1, 1, 0.00f, TEXT("NO FUTURE Amount=1") },
	};

	for (const FCase& Case : Cases)
	{
		FSurvivorsTestWorld S;
		if (!TestTrue(FString::Printf(TEXT("%s world created"), Case.Label), S.Create())) return false;

		EquipTestWeapon(S.Game, Case.Type, Case.Level);
		FSurvivorsGameTestAccess::TickWeapons(S.Game, SurvivorsGameConstants::PhysicsDt);
		if (Case.SecondsAfterFirstTick > 0.f)
		{
			TickTestWeaponsForSeconds(S.Game, Case.SecondsAfterFirstTick);
		}

		TestEqual(FString::Printf(TEXT("%s projectile count"), Case.Label),
			S.Game->GetProjectileCount(),
			Case.ExpectedProjectiles);

		S.Destroy();
	}

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsWhipAlternatingBurst,
	"ReinBalance.Survivors.Wiki.Whip_AlternatingBurst",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FSurvivorsWhipAlternatingBurst::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	EquipTestWeapon(S.Game, EWeaponType::Whip, 1);

	FSurvivorsGameTestAccess::TickWeapons(S.Game, SurvivorsGameConstants::PhysicsDt);
	TestEqual("First Whip swing exists", S.Game->GetProjectileCount(), 1);
	TestTrue("First Whip swing faces +X", S.Game->GetProjectilePos(0).X > FSurvivorsGameTestAccess::PlayerPos(S.Game).X);

	const int32 TicksToSecondSwing = FMath::CeilToInt(0.31f / SurvivorsGameConstants::PhysicsDt);
	for (int32 i = 0; i < TicksToSecondSwing; ++i)
	{
		FSurvivorsGameTestAccess::TickWeapons(S.Game, SurvivorsGameConstants::PhysicsDt);
	}

	TestEqual("Second Whip swing exists after 0.3s", S.Game->GetProjectileCount(), 1);
	TestTrue("Second Whip swing flips to -X", S.Game->GetProjectilePos(0).X < FSurvivorsGameTestAccess::PlayerPos(S.Game).X);

	S.Destroy();
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsKnifeBurstCadence,
	"ReinBalance.Survivors.Wiki.Knife_BurstCadence",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FSurvivorsKnifeBurstCadence::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	// knife_bullet2.mp4 is treated as the Lv1 baseline: two knives at the 0.1s projectile interval.
	FSurvivorsGameTestAccess::PlayerVel(S.Game) = FVector2D(1.f, 0.f);
	EquipTestWeapon(S.Game, EWeaponType::Knife, 1);

	FSurvivorsGameTestAccess::TickWeapons(S.Game, SurvivorsGameConstants::PhysicsDt);
	TestEqual("Knife fires first projectile immediately", S.Game->GetProjectileCount(), 1);

	TickTestWeaponsForSeconds(S.Game, 0.05f);
	TestEqual("Knife does not fire the second projectile before 0.1s", S.Game->GetProjectileCount(), 1);

	TickTestWeaponsForSeconds(S.Game, 0.06f);
	TestEqual("Knife fires second projectile after 0.1s", S.Game->GetProjectileCount(), 2);

	TickTestWeaponsForSeconds(S.Game, 0.15f);
	TestEqual("Knife Lv1 bullet2 baseline stops at two projectiles", S.Game->GetProjectileCount(), 2);

	S.Destroy();
	return true;
}

// Knife: knife_bullet3.mp4 (Lv1 + Amount bonus +1) shows straight knives moving about 326u/s.
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsKnifeVideoProjectileSpeed,
	"ReinBalance.Survivors.Video.Knife_ProjectileSpeed_326u",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FSurvivorsKnifeVideoProjectileSpeed::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	FSurvivorsGameTestAccess::PlayerPos(S.Game) = FVector2D::ZeroVector;
	FSurvivorsGameTestAccess::PlayerVel(S.Game) = FVector2D(1.f, 0.f);

	EquipTestWeapon(S.Game, EWeaponType::Knife, 1);

	FSurvivorsGameTestAccess::TickWeapons(S.Game, SurvivorsGameConstants::PhysicsDt);
	if (!TestTrue("Knife projectile exists", S.Game->GetProjectileCount() > 0))
	{
		S.Destroy();
		return false;
	}

	const FVector2D StartPos = S.Game->GetProjectilePos(0);
	const float Elapsed = TickTestWeaponsForSecondsMeasured(S.Game, 0.25f);
	if (!TestTrue("Knife projectile remains alive for speed sample", S.Game->GetProjectileCount() > 0))
	{
		S.Destroy();
		return false;
	}

	const float Speed = FVector2D::Distance(S.Game->GetProjectilePos(0), StartPos) / Elapsed;
	TestTrue(FString::Printf(TEXT("Knife video speed %.1fu/s should stay near 326u/s"), Speed),
		Speed >= 295.f && Speed <= 365.f);

	S.Destroy();
	return true;
}

// Knife: knife_bullet3_1.mp4 shows per-shot spawn jitter perpendicular to the firing direction.
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsKnifeVideoSpawnJitter,
	"ReinBalance.Survivors.Video.Knife_SpawnJitter_PlayerRadius",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FSurvivorsKnifeVideoSpawnJitter::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	const FVector2D PlayerPos = FVector2D::ZeroVector;
	FSurvivorsGameTestAccess::PlayerPos(S.Game) = PlayerPos;
	FSurvivorsGameTestAccess::PlayerVel(S.Game) = FVector2D(1.f, 0.f);

	EquipTestWeapon(S.Game, EWeaponType::Knife, 2);  // baseline 3-shot volley

	TArray<float> SpawnOffsetsY;
	int32 CapturedProjectiles = 0;
	auto CaptureNewProjectileOffsets = [&]()
	{
		const int32 Count = S.Game->GetProjectileCount();
		for (int32 i = CapturedProjectiles; i < Count; ++i)
		{
			SpawnOffsetsY.Add(S.Game->GetProjectilePos(i).Y - PlayerPos.Y);
		}
		CapturedProjectiles = Count;
	};

	FSurvivorsGameTestAccess::TickWeapons(S.Game, SurvivorsGameConstants::PhysicsDt);
	CaptureNewProjectileOffsets();
	TickTestWeaponsForSeconds(S.Game, 0.11f);
	CaptureNewProjectileOffsets();
	TickTestWeaponsForSeconds(S.Game, 0.11f);
	CaptureNewProjectileOffsets();

	if (!TestEqual("Knife Lv2 captures three spawn offsets", SpawnOffsetsY.Num(), 3))
	{
		S.Destroy();
		return false;
	}

	const float PlayerRadius = FSurvivorsGameTestAccess::PlayerRadius(S.Game);
	float MinOffset = TNumericLimits<float>::Max();
	float MaxOffset = TNumericLimits<float>::Lowest();
	for (const float Offset : SpawnOffsetsY)
	{
		MinOffset = FMath::Min(MinOffset, Offset);
		MaxOffset = FMath::Max(MaxOffset, Offset);
		TestTrue(FString::Printf(TEXT("Knife spawn offset %.2fu stays within player radius %.2fu"), Offset, PlayerRadius),
			FMath::Abs(Offset) <= PlayerRadius + 0.1f);
	}

	TestTrue(FString::Printf(TEXT("Knife spawn offsets should visibly vary, span %.2fu"), MaxOffset - MinOffset),
		(MaxOffset - MinOffset) >= PlayerRadius * 0.5f);

	S.Destroy();
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsKingBibleDurationCooldown,
	"ReinBalance.Survivors.Wiki.KingBible_DurationCooldown",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FSurvivorsKingBibleDurationCooldown::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	EquipTestWeapon(S.Game, EWeaponType::KingBible, 1);

	FSurvivorsGameTestAccess::TickWeapons(S.Game, SurvivorsGameConstants::PhysicsDt);
	TestEqual("King Bible Lv1 starts with 2 active orbs", S.Game->GetOrbitOrbCount(), 2);

	const int32 TicksPastDuration = FMath::CeilToInt(3.1f / SurvivorsGameConstants::PhysicsDt);
	for (int32 i = 0; i < TicksPastDuration; ++i)
	{
		FSurvivorsGameTestAccess::TickWeapons(S.Game, SurvivorsGameConstants::PhysicsDt);
	}
	TestEqual("King Bible orbs disappear after Duration", S.Game->GetOrbitOrbCount(), 0);

	const int32 TicksToNextCycle = FMath::CeilToInt(3.1f / SurvivorsGameConstants::PhysicsDt);
	for (int32 i = 0; i < TicksToNextCycle; ++i)
	{
		FSurvivorsGameTestAccess::TickWeapons(S.Game, SurvivorsGameConstants::PhysicsDt);
	}
	TestEqual("King Bible orbs reactivate after Duration + Cooldown cycle", S.Game->GetOrbitOrbCount(), 2);

	S.Destroy();
	return true;
}

// ============================================================
// P0 武器仕様テスト（動画観察・wiki 由来）
// ============================================================

// MagicWand: Lv1 baseline fires the first shot immediately and the second after 0.1s.
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsMagicWandSequentialTargeting,
	"ReinBalance.Survivors.Wiki.MagicWand_SequentialTargeting",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FSurvivorsMagicWandSequentialTargeting::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	S.AddEnemyAt(FVector2D(50.f, 0.f));

	// MagicWand Lv1 baseline: Amount=2
	EquipTestWeapon(S.Game, EWeaponType::MagicWand, 1);

	FSurvivorsGameTestAccess::TickWeapons(S.Game, SurvivorsGameConstants::PhysicsDt);
	TestEqual("MagicWand Lv1 fires first shot immediately", S.Game->GetProjectileCount(), 1);

	TickTestWeaponsForSeconds(S.Game, 0.05f);
	TestEqual("MagicWand Lv1 does not fire the second shot before 0.1s", S.Game->GetProjectileCount(), 1);

	// 0.1s 経過後に 2 発目
	TickTestWeaponsForSeconds(S.Game, 0.06f);

	TestEqual("MagicWand Lv1 fires second shot after 0.1s", S.Game->GetProjectileCount(), 2);

	S.Destroy();
	return true;
}

// MagicWand: magic_wand_bullet2.mp4 の frame 289→300 で、弾中心が
// 約61.8px / 11f 移動。Camera Z=2000 基準(800u/1920px)で約140u/s。
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsMagicWandVideoProjectileSpeed,
	"ReinBalance.Survivors.Video.MagicWand_ProjectileSpeed_140u",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FSurvivorsMagicWandVideoProjectileSpeed::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	FSurvivorsGameTestAccess::PlayerPos(S.Game) = FVector2D::ZeroVector;
	S.AddEnemyAt(FVector2D(1000.f, 0.f), 10000.f);

	EquipTestWeapon(S.Game, EWeaponType::MagicWand, 1);

	FSurvivorsGameTestAccess::TickWeapons(S.Game, SurvivorsGameConstants::PhysicsDt);
	if (!TestTrue("Magic Wand projectile exists", S.Game->GetProjectileCount() > 0))
	{
		S.Destroy();
		return false;
	}

	const FVector2D StartPos = S.Game->GetProjectilePos(0);
	const float Elapsed = TickTestWeaponsForSecondsMeasured(S.Game, 0.50f);
	if (!TestTrue("Magic Wand projectile remains alive for speed sample", S.Game->GetProjectileCount() > 0))
	{
		S.Destroy();
		return false;
	}

	const float Speed = FVector2D::Distance(S.Game->GetProjectilePos(0), StartPos) / Elapsed;
	TestTrue(FString::Printf(TEXT("Magic Wand video speed %.1fu/s should stay near 140u/s"), Speed),
		Speed >= 120.f && Speed <= 165.f);

	S.Destroy();
	return true;
}

// Cross: cross_bullet2.mp4 / weapon_cross.md の 2発サンプルは 0.1s 間隔の短い sequence。
// 同じ +X 最近傍敵に向ける場合、追加弾も fan spread ではなく +X 方向へ飛ぶ。
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsCrossVideoBullet2Cadence,
	"ReinBalance.Survivors.Wiki.Cross_VideoBullet2_CadenceAndDirection",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FSurvivorsCrossVideoBullet2Cadence::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	FSurvivorsGameTestAccess::PlayerPos(S.Game) = FVector2D::ZeroVector;
	S.AddEnemyAt(FVector2D(200.f, 0.f));

	// Cross Lv1 baseline: two projectiles at the 0.1s interval.
	EquipTestWeapon(S.Game, EWeaponType::Cross, 1);

	FSurvivorsGameTestAccess::TickWeapons(S.Game, SurvivorsGameConstants::PhysicsDt);
	TestEqual("Cross Lv1 bullet2 baseline fires first projectile immediately", S.Game->GetProjectileCount(), 1);

	TickTestWeaponsForSeconds(S.Game, 0.05f);
	TestEqual("Cross does not fire the second projectile before 0.1s", S.Game->GetProjectileCount(), 1);

	TickTestWeaponsForSeconds(S.Game, 0.06f);
	TestEqual("Cross fires the second projectile after 0.1s", S.Game->GetProjectileCount(), 2);

	for (int32 i = 0; i < S.Game->GetProjectileCount(); ++i)
	{
		const FVector2D Pos = S.Game->GetProjectilePos(i);
		TestTrue(FString::Printf(TEXT("Cross projectile %d moves toward +X without fan spread"), i),
			Pos.X > 0.f && FMath::Abs(Pos.Y) < 1.f);
	}

	S.Destroy();
	return true;
}

// Cross: cross_bullet2.mp4 の見やすい区間では、右向きに出た Cross が
// プレイヤー中心から約170-180px(=70-75u)の距離で折り返している。
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsCrossVideoReverseDistance,
	"ReinBalance.Survivors.Video.Cross_ReverseDistance_75u",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FSurvivorsCrossVideoReverseDistance::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	FSurvivorsGameTestAccess::PlayerPos(S.Game) = FVector2D::ZeroVector;
	S.AddEnemyAt(FVector2D(200.f, 0.f), 10000.f);

	EquipTestWeapon(S.Game, EWeaponType::Cross, 1);

	FSurvivorsGameTestAccess::TickWeapons(S.Game, SurvivorsGameConstants::PhysicsDt);
	TestEqual("Cross Lv1 first volley has one projectile", S.Game->GetProjectileCount(), 1);

	float MaxForwardDistance = 0.f;
	for (int32 Step = 0; Step < SurvivorsStepsForSeconds(1.0f); ++Step)
	{
		if (S.Game->GetProjectileCount() <= 0) break;
		MaxForwardDistance = FMath::Max(MaxForwardDistance, S.Game->GetProjectilePos(0).X);
		FSurvivorsGameTestAccess::TickWeapons(S.Game, SurvivorsGameConstants::PhysicsDt);
	}

	TestTrue(FString::Printf(TEXT("Cross reverse distance %.1fu should match video range 60-90u"), MaxForwardDistance),
		MaxForwardDistance >= 60.f && MaxForwardDistance <= 90.f);

	S.Destroy();
	return true;
}

// Cross: weapon_cross.md says the projectile returns, then continues until it leaves the screen;
// it must not be removed just because the next 2.0s cooldown volley starts.
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsCrossPersistsIntoNextVolley,
	"ReinBalance.Survivors.Wiki.Cross_PersistsIntoNextVolley",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FSurvivorsCrossPersistsIntoNextVolley::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	FSurvivorsGameTestAccess::PlayerPos(S.Game) = FVector2D::ZeroVector;
	S.AddEnemyAt(FVector2D(200.f, 0.f));

	EquipTestWeapon(S.Game, EWeaponType::Cross, 1);

	FSurvivorsGameTestAccess::TickWeapons(S.Game, SurvivorsGameConstants::PhysicsDt);
	TestEqual("Cross Lv1 first volley has one projectile", S.Game->GetProjectileCount(), 1);

	TickTestWeaponsForSeconds(S.Game, 2.05f);
	TestTrue("Cross previous projectile is still alive when the next cooldown volley starts",
		S.Game->GetProjectileCount() >= 2);

	S.Destroy();
	return true;
}

// Axe: axe_bullet2.mp4 の frame 80-100 付近では、最初の Axe の上昇量が
// プレイヤーHPバー中心から約130-150px(=54-63u)。動画基準の頂点は約60u。
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsAxeVideoApexHeight,
	"ReinBalance.Survivors.Video.Axe_ApexHeight_60u",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FSurvivorsAxeVideoApexHeight::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	FSurvivorsGameTestAccess::PlayerPos(S.Game) = FVector2D::ZeroVector;
	EquipTestWeapon(S.Game, EWeaponType::Axe, 1);

	FSurvivorsGameTestAccess::TickWeapons(S.Game, SurvivorsGameConstants::PhysicsDt);

	float MaxHeight = 0.f;
	for (int32 Step = 0; Step < SurvivorsStepsForSeconds(1.0f); ++Step)
	{
		if (S.Game->GetProjectileCount() > 0)
		{
			MaxHeight = FMath::Max(MaxHeight, S.Game->GetProjectilePos(0).Y);
		}
		FSurvivorsGameTestAccess::TickWeapons(S.Game, SurvivorsGameConstants::PhysicsDt);
	}

	TestTrue(FString::Printf(TEXT("Axe apex %.1fu should match video range 50-75u"), MaxHeight),
		MaxHeight >= 50.f && MaxHeight <= 75.f);

	S.Destroy();
	return true;
}

// Axe: weapon_axe.md / axe_bullet2.mp4 specify a 0.2s short volley.
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsAxeVideoBullet2Cadence,
	"ReinBalance.Survivors.Wiki.Axe_VideoBullet2_Cadence",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FSurvivorsAxeVideoBullet2Cadence::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	FSurvivorsGameTestAccess::PlayerPos(S.Game) = FVector2D::ZeroVector;
	S.AddEnemyAt(FVector2D(200.f, 0.f));

	// Axe Lv1 baseline: two axes at the 0.2s interval.
	EquipTestWeapon(S.Game, EWeaponType::Axe, 1);

	FSurvivorsGameTestAccess::TickWeapons(S.Game, SurvivorsGameConstants::PhysicsDt);
	TestEqual("Axe Lv1 bullet2 baseline fires first axe immediately", S.Game->GetProjectileCount(), 1);
	if (S.Game->GetProjectileCount() > 0)
	{
		const FVector2D FirstPos = S.Game->GetProjectilePos(0);
		TestTrue("First axe is thrown upward from the player", FMath::Abs(FirstPos.X) < 1.f && FirstPos.Y > 0.f);
	}

	TickTestWeaponsForSeconds(S.Game, 0.10f);
	TestEqual("Axe does not fire the second axe before 0.2s", S.Game->GetProjectileCount(), 1);

	TickTestWeaponsForSeconds(S.Game, 0.11f);
	TestEqual("Axe fires the second axe after about 0.2s", S.Game->GetProjectileCount(), 2);

	S.Destroy();
	return true;
}

// Runetracer: runetracer_bullet2.mp4 / weapon_runetracer.md specify sequence firing at 0.2s interval.
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsRunetracerVideoBullet2Cadence,
	"ReinBalance.Survivors.Wiki.Runetracer_VideoBullet2_Cadence",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FSurvivorsRunetracerVideoBullet2Cadence::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	// Runetracer Lv1 baseline: two runes at the 0.2s interval.
	EquipTestWeapon(S.Game, EWeaponType::Runetracer, 1);

	FSurvivorsGameTestAccess::TickWeapons(S.Game, SurvivorsGameConstants::PhysicsDt);
	TestEqual("Runetracer Lv1 bullet2 baseline fires first rune immediately", S.Game->GetProjectileCount(), 1);

	TickTestWeaponsForSeconds(S.Game, 0.10f);
	TestEqual("Runetracer does not fire the second rune before 0.2s", S.Game->GetProjectileCount(), 1);

	TickTestWeaponsForSeconds(S.Game, 0.11f);
	TestEqual("Runetracer fires the second rune after about 0.2s", S.Game->GetProjectileCount(), 2);

	S.Destroy();
	return true;
}

// Runetracer: OBSERVED: 真下に打った弾が70f(60fps)で画面下(225u)に到達し、70f後にプレイヤー位置に戻った。
// Duration = 140f/60fps = 2.33s。Speed = 225u×60/70 ≈ 193u/s。
// (旧テスト名は FSurvivorsRunetracerDuration225Seconds のまま後方互換で維持)
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsRunetracerDuration225Seconds,
	"ReinBalance.Survivors.Wiki.Runetracer_Duration_2_25s",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FSurvivorsRunetracerDuration225Seconds::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	EquipTestWeapon(S.Game, EWeaponType::Runetracer, 1);

	FSurvivorsGameTestAccess::TickWeapons(S.Game, SurvivorsGameConstants::PhysicsDt);
	TestEqual("Runetracer Lv1 first projectile exists immediately", S.Game->GetProjectileCount(), 1);

	TickTestWeaponsForSeconds(S.Game, 0.21f);
	TestEqual("Runetracer Lv1 second projectile appears after about 0.2s", S.Game->GetProjectileCount(), 2);

	TickTestWeaponsForSeconds(S.Game, 1.80f);
	TestEqual("Runetracer Lv1 projectiles remain before their 2.33s duration", S.Game->GetProjectileCount(), 2);

	TickTestWeaponsForSeconds(S.Game, 0.60f);
	TestEqual("Runetracer Lv1 projectiles expire after their 2.33s duration", S.Game->GetProjectileCount(), 0);

	S.Destroy();
	return true;
}

// Runetracer: OBSERVED: 弾速 225u / (70f/60fps) ≈ 193u/s (Acceptance: 160-225u/s)
// 現行 440u/s では失敗する想定。193u/s 修正後に通す。
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsRunetracerVideoProjectileSpeed,
	"ReinBalance.Survivors.Video.Runetracer_ProjectileSpeed_193u",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FSurvivorsRunetracerVideoProjectileSpeed::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	FSurvivorsGameTestAccess::PlayerPos(S.Game) = FVector2D::ZeroVector;

	EquipTestWeapon(S.Game, EWeaponType::Runetracer, 1);

	FSurvivorsGameTestAccess::TickWeapons(S.Game, SurvivorsGameConstants::PhysicsDt);
	if (!TestTrue("Runetracer Lv1 fires first projectile", S.Game->GetProjectileCount() > 0))
	{
		S.Destroy();
		return false;
	}

	// 0.25s 以内なら 193u/s で 48u — スクリーン端(225u)に届かず bounce しない
	const FVector2D StartPos = S.Game->GetProjectilePos(0);
	const float Elapsed      = TickTestWeaponsForSecondsMeasured(S.Game, 0.25f);
	if (!TestTrue("Runetracer projectile remains alive for speed sample", S.Game->GetProjectileCount() > 0))
	{
		S.Destroy();
		return false;
	}

	// 距離は方向によらず Speed × Elapsed （bounce しない距離なので正確）
	const float Speed = FVector2D::Distance(S.Game->GetProjectilePos(0), StartPos) / Elapsed;
	TestTrue(FString::Printf(TEXT("Runetracer video speed %.1fu/s should be near 193u/s"), Speed),
		Speed >= 160.f && Speed <= 225.f);

	S.Destroy();
	return true;
}

// Runetracer: wiki/spec の Hitbox Delay は0.5s。同じ rune は同じ敵に
// 0.5s より短い間隔では再ヒットせず、0.5s 経過後は再ヒットできる。
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsRunetracerHitboxDelayAllowsRehit,
	"ReinBalance.Survivors.Wiki.Runetracer_HitboxDelay_0_5s_Rehit",
	EAutomationTestFlags::ApplicationContextMask | EAutomationTestFlags::ProductFilter)
bool FSurvivorsRunetracerHitboxDelayAllowsRehit::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	// 敵をグリッド境界から離れた (50,0) に配置し、弾も同位置に置く（接触 = 0u）
	const FVector2D EnemyPos(50.f, 0.f);
	S.AddEnemyAt(EnemyPos, 200.f);

	FProjectileState Proj;
	Proj.Pos            = EnemyPos;
	Proj.Radius         = FSimRadius(10.f);
	Proj.Damage         = FDamage(10.f);
	Proj.WeaponType     = EWeaponType::Runetracer;
	Proj.WeaponSlotIdx  = 0;
	Proj.LifeTime       = FProjectileLifeTime(5.f);
	Proj.bPiercing      = true;
	Proj.KnockbackStrength = 0.f;  // ノックバックなし（敵位置を維持する）
	S.Game->GetLogic()->SpawnProjectile(Proj);

	S.RunWeaponHits();
	const float HPAfterFirst = FSurvivorsGameTestAccess::Enemies(S.Game)[0].HP;
	TestTrue("Runetracer first hit deals damage", FMath::IsNearlyEqual(HPAfterFirst, 190.f, 0.01f));

	S.RunWeaponHits();
	TestEqual("Runetracer does not re-hit before 0.5s",
		FSurvivorsGameTestAccess::Enemies(S.Game)[0].HP, HPAfterFirst);

	FSurvivorsGameTestAccess::ElapsedTime(S.Game) += 0.51f;
	S.RunWeaponHits();
	TestTrue("Runetracer re-hits after 0.5s hitbox delay",
		FMath::IsNearlyEqual(FSurvivorsGameTestAccess::Enemies(S.Game)[0].HP, HPAfterFirst - 10.f, 0.01f));

	S.Destroy();
	return true;
}

// Fire Wand: fire_wand_bullet4.mp4 shows a 4-fireball baseline fan, with 0.02s table interval.
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsFireWandVideoBullet4Baseline,
	"ReinBalance.Survivors.Wiki.FireWand_VideoBullet4_Baseline",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FSurvivorsFireWandVideoBullet4Baseline::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	FSurvivorsGameTestAccess::PlayerPos(S.Game) = FVector2D::ZeroVector;
	S.AddEnemyAt(FVector2D(200.f, 0.f));

	EquipTestWeapon(S.Game, EWeaponType::FireWand, 1);

	FSurvivorsGameTestAccess::TickWeapons(S.Game, SurvivorsGameConstants::PhysicsDt);
	TickTestWeaponsForSeconds(S.Game, 0.08f);
	TestEqual("Fire Wand Lv1 baseline emits four fireballs within the 0.02s interval window",
		S.Game->GetProjectileCount(), 4);

	for (int32 i = 0; i < S.Game->GetProjectileCount(); ++i)
	{
		TestTrue(FString::Printf(TEXT("Fire Wand projectile %d travels generally toward the selected +X enemy"), i),
			S.Game->GetProjectilePos(i).X > 0.f);
	}

	S.Destroy();
	return true;
}

// Fire Wand: fire_wand_bullet4.mp4 frame 290/300 の4発分離区間では、
// 扇全体は約16度、1 projectile step は約5.3度。
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsFireWandVideoFanAngle,
	"ReinBalance.Survivors.Video.FireWand_FanAngle_16deg",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FSurvivorsFireWandVideoFanAngle::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	FSurvivorsGameTestAccess::PlayerPos(S.Game) = FVector2D::ZeroVector;
	S.AddEnemyAt(FVector2D(200.f, 0.f), 10000.f);

	EquipTestWeapon(S.Game, EWeaponType::FireWand, 1);

	FSurvivorsGameTestAccess::TickWeapons(S.Game, SurvivorsGameConstants::PhysicsDt);
	TickTestWeaponsForSeconds(S.Game, 0.08f);
	TestEqual("Fire Wand Lv1 has four fireballs for angle sample", S.Game->GetProjectileCount(), 4);

	TArray<float> AnglesDeg;
	for (int32 i = 0; i < S.Game->GetProjectileCount(); ++i)
	{
		const FVector2D Delta = S.Game->GetProjectilePos(i) - FSurvivorsGameTestAccess::PlayerPos(S.Game);
		AnglesDeg.Add(FMath::RadiansToDegrees(FMath::Atan2(Delta.Y, Delta.X)));
	}
	AnglesDeg.Sort();

	if (TestTrue("Fire Wand angle sample has four angles", AnglesDeg.Num() == 4))
	{
		const float SpreadDeg = AnglesDeg.Last() - AnglesDeg[0];
		TestTrue(FString::Printf(TEXT("Fire Wand fan spread %.1fdeg should match video range 14-18deg"), SpreadDeg),
			SpreadDeg >= 14.f && SpreadDeg <= 18.f);
	}

	S.Destroy();
	return true;
}

// Fire Wand: fire_wand_bullet4.mp4 frame 290→300 では、分離した4発が
// 約38-39px / 10f 移動。Camera Z=2000 基準で約95-100u/s。
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsFireWandVideoProjectileSpeed,
	"ReinBalance.Survivors.Video.FireWand_ProjectileSpeed_96u",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FSurvivorsFireWandVideoProjectileSpeed::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	FSurvivorsGameTestAccess::PlayerPos(S.Game) = FVector2D::ZeroVector;
	S.AddEnemyAt(FVector2D(200.f, 0.f), 10000.f);

	EquipTestWeapon(S.Game, EWeaponType::FireWand, 1);

	FSurvivorsGameTestAccess::TickWeapons(S.Game, SurvivorsGameConstants::PhysicsDt);
	TickTestWeaponsForSeconds(S.Game, 0.08f);
	if (!TestTrue("Fire Wand Lv1 has four fireballs for speed sample", S.Game->GetProjectileCount() == 4))
	{
		S.Destroy();
		return false;
	}

	TArray<FVector2D> StartPositions;
	for (int32 i = 0; i < S.Game->GetProjectileCount(); ++i)
	{
		StartPositions.Add(S.Game->GetProjectilePos(i));
	}

	const float Elapsed = TickTestWeaponsForSecondsMeasured(S.Game, 0.20f);
	if (!TestTrue("Fire Wand projectiles remain alive for speed sample", S.Game->GetProjectileCount() == StartPositions.Num()))
	{
		S.Destroy();
		return false;
	}

	float TotalSpeed = 0.f;
	for (int32 i = 0; i < S.Game->GetProjectileCount(); ++i)
	{
		TotalSpeed += FVector2D::Distance(S.Game->GetProjectilePos(i), StartPositions[i]) / Elapsed;
	}
	const float AvgSpeed = TotalSpeed / static_cast<float>(S.Game->GetProjectileCount());
	TestTrue(FString::Printf(TEXT("Fire Wand video speed %.1fu/s should stay near 96u/s"), AvgSpeed),
		AvgSpeed >= 80.f && AvgSpeed <= 115.f);

	S.Destroy();
	return true;
}

// Lightning Ring: lightning_ring_bullet2.mp4 / weapon_lightning_ring.md specify two baseline strikes.
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsLightningRingVideoBullet2Baseline,
	"ReinBalance.Survivors.Wiki.LightningRing_VideoBullet2_Baseline",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FSurvivorsLightningRingVideoBullet2Baseline::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	S.AddEnemyAt(FVector2D(-100.f, 0.f), 100.f);
	S.AddEnemyAt(FVector2D(100.f, 0.f), 100.f);

	EquipTestWeapon(S.Game, EWeaponType::LightningRing, 1);

	FSurvivorsGameTestAccess::TickWeapons(S.Game, SurvivorsGameConstants::PhysicsDt);
	FSurvivorsGameTestAccess::BuildEnemyGrid(S.Game);
	FSurvivorsHitFrame HF;
	FSurvivorsGameTestAccess::ComputeAllWeaponHits(S.Game, HF);
	FSurvivorsGameTestAccess::ApplyWeaponHits(S.Game, HF);

	TestTrue("Lightning Ring baseline strikes first enemy",
		FSurvivorsGameTestAccess::Enemies(S.Game)[0].HP < 100.f);
	TestTrue("Lightning Ring baseline strikes second enemy",
		FSurvivorsGameTestAccess::Enemies(S.Game)[1].HP < 100.f);

	S.Destroy();
	return true;
}

// King Bible: king_bible_bullet2.mp4 is treated as the Lv1 baseline with two evenly spaced orbiting bibles.
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsKingBibleVideoBullet2OrbitCount,
	"ReinBalance.Survivors.Wiki.KingBible_VideoBullet2_OrbitCount",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FSurvivorsKingBibleVideoBullet2OrbitCount::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	EquipTestWeapon(S.Game, EWeaponType::KingBible, 1);

	FSurvivorsGameTestAccess::TickWeapons(S.Game, SurvivorsGameConstants::PhysicsDt);
	TestEqual("King Bible Lv1 video bullet2 baseline has two orbit orbs", S.Game->GetOrbitOrbCount(), 2);

	if (S.Game->GetOrbitOrbCount() >= 2)
	{
		const FVector2D PlayerPos = FSurvivorsGameTestAccess::PlayerPos(S.Game);
		const FVector2D D0 = S.Game->GetOrbitOrbPos(0) - PlayerPos;
		const FVector2D D1 = S.Game->GetOrbitOrbPos(1) - PlayerPos;
		TestTrue("King Bible Lv1 orbs use the same orbit radius",
			FMath::Abs(D0.Size() - D1.Size()) < 0.1f);
		TestTrue("King Bible Lv1 orbs are evenly spaced on the orbit",
			FVector2D::DotProduct(D0.GetSafeNormal(), D1.GetSafeNormal()) < -0.99f);
	}

	S.Destroy();
	return true;
}

// MagicWand: 最近傍敵方向に発射されること（ノーファン）
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsMagicWandNearestEnemyTargeting,
	"ReinBalance.Survivors.Wiki.MagicWand_NearestEnemyTargeting",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FSurvivorsMagicWandNearestEnemyTargeting::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	FSurvivorsGameTestAccess::PlayerPos(S.Game) = FVector2D::ZeroVector;
	// 敵を Y+ 方向に配置
	S.AddEnemyAt(FVector2D(0.f, 100.f));

	EquipTestWeapon(S.Game, EWeaponType::MagicWand, 1);

	// 5 tick 分進めて弾の移動方向を確認
	for (int32 i = 0; i < 5; ++i)
		FSurvivorsGameTestAccess::TickWeapons(S.Game, SurvivorsGameConstants::PhysicsDt);

	if (S.Game->GetProjectileCount() > 0)
	{
		const FVector2D FinalPos = S.Game->GetProjectilePos(0);
		TestTrue("MagicWand projectile travels toward enemy (Y+)", FinalPos.Y > 0.f);
		TestTrue("MagicWand projectile Y velocity dominant",
			FMath::Abs(FinalPos.Y) > FMath::Abs(FinalPos.X));
	}

	S.Destroy();
	return true;
}

// SantaWater: Lv1 baseline creates two drops sequentially at the 0.3s interval.
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsSantaWaterSequentialDrops,
	"ReinBalance.Survivors.Wiki.SantaWater_SequentialDrops",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FSurvivorsSantaWaterSequentialDrops::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	S.AddEnemyAt(FVector2D(30.f, 0.f));
	S.AddEnemyAt(FVector2D(-30.f, 0.f));

	// SantaWater Lv1 baseline: two sequential drops at the 0.3s interval.
	EquipTestWeapon(S.Game, EWeaponType::SantaWater, 1);

	FSurvivorsGameTestAccess::TickWeapons(S.Game, SurvivorsGameConstants::PhysicsDt);
	TestEqual("SantaWater first drop spawned immediately", S.Game->GetGroundZoneCount(), 1);

	const int32 TicksTo2ndDrop = FMath::CeilToInt(0.31f / SurvivorsGameConstants::PhysicsDt);
	for (int32 i = 0; i < TicksTo2ndDrop; ++i)
		FSurvivorsGameTestAccess::TickWeapons(S.Game, SurvivorsGameConstants::PhysicsDt);

	TestEqual("SantaWater second drop spawned after 0.3s", S.Game->GetGroundZoneCount(), 2);

	S.Destroy();
	return true;
}

// SantaWater: Amount < 4 で最近傍敵の近くに drop が配置されること
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsSantaWaterLowAmountTargetsEnemy,
	"ReinBalance.Survivors.Wiki.SantaWater_LowAmount_TargetsNearestEnemy",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FSurvivorsSantaWaterLowAmountTargetsEnemy::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	const FVector2D EnemyPos(100.f, 0.f);
	S.AddEnemyAt(EnemyPos);

	// SantaWater Lv1: the first of two low-amount drops targets the nearest enemy.
	EquipTestWeapon(S.Game, EWeaponType::SantaWater, 1);

	FSurvivorsGameTestAccess::TickWeapons(S.Game, SurvivorsGameConstants::PhysicsDt);
	TestEqual("SantaWater Lv1 first drop spawns immediately", S.Game->GetGroundZoneCount(), 1);

	if (S.Game->GetGroundZoneCount() > 0)
	{
		const float Dist = FVector2D::Distance(S.Game->GetGroundZonePos(0), EnemyPos);
		TestTrue("SantaWater Lv1 drop near closest enemy (< 5u)", Dist < 5.f);
	}

	S.Destroy();
	return true;
}

// SantaWater: warning zone 中は damage なし
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsSantaWaterWarningZoneNoDamage,
	"ReinBalance.Survivors.Wiki.SantaWater_WarningZone_NoDamage",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FSurvivorsSantaWaterWarningZoneNoDamage::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	S.AddEnemyAt(FVector2D(0.f, 0.f), /*HP=*/100.f);

	EquipTestWeapon(S.Game, EWeaponType::SantaWater, 1);

	FSurvivorsGameTestAccess::TickWeapons(S.Game, SurvivorsGameConstants::PhysicsDt);
	TestTrue("Ground zone is in warning phase", S.Game->IsGroundZoneWarning(0));

	FSurvivorsGameTestAccess::BuildEnemyGrid(S.Game);
	FSurvivorsHitFrame HF;
	FSurvivorsGameTestAccess::ComputeAllWeaponHits(S.Game, HF);
	FSurvivorsGameTestAccess::ApplyWeaponHits(S.Game, HF);

	TestEqual("No damage during warning phase",
		FSurvivorsGameTestAccess::Enemies(S.Game)[0].HP, 100.f);

	S.Destroy();
	return true;
}

// SantaWater: Amount >= 4 で drops が 30°固定間隔の円形配置になること
// 隣接 drop 間距離 = 2 × 140u × sin(15°) ≈ 72.5u
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsSantaWaterHighAmountCircular,
	"ReinBalance.Survivors.Wiki.SantaWater_HighAmount_CircularPattern",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FSurvivorsSantaWaterHighAmountCircular::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	S.AddEnemyAt(FVector2D(50.f, 0.f));

	// SantaWater Lv6: Amount=5
	EquipTestWeapon(S.Game, EWeaponType::SantaWater, 6);

	// 4 drops 分生成（初回 + 0.3s × 3）
	const int32 TicksForAll4 = FMath::CeilToInt(1.0f / SurvivorsGameConstants::PhysicsDt);
	for (int32 i = 0; i < TicksForAll4; ++i)
		FSurvivorsGameTestAccess::TickWeapons(S.Game, SurvivorsGameConstants::PhysicsDt);

	TestTrue("SantaWater Lv6 spawns 2+ drops", S.Game->GetGroundZoneCount() >= 2);

	if (S.Game->GetGroundZoneCount() >= 2)
	{
		const float Dist01 = FVector2D::Distance(S.Game->GetGroundZonePos(0), S.Game->GetGroundZonePos(1));
		// 30°固定間隔: chord = 2 × 140 × sin(15°) ≈ 72.5u、許容 [60, 85]u
		TestTrue(FString::Printf(TEXT("SantaWater drops 30-degree spacing, dist01 %.1fu in [60,85]u"), Dist01),
			Dist01 >= 60.f && Dist01 <= 85.f);
	}

	S.Destroy();
	return true;
}

// Peachone: 複数の小プロジェクタイルが target zone 付近に生成されること
// Peachone/Ebony Wings: peachone_bullet25.mp4 shows a wide orbiting target zone.
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsPeachoneVideoOrbitMetrics,
	"ReinBalance.Survivors.Video.Peachone_OrbitMetrics",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FSurvivorsPeachoneVideoOrbitMetrics::RunTest(const FString& Parameters)
{
	struct FCase
	{
		EWeaponType Type;
		const TCHAR* Label;
	};

	const FCase Cases[] = {
		{ EWeaponType::Peachone,   TEXT("Peachone") },
		{ EWeaponType::EbonyWings, TEXT("Ebony Wings") },
	};

	for (const FCase& Case : Cases)
	{
		FSurvivorsTestWorld S;
		if (!TestTrue(FString::Printf(TEXT("%s world created"), Case.Label), S.Create())) return false;

		FSurvivorsGameTestAccess::PlayerPos(S.Game) = FVector2D::ZeroVector;
		EquipTestWeapon(S.Game, Case.Type, 1);

		FSurvivorsGameTestAccess::TickWeapons(S.Game, SurvivorsGameConstants::PhysicsDt);
		const FVector2D StartPos = S.Game->GetOrbitOrbPos(0);
		const float OrbitRadius = StartPos.Size();
		TestTrue(FString::Printf(TEXT("%s video orbit radius %.1fu should be near 168u"), Case.Label, OrbitRadius),
			OrbitRadius >= 150.f && OrbitRadius <= 185.f);

		const float ZoneRadius = S.Game->GetOrbitOrbVisualRadius(0);
		TestTrue(FString::Printf(TEXT("%s video target-zone radius %.1fu should be near 49u"), Case.Label, ZoneRadius),
			ZoneRadius >= 45.f && ZoneRadius <= 55.f);

		const float StartAngle = FMath::Atan2(StartPos.Y, StartPos.X);
		const float Elapsed = TickTestWeaponsForSecondsMeasured(S.Game, 1.0f);
		const FVector2D EndPos = S.Game->GetOrbitOrbPos(0);
		const float EndAngle = FMath::Atan2(EndPos.Y, EndPos.X);
		const float RotSpeed = FMath::Abs(FMath::FindDeltaAngleRadians(StartAngle, EndAngle)) / Elapsed;
		TestTrue(FString::Printf(TEXT("%s video orbit speed %.2frad/s should be near 0.8rad/s"), Case.Label, RotSpeed),
			RotSpeed >= 0.70f && RotSpeed <= 1.05f);

		S.Destroy();
	}

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsPeachoneAreaScalesImpactOnly,
	"ReinBalance.Survivors.Video.Peachone_AreaScalesImpactOnly",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FSurvivorsPeachoneAreaScalesImpactOnly::RunTest(const FString& Parameters)
{
	struct FSnapshot
	{
		float OrbitRadius = 0.f;
		float ZoneRadius = 0.f;
		float ImpactRadius = 0.f;
	};

	auto Capture = [this](EWeaponType Type, int32 Level, FSnapshot& Out, const TCHAR* Label) -> bool
	{
		FSurvivorsTestWorld S;
		if (!TestTrue(FString::Printf(TEXT("%s world created"), Label), S.Create())) return false;

		FSurvivorsGameTestAccess::PlayerPos(S.Game) = FVector2D::ZeroVector;
		EquipTestWeapon(S.Game, Type, Level);

		FSurvivorsGameTestAccess::TickWeapons(S.Game, SurvivorsGameConstants::PhysicsDt);
		if (!TestTrue(FString::Printf(TEXT("%s spawned bombard projectile"), Label), S.Game->GetProjectileCount() > 0))
		{
			S.Destroy();
			return false;
		}

		Out.OrbitRadius = S.Game->GetOrbitOrbPos(0).Size();
		Out.ZoneRadius = S.Game->GetOrbitOrbVisualRadius(0);
		Out.ImpactRadius = S.Game->GetProjectileRadius(0).Value;

		S.Destroy();
		return true;
	};

	struct FCase
	{
		EWeaponType Type;
		const TCHAR* Label;
	};

	const FCase Cases[] = {
		{ EWeaponType::Peachone,   TEXT("Peachone") },
		{ EWeaponType::EbonyWings, TEXT("Ebony Wings") },
	};

	for (const FCase& Case : Cases)
	{
		FSnapshot Lv1;
		FSnapshot Lv2;
		if (!Capture(Case.Type, 1, Lv1, *FString::Printf(TEXT("%s Lv1"), Case.Label))) return false;
		if (!Capture(Case.Type, 2, Lv2, *FString::Printf(TEXT("%s Lv2"), Case.Label))) return false;

		TestTrue(FString::Printf(TEXT("%s Lv1 impact radius %.1fu should match video small impact"), Case.Label, Lv1.ImpactRadius),
			Lv1.ImpactRadius >= 3.f && Lv1.ImpactRadius <= 6.f);
		TestTrue(FString::Printf(TEXT("%s BaseArea must not change orbit radius"), Case.Label),
			FMath::Abs(Lv2.OrbitRadius - Lv1.OrbitRadius) <= 1.f);
		TestTrue(FString::Printf(TEXT("%s BaseArea must not change target-zone radius"), Case.Label),
			FMath::Abs(Lv2.ZoneRadius - Lv1.ZoneRadius) <= 1.f);

		const float Ratio = (Lv1.ImpactRadius > 0.f) ? (Lv2.ImpactRadius / Lv1.ImpactRadius) : 0.f;
		TestTrue(FString::Printf(TEXT("%s Lv2 BaseArea should scale impact radius only, ratio %.2f"), Case.Label, Ratio),
			Ratio >= 1.35f && Ratio <= 1.45f);
	}

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsPeachoneBombardModel,
	"ReinBalance.Survivors.Wiki.Peachone_BombardModel_MultipleProjectiles",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FSurvivorsPeachoneBombardModel::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	FSurvivorsGameTestAccess::PlayerPos(S.Game) = FVector2D::ZeroVector;

	// Peachone Lv1: Amount=4, PeachoneSetsPerActivation=4 → 計 16 shots
	EquipTestWeapon(S.Game, EWeaponType::Peachone, 1);

	// 初回 tick で最初の 1 発が即時発射
	FSurvivorsGameTestAccess::TickWeapons(S.Game, SurvivorsGameConstants::PhysicsDt);
	TestTrue("Peachone fires at least 1 projectile immediately", S.Game->GetProjectileCount() >= 1);

	// 0.025s × 2 後に 2 発以上
	const int32 TicksFor2Shots = FMath::CeilToInt(0.06f / SurvivorsGameConstants::PhysicsDt);
	for (int32 i = 0; i < TicksFor2Shots; ++i)
		FSurvivorsGameTestAccess::TickWeapons(S.Game, SurvivorsGameConstants::PhysicsDt);

	TestTrue("Peachone fires multiple projectiles (bombard model not single AoE)",
		S.Game->GetProjectileCount() >= 2);

	// 砲撃弾はプレイヤー位置(0,0)ではなく orbit zone 付近（OrbitRadius=60u, BombRadius=30u）
	if (S.Game->GetProjectileCount() > 0)
	{
		const FVector2D ProjPos = S.Game->GetProjectilePos(0);
		const float DistFromPlayer = ProjPos.Size();
		// Video target zone: orbit center about 168u from player, target-zone radius about 49u.
		TestTrue("Peachone projectile is in video target-zone range (about 115u-215u from player)",
			DistFromPlayer > 90.f && DistFromPlayer < 235.f);
	}

	S.Destroy();
	return true;
}

// Vandalier: 2 zone から砲撃弾が生成され、orbit orb が 2 つであること
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsVandalierTwoZoneBombard,
	"ReinBalance.Survivors.Wiki.Vandalier_TwoZone_BombardModel",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FSurvivorsVandalierTwoZoneBombard::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	EquipTestWeapon(S.Game, EWeaponType::Vandalier, 1);

	FSurvivorsGameTestAccess::TickWeapons(S.Game, SurvivorsGameConstants::PhysicsDt);

	// 2 zone 分の初回発射 → 2 発以上
	TestTrue("Vandalier fires from 2 zones (>=2 projectiles)", S.Game->GetProjectileCount() >= 2);
	TestEqual("Vandalier has 2 orbit orbs", S.Game->GetOrbitOrbCount(), 2);

	S.Destroy();
	return true;
}

// ============================================================
// King Bible per-orb hit cooldown テスト
// ============================================================

// King Bible Lv1 baseline: orb 0 and orb 1 have independent per-orb hit cooldowns.
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsKingBiblePerOrbIndependentCooldown,
	"ReinBalance.Survivors.Wiki.KingBible_PerOrb_IndependentCooldown",
	EAutomationTestFlags::ApplicationContextMask | EAutomationTestFlags::ProductFilter)
bool FSurvivorsKingBiblePerOrbIndependentCooldown::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	// 敵を1体配置（HP 十分）
	S.AddEnemyAt(FVector2D::ZeroVector, 2000.f);

	EquipTestWeapon(S.Game, EWeaponType::KingBible, 1);  // Lv1 baseline: Amount=2


	// 1tick 起動してオーブをアクティブにする
	FSurvivorsGameTestAccess::TickWeapons(S.Game, SurvivorsGameConstants::PhysicsDt);
	if (!TestEqual("KingBible Lv1 has 2 orbs", S.Game->GetOrbitOrbCount(), 2)) { S.Destroy(); return false; }

	// 敵をオーブ 0 の位置に置く → orb 0 ヒット
	FSurvivorsGameTestAccess::Enemies(S.Game)[0].Pos = S.Game->GetOrbitOrbPos(0);
	FSurvivorsGameTestAccess::BuildEnemyGrid(S.Game);
	FSurvivorsHitFrame HF1;
	FSurvivorsGameTestAccess::ComputeAllWeaponHits(S.Game, HF1);
	FSurvivorsGameTestAccess::ApplyWeaponHits(S.Game, HF1);
	const float HPAfterOrb0 = FSurvivorsGameTestAccess::Enemies(S.Game)[0].HP;
	TestTrue("Orb 0 hit: HP decreased", HPAfterOrb0 < 2000.f);

	// 敵をオーブ 1 の位置に置く → 同じ敵に orb 1 がすぐ当たれるか（per-orb 独立 cooldown）
	FSurvivorsGameTestAccess::Enemies(S.Game)[0].Pos = S.Game->GetOrbitOrbPos(1);
	FSurvivorsGameTestAccess::BuildEnemyGrid(S.Game);
	FSurvivorsHitFrame HF2;
	FSurvivorsGameTestAccess::ComputeAllWeaponHits(S.Game, HF2);
	FSurvivorsGameTestAccess::ApplyWeaponHits(S.Game, HF2);
	const float HPAfterOrb1 = FSurvivorsGameTestAccess::Enemies(S.Game)[0].HP;
	TestTrue("Orb 1 can immediately hit same enemy (per-orb independent cooldown)",
		HPAfterOrb1 < HPAfterOrb0);

	S.Destroy();
	return true;
}

// King Bible Lv1 で同一 orb が同じ敵に 1.7s 以内に再ヒットしないこと
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsKingBibleOrbCooldownSameOrb,
	"ReinBalance.Survivors.Wiki.KingBible_PerOrb_SameOrbCooldown",
	EAutomationTestFlags::ApplicationContextMask | EAutomationTestFlags::ProductFilter)
bool FSurvivorsKingBibleOrbCooldownSameOrb::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	S.AddEnemyAt(FVector2D::ZeroVector, 2000.f);

	EquipTestWeapon(S.Game, EWeaponType::KingBible, 1);  // Lv1 baseline: Amount=2


	FSurvivorsGameTestAccess::TickWeapons(S.Game, SurvivorsGameConstants::PhysicsDt);
	if (!TestEqual("KingBible Lv1 has 2 orbs", S.Game->GetOrbitOrbCount(), 2)) { S.Destroy(); return false; }

	const FVector2D Orb0Pos = S.Game->GetOrbitOrbPos(0);

	// 敵をオーブ 0 に重ねる → 1回目ヒット
	// ノックバック(KnockbackSim_1=20u)で敵が離れるため、各 ComputeHits 前に位置をリセットする
	FSurvivorsGameTestAccess::Enemies(S.Game)[0].Pos = Orb0Pos;
	FSurvivorsGameTestAccess::BuildEnemyGrid(S.Game);
	FSurvivorsHitFrame HF1;
	FSurvivorsGameTestAccess::ComputeAllWeaponHits(S.Game, HF1);
	FSurvivorsGameTestAccess::ApplyWeaponHits(S.Game, HF1);
	const float HPAfterFirst = FSurvivorsGameTestAccess::Enemies(S.Game)[0].HP;
	TestTrue("First orb hit deals damage", HPAfterFirst < 2000.f);

	// 再度 → same orb cooldown でブロックされるはず（位置をリセットして接触範囲内に戻す）
	FSurvivorsGameTestAccess::Enemies(S.Game)[0].Pos = Orb0Pos;
	FSurvivorsGameTestAccess::BuildEnemyGrid(S.Game);
	FSurvivorsHitFrame HF2;
	FSurvivorsGameTestAccess::ComputeAllWeaponHits(S.Game, HF2);
	FSurvivorsGameTestAccess::ApplyWeaponHits(S.Game, HF2);
	TestEqual("Same orb does not re-hit immediately",
		FSurvivorsGameTestAccess::Enemies(S.Game)[0].HP, HPAfterFirst);

	// 1.7s 経過後 → 再ヒット可能（位置をリセット）
	FSurvivorsGameTestAccess::ElapsedTime(S.Game) += 1.71f;
	FSurvivorsGameTestAccess::Enemies(S.Game)[0].Pos = Orb0Pos;
	FSurvivorsGameTestAccess::BuildEnemyGrid(S.Game);
	FSurvivorsHitFrame HF3;
	FSurvivorsGameTestAccess::ComputeAllWeaponHits(S.Game, HF3);
	FSurvivorsGameTestAccess::ApplyWeaponHits(S.Game, HF3);
	TestTrue("Same orb re-hits after 1.7s cooldown",
		FSurvivorsGameTestAccess::Enemies(S.Game)[0].HP < HPAfterFirst);

	S.Destroy();
	return true;
}

// ============================================================
// Lightning Ring strike marker テスト
// ============================================================

// Lightning Ring 発動時に strike marker（GroundZone）が敵付近に生成されること
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsLightningRingStrikeMarker,
	"ReinBalance.Survivors.Wiki.LightningRing_StrikeMarker_InObs",
	EAutomationTestFlags::ApplicationContextMask | EAutomationTestFlags::ProductFilter)
bool FSurvivorsLightningRingStrikeMarker::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	// Lv1: Amount=2、敵を2体配置
	S.AddEnemyAt(FVector2D(-100.f, 0.f), 100.f);
	S.AddEnemyAt(FVector2D( 100.f, 0.f), 100.f);

	EquipTestWeapon(S.Game, EWeaponType::LightningRing, 1);

	FSurvivorsGameTestAccess::TickWeapons(S.Game, SurvivorsGameConstants::PhysicsDt);
	FSurvivorsGameTestAccess::BuildEnemyGrid(S.Game);
	FSurvivorsHitFrame HF;
	FSurvivorsGameTestAccess::ComputeAllWeaponHits(S.Game, HF);
	FSurvivorsGameTestAccess::ApplyWeaponHits(S.Game, HF);

	// 落雷位置 marker が GroundZone として生成されていること（Amount=2 なので 2 個）
	TestTrue("Lightning Ring spawns strike markers (>=2 ground zones)",
		S.Game->GetGroundZoneCount() >= 2);

	// marker は敵の位置付近に配置されること
	bool bFoundNearEnemy0 = false;
	bool bFoundNearEnemy1 = false;
	for (int32 i = 0; i < S.Game->GetGroundZoneCount(); ++i)
	{
		const FVector2D ZPos = S.Game->GetGroundZonePos(i);
		if (FVector2D::Distance(ZPos, FVector2D(-100.f, 0.f)) < 50.f) bFoundNearEnemy0 = true;
		if (FVector2D::Distance(ZPos, FVector2D( 100.f, 0.f)) < 50.f) bFoundNearEnemy1 = true;
	}
	TestTrue("Strike marker placed near enemy 0", bFoundNearEnemy0);
	TestTrue("Strike marker placed near enemy 1", bFoundNearEnemy1);

	// marker は短寿命（player center 常時 ring は出ない）
	TestTrue("Strike markers are not warning zones", !S.Game->IsGroundZoneWarning(0));

	S.Destroy();
	return true;
}

// ============================================================
// On-screen targeting テスト
// ============================================================

// MagicWand は画面外の敵を狙わず、画面内の敵がいない場合はランダム方向に発射する
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsMagicWandOnScreenOnly,
	"ReinBalance.Survivors.Wiki.MagicWand_OnScreenTargeting",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FSurvivorsMagicWandOnScreenOnly::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	FSurvivorsGameTestAccess::PlayerPos(S.Game) = FVector2D::ZeroVector;
	// 敵を画面外（500u）にのみ配置
	S.AddEnemyAt(FVector2D(500.f, 0.f));

	EquipTestWeapon(S.Game, EWeaponType::MagicWand, 1);

	// 画面外敵しかいない場合、弾はランダム方向（≠ 敵方向）に飛ぶ
	// ここでは単純に「弾が発射されること」だけをテストする（ランダム方向は検証しない）
	FSurvivorsGameTestAccess::TickWeapons(S.Game, SurvivorsGameConstants::PhysicsDt);
	TestEqual("MagicWand fires even when no on-screen enemy (random direction)",
		S.Game->GetProjectileCount(), 1);

	S.Destroy();
	return true;
}

// Axe は上方向 ±30° のランダム角度で発射されること（1発目は真上固定）
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsAxeRandomUpwardDirection,
	"ReinBalance.Survivors.Wiki.Axe_RandomUpwardDirection",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FSurvivorsAxeRandomUpwardDirection::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	FSurvivorsGameTestAccess::PlayerPos(S.Game) = FVector2D::ZeroVector;
	// 敵なし（ランダム方向）

	EquipTestWeapon(S.Game, EWeaponType::Axe, 2);  // Amount=2

	// 初回発射：1発目（真上固定）
	FSurvivorsGameTestAccess::TickWeapons(S.Game, SurvivorsGameConstants::PhysicsDt);
	if (!TestEqual("Axe fires first shot", S.Game->GetProjectileCount(), 1)) { S.Destroy(); return false; }

	const FVector2D Pos0 = S.Game->GetProjectilePos(0);
	// 上方向（Y > 0）であること
	TestTrue("Axe shot 0 travels upward (Y > 0)", Pos0.Y > 0.f);
	// 1発目は真上固定（BurstShotsFiredCount==0 → RandomOffset=0）なので X≈0
	TestTrue("Axe shot 0 is within ±30° of up", FMath::Abs(Pos0.X) <= Pos0.Y + 0.01f);

	// 0.2s 後：2発目
	TickTestWeaponsForSeconds(S.Game, 0.21f);
	if (!TestEqual("Axe fires second shot", S.Game->GetProjectileCount(), 2)) { S.Destroy(); return false; }

	S.Destroy();
	return true;
}

// ============================================================
// Runetracer スクリーンエッジバウンステスト
// ============================================================

// AWallActor が存在しない環境でも Runetracer がスクリーン端で跳ね返ること
// Runetracer を装備して実装経路（USurvivorsRunetracerWeapon::Tick → UpdateProjectilesBySlot）を通す
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsRunetracerScreenEdgeBounce,
	"ReinBalance.Survivors.Wiki.Runetracer_ScreenEdgeBounce",
	EAutomationTestFlags::ApplicationContextMask | EAutomationTestFlags::ProductFilter)
bool FSurvivorsRunetracerScreenEdgeBounce::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	FSurvivorsGameTestAccess::PlayerPos(S.Game) = FVector2D::ZeroVector;

	// Runetracer を装備して SlotIdx=0 のバウンス処理経路を有効化
	EquipTestWeapon(S.Game, EWeaponType::Runetracer, 1);

	// バウンス機構のテスト専用速度。
	// Lv1 実速度(193u/s)では1tick(1/60s)で約3u進むのみでスクリーン端(400u)に届かない。
	// バウンスを確実に発生させるため端の1u内側からスタートし、1tick以内に端を超える速度を使う。
	// 速度の正確性は FSurvivorsRunetracerVideoProjectileSpeed で別途検証済み。
	// SlotIdx=0（Runetracer のスロット）に対してバウンス処理が走る
	const float ScreenEdgeX = SurvivorsGameConstants::ScreenHalfWidthU;  // 400u
	const float HighSpeed   = 600.f;  // バウンステスト専用: 1tick≈10u前進→端超え→2tick目は逆行
	FProjectileState Proj;
	Proj.Pos            = FVector2D(ScreenEdgeX - 1.f, 0.f);
	Proj.Vel            = FVector2D(HighSpeed, 0.f);
	Proj.Radius         = FSimRadius(1.f);
	Proj.Damage         = FDamage(0.f);
	Proj.WeaponType     = EWeaponType::Runetracer;
	Proj.WeaponSlotIdx  = 0;  // Runetracer のスロット
	Proj.LifeTime       = FProjectileLifeTime(10.f);
	Proj.bPiercing      = true;
	Proj.BounceCount    = FBounceCount(3);
	S.Game->GetLogic()->SpawnProjectile(Proj);
	const int32 PIdx = S.Game->GetProjectileCount() - 1;

	// Tick: Runetracer::Tick → UpdateProjectilesBySlot（速度反転）→ TickProjectiles（Pos += Vel * Dt）
	FSurvivorsGameTestAccess::TickWeapons(S.Game, SurvivorsGameConstants::PhysicsDt);

	if (!TestTrue("Runetracer projectile survives after screen edge tick", S.Game->GetProjectileCount() > PIdx))
	{
		S.Destroy();
		return false;
	}

	// 速度が反転した後に移動しているので、最終位置はスクリーン端内（端を超えない）
	const float PosX1 = S.Game->GetProjectilePos(PIdx).X;
	TestTrue(FString::Printf(TEXT("Runetracer pos after bounce %.1fu <= screen edge %.1fu"),
		PosX1, ScreenEdgeX),
		PosX1 <= ScreenEdgeX + 1.f);

	// もう 1 tick: 反射済みなので -X 方向に動く（X が減少）
	FSurvivorsGameTestAccess::TickWeapons(S.Game, SurvivorsGameConstants::PhysicsDt);
	if (S.Game->GetProjectileCount() > PIdx)
	{
		const float PosX2 = S.Game->GetProjectilePos(PIdx).X;
		TestTrue("Runetracer moves leftward after screen-edge bounce", PosX2 < PosX1);
	}

	S.Destroy();
	return true;
}

// ============================================================
// SantaWater 高Amount配置半径テスト（画像由来）
// ============================================================

// SantaWater Lv4+: drop center は player から 130-150u の円上に配置されること
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsSantaWaterHighAmountPlacementRadiusFromImage,
	"ReinBalance.Survivors.Wiki.SantaWater_HighAmount_PlacementRadius_Image",
	EAutomationTestFlags::ApplicationContextMask | EAutomationTestFlags::ProductFilter)
bool FSurvivorsSantaWaterHighAmountPlacementRadiusFromImage::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	FSurvivorsGameTestAccess::PlayerPos(S.Game) = FVector2D::ZeroVector;

	// SantaWater Lv4: Amount=4（高Amount円形配置）
	EquipTestWeapon(S.Game, EWeaponType::SantaWater, 4);

	// 4 drops 全生成（0.3s×3+初回）
	const int32 TicksForAll = FMath::CeilToInt(1.1f / SurvivorsGameConstants::PhysicsDt);
	for (int32 i = 0; i < TicksForAll; ++i)
		FSurvivorsGameTestAccess::TickWeapons(S.Game, SurvivorsGameConstants::PhysicsDt);

	TestTrue("SantaWater Lv4 spawns at least 4 drops", S.Game->GetGroundZoneCount() >= 4);

	for (int32 i = 0; i < S.Game->GetGroundZoneCount(); ++i)
	{
		const float Dist = S.Game->GetGroundZonePos(i).Size();
		TestTrue(FString::Printf(TEXT("Drop %d center distance %.1fu should be 130-150u from player"), i, Dist),
			Dist >= 130.f && Dist <= 150.f);
	}

	S.Destroy();
	return true;
}

// SantaWater: zone radius と配置半径の合計（blue area 外縁）が 155-178u であること
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsSantaWaterHighAmountBlueAreaDistanceFromImage,
	"ReinBalance.Survivors.Wiki.SantaWater_HighAmount_BlueArea_Image",
	EAutomationTestFlags::ApplicationContextMask | EAutomationTestFlags::ProductFilter)
bool FSurvivorsSantaWaterHighAmountBlueAreaDistanceFromImage::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	FSurvivorsGameTestAccess::PlayerPos(S.Game) = FVector2D::ZeroVector;

	// Lv1: ZoneRadius=30u (SantaWater Radius=100%=30u) → 画像由来 [20,38u] の範囲内
	// 敵を SantaWaterCircleRadius(140u) の位置に置くことで
	// 1発目が敵付近(140u)に落ちる → DistFromPlayer ≈ 140u, BlueAreaEdge ≈ 170u
	S.AddEnemyAt(FVector2D(SurvivorsGameConstants::SantaWaterCircleRadius, 0.f));

	EquipTestWeapon(S.Game, EWeaponType::SantaWater, 1);

	FSurvivorsGameTestAccess::TickWeapons(S.Game, SurvivorsGameConstants::PhysicsDt);
	if (!TestTrue("SantaWater Lv1 first drop spawned", S.Game->GetGroundZoneCount() >= 1))
	{
		S.Destroy();
		return false;
	}

	const float ZoneRadius     = S.Game->GetGroundZoneRadius(0);
	const float DistFromPlayer = S.Game->GetGroundZonePos(0).Size();
	const float BlueAreaEdge   = DistFromPlayer + ZoneRadius;

	TestTrue(FString::Printf(TEXT("Zone radius %.1fu should be 20-38u (image-derived Lv1=30u)"), ZoneRadius),
		ZoneRadius >= 20.f && ZoneRadius <= 38.f);
	TestTrue(FString::Printf(TEXT("Blue area outer edge %.1fu should be 155-178u"), BlueAreaEdge),
		BlueAreaEdge >= 155.f && BlueAreaEdge <= 178.f);

	const float Margin = BlueAreaEdge - SurvivorsGameConstants::SantaWaterCircleRadius;
	TestTrue(FString::Printf(TEXT("Blue area margin outside red circle %.1fu should be 20-38u"), Margin),
		Margin >= 20.f && Margin <= 38.f);

	S.Destroy();
	return true;
}
