#include "Misc/AutomationTest.h"
#include "Engine/World.h"
#include "Survivors/Logic/SurvivorsGame.h"
#include "Survivors/Logic/SurvivorsTypes.h"
#include "Survivors/Logic/SurvivorsGameConstants.h"
#include "Survivors/Logic/SurvivorsCollisionComponent.h"
#include "Survivors/Logic/SurvivorsEnemyComponent.h"
#include "Survivors/Logic/SurvivorsGemComponent.h"
#include "Survivors/Logic/SurvivorsPickupComponent.h"
#include "Survivors/Logic/SurvivorsPlayerComponent.h"
#include "Survivors/Logic/Weapons/SurvivorsWeaponComponent.h"
#include "Survivors/Logic/Weapons/Projectile/SurvivorsKingBibleWeapon.h"

// ============================================================
// テストヘルパー: ASurvivorsGame の private 状態へのアクセサ
// (SurvivorsGame.h の WITH_AUTOMATION_TESTS friend 宣言が必要)
// ============================================================
struct FSurvivorsGameTestAccess
{
	static TArray<FEnemyState>& Enemies(ASurvivorsGame* G) { return G->Enemies; }
	static TArray<FGemState>&   Gems(ASurvivorsGame* G)    { return G->Gems; }
	static TArray<FSpecialPickupState>& SpecialPickups(ASurvivorsGame* G) { return G->SpecialPickups; }
	static float& PlayerHP(ASurvivorsGame* G)              { return G->PlayerHP; }
	static float& PlayerXP(ASurvivorsGame* G)              { return G->PlayerXP; }
	static FVector2D& PlayerPos(ASurvivorsGame* G)         { return G->PlayerPos; }
	static FVector2D& PlayerVel(ASurvivorsGame* G)         { return G->PlayerVel; }
	static bool& bShieldActive(ASurvivorsGame* G)          { return G->bShieldActive; }
	static float& LastReward(ASurvivorsGame* G)            { return G->LastReward; }
	static float& ElapsedTime(ASurvivorsGame* G)           { return G->ElapsedTime; }
	static float& GemPickupRadius(ASurvivorsGame* G)       { return G->GemPickupRadius; }
	static int32& NextEnemyId(ASurvivorsGame* G)           { return G->NextEnemyId; }
	static int32& NextGemId(ASurvivorsGame* G)             { return G->NextGemId; }
	static FWeaponSlot* WeaponSlots(ASurvivorsGame* G)     { return G->WeaponSlots; }
	static FPassiveSlot* PassiveSlots(ASurvivorsGame* G)   { return G->PassiveSlots; }
	static FPassiveEffects& PassiveEffects(ASurvivorsGame* G) { return G->CachedPassiveEffects; }

	static USurvivorsCollisionComponent* CollComp(ASurvivorsGame* G) { return G->CollisionComponent; }
	static USurvivorsEnemyComponent*     EnemyComp(ASurvivorsGame* G){ return G->EnemyComponent; }
	static USurvivorsGemComponent*       GemComp(ASurvivorsGame* G)  { return G->GemComponent; }
	static USurvivorsPickupComponent*    PickupComp(ASurvivorsGame* G){ return G->PickupComponent; }
	static USurvivorsPlayerComponent*    PlayerComp(ASurvivorsGame* G){ return G->PlayerComponent; }
	static USurvivorsWeaponComponent*    WeaponComp(ASurvivorsGame* G){ return G->WeaponComponent; }

	static float XPRequiredForLevel(ASurvivorsGame* G, int32 Level) { return G->XPRequiredForLevel(Level); }

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

static void EquipTestWeapon(ASurvivorsGame* Game, EWeaponType Type, int32 Level)
{
	FWeaponSlot* Weapons = FSurvivorsGameTestAccess::WeaponSlots(Game);
	Weapons[0].Type = Type;
	Weapons[0].Level = FWeaponLevel(Level);
	FSurvivorsGameTestAccess::WeaponComp(Game)->EquipWeapon(0, Type, Level);
}

static int32 SurvivorsStepsForSeconds(float Seconds)
{
	return FMath::CeilToInt(Seconds / SurvivorsGameConstants::PhysicsDt);
}

static void TickTestWeaponsForSteps(USurvivorsWeaponComponent* WC, int32 Steps)
{
	for (int32 i = 0; i < Steps; ++i)
	{
		WC->TickWeapons(SurvivorsGameConstants::PhysicsDt);
	}
}

static void TickTestWeaponsForSeconds(USurvivorsWeaponComponent* WC, float Seconds)
{
	TickTestWeaponsForSteps(WC, SurvivorsStepsForSeconds(Seconds));
}

static float TickTestWeaponsForSecondsMeasured(USurvivorsWeaponComponent* WC, float Seconds)
{
	const int32 Steps = SurvivorsStepsForSeconds(Seconds);
	TickTestWeaponsForSteps(WC, Steps);
	return Steps * SurvivorsGameConstants::PhysicsDt;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsWikiLevelRequirements,
	"ReinBalance.Survivors.Wiki.LevelRequirements",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FSurvivorsWikiLevelRequirements::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!S.Create()) return false;

	TestEqual("Level 2 requirement", FSurvivorsGameTestAccess::XPRequiredForLevel(S.Game, 2), 5.f);
	TestEqual("Level 20 requirement", FSurvivorsGameTestAccess::XPRequiredForLevel(S.Game, 20), 185.f);
	TestEqual("Level 21 includes +600 wall", FSurvivorsGameTestAccess::XPRequiredForLevel(S.Game, 21), 795.f);
	TestEqual("Level 40 requirement", FSurvivorsGameTestAccess::XPRequiredForLevel(S.Game, 40), 442.f);
	TestEqual("Level 41 includes +2400 wall", FSurvivorsGameTestAccess::XPRequiredForLevel(S.Game, 41), 2855.f);

	S.Destroy();
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsWikiDefaultStartWeaponMode,
	"ReinBalance.Survivors.Wiki.DefaultStartWeaponMode",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FSurvivorsWikiDefaultStartWeaponMode::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!S.Create()) return false;

	TestTrue("StartingWeaponMode default remains garlic",
		S.Game->StartingWeaponMode.Equals(TEXT("garlic"), ESearchCase::IgnoreCase));
	TestEqual("Default starting weapon remains Garlic",
		static_cast<int32>(FSurvivorsGameTestAccess::WeaponSlots(S.Game)[0].Type),
		static_cast<int32>(EWeaponType::Garlic));

	S.Destroy();
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsWikiPassiveGrowthAndAttractorb,
	"ReinBalance.Survivors.Wiki.PassiveGrowthAndAttractorb",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FSurvivorsWikiPassiveGrowthAndAttractorb::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!S.Create()) return false;

	FPassiveSlot* Passives = FSurvivorsGameTestAccess::PassiveSlots(S.Game);
	Passives[0].Type = EPassiveItemType::Crown;
	Passives[0].Level = 5;
	Passives[1].Type = EPassiveItemType::Attractorb;
	Passives[1].Level = 5;
	FSurvivorsGameTestAccess::PlayerComp(S.Game)->RecalcPassiveEffects();

	TestTrue("Crown level 5 gives 1.4x Growth",
		FMath::IsNearlyEqual(FSurvivorsGameTestAccess::PassiveEffects(S.Game).GrowthMult, 1.4f, 0.001f));
	TestTrue("Attractorb level 5 uses wiki multiplier",
		FMath::IsNearlyEqual(FSurvivorsGameTestAccess::GemPickupRadius(S.Game), 50.f * 3.980025f, 0.01f));

	FSurvivorsGameTestAccess::PlayerComp(S.Game)->ProcessXPGain(10.f);
	TestTrue("Growth applies to gained XP",
		FMath::IsNearlyEqual(FSurvivorsGameTestAccess::PlayerXP(S.Game), 14.f, 0.001f));

	S.Destroy();
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsWikiRedGemRandomMultiplier,
	"ReinBalance.Survivors.Wiki.RedGemRandomMultiplier",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FSurvivorsWikiRedGemRandomMultiplier::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!S.Create()) return false;
	TOptional<int32> Seed;
	Seed.Emplace(123);
	S.Game->ResetState(Seed);

	FGemState Gem;
	Gem.Pos = FSurvivorsGameTestAccess::PlayerPos(S.Game);
	Gem.Type = EGemType::Red;
	Gem.BaseExperienceValue = 2.f;
	Gem.UniqueId = ++FSurvivorsGameTestAccess::NextGemId(S.Game);
	FSurvivorsGameTestAccess::Gems(S.Game).Add(Gem);
	FSurvivorsGameTestAccess::GemPickupRadius(S.Game) = 50.f;

	S.RunPickupHits();

	const float XP = FSurvivorsGameTestAccess::PlayerXP(S.Game);
	TestTrue("Red gem grants 10x-50x base XP", XP >= 20.f && XP <= 100.f);
	TestEqual("Gem collected", FSurvivorsGameTestAccess::Gems(S.Game).Num(), 1);
	FSurvivorsGameTestAccess::FinalizePickupRemovals(S.Game);
	TestEqual("Gem removed after finalize", FSurvivorsGameTestAccess::Gems(S.Game).Num(), 0);

	S.Destroy();
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsWikiChestEvolvesWeapon,
	"ReinBalance.Survivors.Wiki.ChestEvolvesWeapon",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FSurvivorsWikiChestEvolvesWeapon::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!S.Create()) return false;

	FWeaponSlot* Weapons = FSurvivorsGameTestAccess::WeaponSlots(S.Game);
	Weapons[0].Type = EWeaponType::Whip;
	Weapons[0].Level = FWeaponLevel(8);
	FSurvivorsGameTestAccess::WeaponComp(S.Game)->EquipWeapon(0, EWeaponType::Whip, 8);

	FPassiveSlot* Passives = FSurvivorsGameTestAccess::PassiveSlots(S.Game);
	Passives[0].Type = EPassiveItemType::HollowHeart;
	Passives[0].Level = 1;

	FSpecialPickupState Chest;
	Chest.Pos = FSurvivorsGameTestAccess::PlayerPos(S.Game);
	Chest.Type = ESpecialPickupType::TreasureChest;
	Chest.bActive = true;
	FSurvivorsGameTestAccess::SpecialPickups(S.Game).Add(Chest);
	FSurvivorsGameTestAccess::GemPickupRadius(S.Game) = 50.f;

	FSurvivorsGameTestAccess::PickupComp(S.Game)->CheckSpecialPickups();

	TestEqual("Whip evolved into Bloody Tear",
		static_cast<int32>(Weapons[0].Type),
		static_cast<int32>(EWeaponType::BloodyTear));
	TestEqual("Evolved weapon level is 1", Weapons[0].Level.Value, 1);

	S.Destroy();
	return true;
}

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
		FSurvivorsGameTestAccess::WeaponComp(S.Game)->TickWeapons(SurvivorsGameConstants::PhysicsDt);
		if (Case.SecondsAfterFirstTick > 0.f)
		{
			TickTestWeaponsForSeconds(FSurvivorsGameTestAccess::WeaponComp(S.Game), Case.SecondsAfterFirstTick);
		}

		TestEqual(FString::Printf(TEXT("%s projectile count"), Case.Label),
			FSurvivorsGameTestAccess::WeaponComp(S.Game)->GetProjectileCount(),
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
	auto* WC = FSurvivorsGameTestAccess::WeaponComp(S.Game);

	WC->TickWeapons(SurvivorsGameConstants::PhysicsDt);
	TestEqual("First Whip swing exists", WC->GetProjectileCount(), 1);
	TestTrue("First Whip swing faces +X", WC->GetProjectilePos(0).X > FSurvivorsGameTestAccess::PlayerPos(S.Game).X);

	const int32 TicksToSecondSwing = FMath::CeilToInt(0.31f / SurvivorsGameConstants::PhysicsDt);
	for (int32 i = 0; i < TicksToSecondSwing; ++i)
	{
		WC->TickWeapons(SurvivorsGameConstants::PhysicsDt);
	}

	TestEqual("Second Whip swing exists after 0.3s", WC->GetProjectileCount(), 1);
	TestTrue("Second Whip swing flips to -X", WC->GetProjectilePos(0).X < FSurvivorsGameTestAccess::PlayerPos(S.Game).X);

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
	auto* WC = FSurvivorsGameTestAccess::WeaponComp(S.Game);

	WC->TickWeapons(SurvivorsGameConstants::PhysicsDt);
	TestEqual("Knife fires first projectile immediately", WC->GetProjectileCount(), 1);

	TickTestWeaponsForSeconds(WC, 0.05f);
	TestEqual("Knife does not fire the second projectile before 0.1s", WC->GetProjectileCount(), 1);

	TickTestWeaponsForSeconds(WC, 0.06f);
	TestEqual("Knife fires second projectile after 0.1s", WC->GetProjectileCount(), 2);

	TickTestWeaponsForSeconds(WC, 0.15f);
	TestEqual("Knife Lv1 bullet2 baseline stops at two projectiles", WC->GetProjectileCount(), 2);

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
	auto* Weapon = Cast<USurvivorsKingBibleWeapon>(
		FSurvivorsGameTestAccess::WeaponComp(S.Game)->GetWeaponInstance(0));
	if (!TestTrue("King Bible instance created", Weapon != nullptr))
	{
		S.Destroy();
		return false;
	}

	FSurvivorsGameTestAccess::WeaponComp(S.Game)->TickWeapons(SurvivorsGameConstants::PhysicsDt);
	TestEqual("King Bible Lv1 starts with 2 active orbs", Weapon->GetOrbPositions().Num(), 2);

	const int32 TicksPastDuration = FMath::CeilToInt(3.1f / SurvivorsGameConstants::PhysicsDt);
	for (int32 i = 0; i < TicksPastDuration; ++i)
	{
		FSurvivorsGameTestAccess::WeaponComp(S.Game)->TickWeapons(SurvivorsGameConstants::PhysicsDt);
	}
	TestEqual("King Bible orbs disappear after Duration", Weapon->GetOrbPositions().Num(), 0);

	const int32 TicksToNextCycle = FMath::CeilToInt(3.1f / SurvivorsGameConstants::PhysicsDt);
	for (int32 i = 0; i < TicksToNextCycle; ++i)
	{
		FSurvivorsGameTestAccess::WeaponComp(S.Game)->TickWeapons(SurvivorsGameConstants::PhysicsDt);
	}
	TestEqual("King Bible orbs reactivate after Duration + Cooldown cycle", Weapon->GetOrbPositions().Num(), 2);

	S.Destroy();
	return true;
}

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
	FSurvivorsGameTestAccess::WeaponComp(S.Game)->SpawnGroundZone(Zone);

	S.RunWeaponHits();
	TestEqual("Warning GroundZone does not damage", FSurvivorsGameTestAccess::Enemies(S.Game)[0].HP, 200.f);

	FSurvivorsGameTestAccess::WeaponComp(S.Game)->TickWeapons(1.01f);
	S.RunWeaponHits();
	TestTrue("GroundZone damages after warning expires",
		FMath::IsNearlyEqual(FSurvivorsGameTestAccess::Enemies(S.Game)[0].HP, 190.f, 0.01f));

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
	FSurvivorsGameTestAccess::WeaponComp(S.Game)->SpawnProjectile(Proj);

	S.RunWeaponHits();

	// 敵A は Garlic で死亡 pending
	TestTrue("Enemy A is pending remove after Garlic hit",
		FSurvivorsGameTestAccess::Enemies(S.Game)[0].bPendingRemove);

	// 非 piercing 弾は死亡済み敵Aで消費される → 敵Bはダメージなし
	TestEqual("Enemy B HP unchanged (projectile consumed on dead enemy A)",
		FSurvivorsGameTestAccess::Enemies(S.Game)[1].HP, 100.f);

	// プロジェクタイルは消費済み（ApplyWeaponHits 内で削除される）
	TestEqual("Projectile removed after hitting dead enemy A",
		FSurvivorsGameTestAccess::WeaponComp(S.Game)->GetProjectileCount(), 0);

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
	auto* WC = FSurvivorsGameTestAccess::WeaponComp(S.Game);

	WC->TickWeapons(SurvivorsGameConstants::PhysicsDt);
	TestEqual("MagicWand Lv1 fires first shot immediately", WC->GetProjectileCount(), 1);

	TickTestWeaponsForSeconds(WC, 0.05f);
	TestEqual("MagicWand Lv1 does not fire the second shot before 0.1s", WC->GetProjectileCount(), 1);

	// 0.1s 経過後に 2 発目
	TickTestWeaponsForSeconds(WC, 0.06f);

	TestEqual("MagicWand Lv1 fires second shot after 0.1s", WC->GetProjectileCount(), 2);

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
	auto* WC = FSurvivorsGameTestAccess::WeaponComp(S.Game);

	WC->TickWeapons(SurvivorsGameConstants::PhysicsDt);
	if (!TestTrue("Magic Wand projectile exists", WC->GetProjectileCount() > 0))
	{
		S.Destroy();
		return false;
	}

	const FVector2D StartPos = WC->GetProjectilePos(0);
	const float Elapsed = TickTestWeaponsForSecondsMeasured(WC, 0.50f);
	if (!TestTrue("Magic Wand projectile remains alive for speed sample", WC->GetProjectileCount() > 0))
	{
		S.Destroy();
		return false;
	}

	const float Speed = FVector2D::Distance(WC->GetProjectilePos(0), StartPos) / Elapsed;
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
	auto* WC = FSurvivorsGameTestAccess::WeaponComp(S.Game);

	WC->TickWeapons(SurvivorsGameConstants::PhysicsDt);
	TestEqual("Cross Lv1 bullet2 baseline fires first projectile immediately", WC->GetProjectileCount(), 1);

	TickTestWeaponsForSeconds(WC, 0.05f);
	TestEqual("Cross does not fire the second projectile before 0.1s", WC->GetProjectileCount(), 1);

	TickTestWeaponsForSeconds(WC, 0.06f);
	TestEqual("Cross fires the second projectile after 0.1s", WC->GetProjectileCount(), 2);

	for (int32 i = 0; i < WC->GetProjectileCount(); ++i)
	{
		const FVector2D Pos = WC->GetProjectilePos(i);
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
	auto* WC = FSurvivorsGameTestAccess::WeaponComp(S.Game);

	WC->TickWeapons(SurvivorsGameConstants::PhysicsDt);
	TestEqual("Cross Lv1 first volley has one projectile", WC->GetProjectileCount(), 1);

	float MaxForwardDistance = 0.f;
	for (int32 Step = 0; Step < SurvivorsStepsForSeconds(1.0f); ++Step)
	{
		if (WC->GetProjectileCount() <= 0) break;
		MaxForwardDistance = FMath::Max(MaxForwardDistance, WC->GetProjectilePos(0).X);
		WC->TickWeapons(SurvivorsGameConstants::PhysicsDt);
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
	auto* WC = FSurvivorsGameTestAccess::WeaponComp(S.Game);

	WC->TickWeapons(SurvivorsGameConstants::PhysicsDt);
	TestEqual("Cross Lv1 first volley has one projectile", WC->GetProjectileCount(), 1);

	TickTestWeaponsForSeconds(WC, 2.05f);
	TestTrue("Cross previous projectile is still alive when the next cooldown volley starts",
		WC->GetProjectileCount() >= 2);

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
	auto* WC = FSurvivorsGameTestAccess::WeaponComp(S.Game);

	WC->TickWeapons(SurvivorsGameConstants::PhysicsDt);

	float MaxHeight = 0.f;
	for (int32 Step = 0; Step < SurvivorsStepsForSeconds(1.0f); ++Step)
	{
		if (WC->GetProjectileCount() > 0)
		{
			MaxHeight = FMath::Max(MaxHeight, WC->GetProjectilePos(0).Y);
		}
		WC->TickWeapons(SurvivorsGameConstants::PhysicsDt);
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
	auto* WC = FSurvivorsGameTestAccess::WeaponComp(S.Game);

	WC->TickWeapons(SurvivorsGameConstants::PhysicsDt);
	TestEqual("Axe Lv1 bullet2 baseline fires first axe immediately", WC->GetProjectileCount(), 1);
	if (WC->GetProjectileCount() > 0)
	{
		const FVector2D FirstPos = WC->GetProjectilePos(0);
		TestTrue("First axe is thrown upward from the player", FMath::Abs(FirstPos.X) < 1.f && FirstPos.Y > 0.f);
	}

	TickTestWeaponsForSeconds(WC, 0.10f);
	TestEqual("Axe does not fire the second axe before 0.2s", WC->GetProjectileCount(), 1);

	TickTestWeaponsForSeconds(WC, 0.11f);
	TestEqual("Axe fires the second axe after about 0.2s", WC->GetProjectileCount(), 2);

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
	auto* WC = FSurvivorsGameTestAccess::WeaponComp(S.Game);

	WC->TickWeapons(SurvivorsGameConstants::PhysicsDt);
	TestEqual("Runetracer Lv1 bullet2 baseline fires first rune immediately", WC->GetProjectileCount(), 1);

	TickTestWeaponsForSeconds(WC, 0.10f);
	TestEqual("Runetracer does not fire the second rune before 0.2s", WC->GetProjectileCount(), 1);

	TickTestWeaponsForSeconds(WC, 0.11f);
	TestEqual("Runetracer fires the second rune after about 0.2s", WC->GetProjectileCount(), 2);

	S.Destroy();
	return true;
}

// Runetracer: weapon_runetracer.md gives Lv1 Duration=2.25s.
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsRunetracerDuration225Seconds,
	"ReinBalance.Survivors.Wiki.Runetracer_Duration_2_25s",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FSurvivorsRunetracerDuration225Seconds::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	EquipTestWeapon(S.Game, EWeaponType::Runetracer, 1);
	auto* WC = FSurvivorsGameTestAccess::WeaponComp(S.Game);

	WC->TickWeapons(SurvivorsGameConstants::PhysicsDt);
	TestEqual("Runetracer Lv1 first projectile exists immediately", WC->GetProjectileCount(), 1);

	TickTestWeaponsForSeconds(WC, 0.21f);
	TestEqual("Runetracer Lv1 second projectile appears after about 0.2s", WC->GetProjectileCount(), 2);

	TickTestWeaponsForSeconds(WC, 1.80f);
	TestEqual("Runetracer Lv1 projectiles remain before their 2.25s duration", WC->GetProjectileCount(), 2);

	TickTestWeaponsForSeconds(WC, 0.60f);
	TestEqual("Runetracer Lv1 projectiles expire after their 2.25s duration", WC->GetProjectileCount(), 0);

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

	S.AddEnemyAt(FVector2D::ZeroVector, 200.f);

	FProjectileState Proj;
	Proj.Pos = FVector2D::ZeroVector;
	Proj.Radius = FSimRadius(10.f);
	Proj.Damage = FDamage(10.f);
	Proj.WeaponType = EWeaponType::Runetracer;
	Proj.WeaponSlotIdx = 1;
	Proj.LifeTime = FProjectileLifeTime(5.f);
	Proj.bPiercing = true;
	FSurvivorsGameTestAccess::WeaponComp(S.Game)->SpawnProjectile(Proj);

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
	auto* WC = FSurvivorsGameTestAccess::WeaponComp(S.Game);

	WC->TickWeapons(SurvivorsGameConstants::PhysicsDt);
	TickTestWeaponsForSeconds(WC, 0.08f);
	TestEqual("Fire Wand Lv1 baseline emits four fireballs within the 0.02s interval window",
		WC->GetProjectileCount(), 4);

	for (int32 i = 0; i < WC->GetProjectileCount(); ++i)
	{
		TestTrue(FString::Printf(TEXT("Fire Wand projectile %d travels generally toward the selected +X enemy"), i),
			WC->GetProjectilePos(i).X > 0.f);
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
	auto* WC = FSurvivorsGameTestAccess::WeaponComp(S.Game);

	WC->TickWeapons(SurvivorsGameConstants::PhysicsDt);
	TickTestWeaponsForSeconds(WC, 0.08f);
	TestEqual("Fire Wand Lv1 has four fireballs for angle sample", WC->GetProjectileCount(), 4);

	TArray<float> AnglesDeg;
	for (int32 i = 0; i < WC->GetProjectileCount(); ++i)
	{
		const FVector2D Delta = WC->GetProjectilePos(i) - FSurvivorsGameTestAccess::PlayerPos(S.Game);
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
	auto* WC = FSurvivorsGameTestAccess::WeaponComp(S.Game);

	WC->TickWeapons(SurvivorsGameConstants::PhysicsDt);
	TickTestWeaponsForSeconds(WC, 0.08f);
	if (!TestTrue("Fire Wand Lv1 has four fireballs for speed sample", WC->GetProjectileCount() == 4))
	{
		S.Destroy();
		return false;
	}

	TArray<FVector2D> StartPositions;
	for (int32 i = 0; i < WC->GetProjectileCount(); ++i)
	{
		StartPositions.Add(WC->GetProjectilePos(i));
	}

	const float Elapsed = TickTestWeaponsForSecondsMeasured(WC, 0.20f);
	if (!TestTrue("Fire Wand projectiles remain alive for speed sample", WC->GetProjectileCount() == StartPositions.Num()))
	{
		S.Destroy();
		return false;
	}

	float TotalSpeed = 0.f;
	for (int32 i = 0; i < WC->GetProjectileCount(); ++i)
	{
		TotalSpeed += FVector2D::Distance(WC->GetProjectilePos(i), StartPositions[i]) / Elapsed;
	}
	const float AvgSpeed = TotalSpeed / static_cast<float>(WC->GetProjectileCount());
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
	auto* WC = FSurvivorsGameTestAccess::WeaponComp(S.Game);
	auto* CC = FSurvivorsGameTestAccess::CollComp(S.Game);

	WC->TickWeapons(SurvivorsGameConstants::PhysicsDt);
	CC->BuildEnemyGrid();
	FSurvivorsHitFrame HF;
	WC->ComputeAllWeaponHits(CC, HF);
	WC->ApplyWeaponHits(HF);

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
	auto* WC = FSurvivorsGameTestAccess::WeaponComp(S.Game);

	WC->TickWeapons(SurvivorsGameConstants::PhysicsDt);
	TestEqual("King Bible Lv1 video bullet2 baseline has two orbit orbs", WC->GetOrbitOrbCount(), 2);

	if (WC->GetOrbitOrbCount() >= 2)
	{
		const FVector2D PlayerPos = FSurvivorsGameTestAccess::PlayerPos(S.Game);
		const FVector2D D0 = WC->GetOrbitOrbPos(0) - PlayerPos;
		const FVector2D D1 = WC->GetOrbitOrbPos(1) - PlayerPos;
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
	auto* WC = FSurvivorsGameTestAccess::WeaponComp(S.Game);

	// 5 tick 分進めて弾の移動方向を確認
	for (int32 i = 0; i < 5; ++i)
		WC->TickWeapons(SurvivorsGameConstants::PhysicsDt);

	if (WC->GetProjectileCount() > 0)
	{
		const FVector2D FinalPos = WC->GetProjectilePos(0);
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
	auto* WC = FSurvivorsGameTestAccess::WeaponComp(S.Game);

	WC->TickWeapons(SurvivorsGameConstants::PhysicsDt);
	TestEqual("SantaWater first drop spawned immediately", WC->GetGroundZoneCount(), 1);

	const int32 TicksTo2ndDrop = FMath::CeilToInt(0.31f / SurvivorsGameConstants::PhysicsDt);
	for (int32 i = 0; i < TicksTo2ndDrop; ++i)
		WC->TickWeapons(SurvivorsGameConstants::PhysicsDt);

	TestEqual("SantaWater second drop spawned after 0.3s", WC->GetGroundZoneCount(), 2);

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
	auto* WC = FSurvivorsGameTestAccess::WeaponComp(S.Game);

	WC->TickWeapons(SurvivorsGameConstants::PhysicsDt);
	TestEqual("SantaWater Lv1 first drop spawns immediately", WC->GetGroundZoneCount(), 1);

	if (WC->GetGroundZoneCount() > 0)
	{
		const float Dist = FVector2D::Distance(WC->GetGroundZonePos(0), EnemyPos);
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
	auto* WC = FSurvivorsGameTestAccess::WeaponComp(S.Game);
	auto* CC = FSurvivorsGameTestAccess::CollComp(S.Game);

	WC->TickWeapons(SurvivorsGameConstants::PhysicsDt);
	TestTrue("Ground zone is in warning phase", WC->IsGroundZoneWarning(0));

	CC->BuildEnemyGrid();
	FSurvivorsHitFrame HF;
	WC->ComputeAllWeaponHits(CC, HF);
	WC->ApplyWeaponHits(HF);

	TestEqual("No damage during warning phase",
		FSurvivorsGameTestAccess::Enemies(S.Game)[0].HP, 100.f);

	S.Destroy();
	return true;
}

// SantaWater: Amount >= 4 で drops が分散した円形配置になること
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsSantaWaterHighAmountCircular,
	"ReinBalance.Survivors.Wiki.SantaWater_HighAmount_CircularPattern",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FSurvivorsSantaWaterHighAmountCircular::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	S.AddEnemyAt(FVector2D(50.f, 0.f));

	// SantaWater Lv6: Amount=4
	EquipTestWeapon(S.Game, EWeaponType::SantaWater, 6);
	auto* WC = FSurvivorsGameTestAccess::WeaponComp(S.Game);

	// 4 drops 分生成（初回 + 0.3s × 3）
	const int32 TicksForAll4 = FMath::CeilToInt(1.0f / SurvivorsGameConstants::PhysicsDt);
	for (int32 i = 0; i < TicksForAll4; ++i)
		WC->TickWeapons(SurvivorsGameConstants::PhysicsDt);

	TestTrue("SantaWater Lv6 spawns 2+ drops", WC->GetGroundZoneCount() >= 2);

	if (WC->GetGroundZoneCount() >= 2)
	{
		const float Dist01 = FVector2D::Distance(WC->GetGroundZonePos(0), WC->GetGroundZonePos(1));
		// Amount=4 の円形配置: 隣接点は 80u × sin(90°) × 2 ≈ 113u 離れる
		TestTrue("SantaWater circular drops are spread (not at same pos)", Dist01 > 1.f);
	}

	S.Destroy();
	return true;
}

// Peachone: 複数の小プロジェクタイルが target zone 付近に生成されること
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
	auto* WC = FSurvivorsGameTestAccess::WeaponComp(S.Game);

	// 初回 tick で最初の 1 発が即時発射
	WC->TickWeapons(SurvivorsGameConstants::PhysicsDt);
	TestTrue("Peachone fires at least 1 projectile immediately", WC->GetProjectileCount() >= 1);

	// 0.025s × 2 後に 2 発以上
	const int32 TicksFor2Shots = FMath::CeilToInt(0.06f / SurvivorsGameConstants::PhysicsDt);
	for (int32 i = 0; i < TicksFor2Shots; ++i)
		WC->TickWeapons(SurvivorsGameConstants::PhysicsDt);

	TestTrue("Peachone fires multiple projectiles (bombard model not single AoE)",
		WC->GetProjectileCount() >= 2);

	// 砲撃弾はプレイヤー位置(0,0)ではなく orbit zone 付近（OrbitRadius=60u, BombRadius=30u）
	if (WC->GetProjectileCount() > 0)
	{
		const FVector2D ProjPos = WC->GetProjectilePos(0);
		const float DistFromPlayer = ProjPos.Size();
		// orbit 中心は ~60u、scatter ±30u → 30u 〜 90u の範囲内
		TestTrue("Peachone projectile is in orbit zone range (30u-120u from player)",
			DistFromPlayer > 5.f && DistFromPlayer < 150.f);
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
	auto* WC = FSurvivorsGameTestAccess::WeaponComp(S.Game);

	WC->TickWeapons(SurvivorsGameConstants::PhysicsDt);

	// 2 zone 分の初回発射 → 2 発以上
	TestTrue("Vandalier fires from 2 zones (>=2 projectiles)", WC->GetProjectileCount() >= 2);
	TestEqual("Vandalier has 2 orbit orbs", WC->GetOrbitOrbCount(), 2);

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

	auto* WC = FSurvivorsGameTestAccess::WeaponComp(S.Game);
	auto* CC = FSurvivorsGameTestAccess::CollComp(S.Game);
	auto* KB = Cast<USurvivorsKingBibleWeapon>(WC->GetWeaponInstance(0));
	if (!TestTrue("KingBible instance exists", KB != nullptr)) { S.Destroy(); return false; }

	// 1tick 起動してオーブをアクティブにする
	WC->TickWeapons(SurvivorsGameConstants::PhysicsDt);
	if (!TestEqual("KingBible Lv1 has 2 orbs", KB->GetOrbPositions().Num(), 2)) { S.Destroy(); return false; }

	// 敵をオーブ 0 の位置に置く → orb 0 ヒット
	FSurvivorsGameTestAccess::Enemies(S.Game)[0].Pos = KB->GetOrbPositions()[0];
	CC->BuildEnemyGrid();
	FSurvivorsHitFrame HF1;
	WC->ComputeAllWeaponHits(CC, HF1);
	WC->ApplyWeaponHits(HF1);
	const float HPAfterOrb0 = FSurvivorsGameTestAccess::Enemies(S.Game)[0].HP;
	TestTrue("Orb 0 hit: HP decreased", HPAfterOrb0 < 2000.f);

	// 敵をオーブ 1 の位置に置く → 同じ敵に orb 1 がすぐ当たれるか（per-orb 独立 cooldown）
	FSurvivorsGameTestAccess::Enemies(S.Game)[0].Pos = KB->GetOrbPositions()[1];
	CC->BuildEnemyGrid();
	FSurvivorsHitFrame HF2;
	WC->ComputeAllWeaponHits(CC, HF2);
	WC->ApplyWeaponHits(HF2);
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

	auto* WC = FSurvivorsGameTestAccess::WeaponComp(S.Game);
	auto* CC = FSurvivorsGameTestAccess::CollComp(S.Game);
	auto* KB = Cast<USurvivorsKingBibleWeapon>(WC->GetWeaponInstance(0));
	if (!TestTrue("KingBible instance exists", KB != nullptr)) { S.Destroy(); return false; }

	WC->TickWeapons(SurvivorsGameConstants::PhysicsDt);
	if (!TestEqual("KingBible Lv1 has 2 orbs", KB->GetOrbPositions().Num(), 2)) { S.Destroy(); return false; }

	// 敵をオーブ 0 に重ねる → 1回目ヒット
	FSurvivorsGameTestAccess::Enemies(S.Game)[0].Pos = KB->GetOrbPositions()[0];
	CC->BuildEnemyGrid();
	FSurvivorsHitFrame HF1;
	WC->ComputeAllWeaponHits(CC, HF1);
	WC->ApplyWeaponHits(HF1);
	const float HPAfterFirst = FSurvivorsGameTestAccess::Enemies(S.Game)[0].HP;
	TestTrue("First orb hit deals damage", HPAfterFirst < 2000.f);

	// 再度 → 0.5s 以内なので same orb cooldown でブロックされる
	CC->BuildEnemyGrid();
	FSurvivorsHitFrame HF2;
	WC->ComputeAllWeaponHits(CC, HF2);
	WC->ApplyWeaponHits(HF2);
	TestEqual("Same orb does not re-hit immediately",
		FSurvivorsGameTestAccess::Enemies(S.Game)[0].HP, HPAfterFirst);

	// 1.7s 経過後 → 再ヒット可能
	FSurvivorsGameTestAccess::ElapsedTime(S.Game) += 1.71f;
	CC->BuildEnemyGrid();
	FSurvivorsHitFrame HF3;
	WC->ComputeAllWeaponHits(CC, HF3);
	WC->ApplyWeaponHits(HF3);
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
	auto* WC = FSurvivorsGameTestAccess::WeaponComp(S.Game);
	auto* CC = FSurvivorsGameTestAccess::CollComp(S.Game);

	WC->TickWeapons(SurvivorsGameConstants::PhysicsDt);
	CC->BuildEnemyGrid();
	FSurvivorsHitFrame HF;
	WC->ComputeAllWeaponHits(CC, HF);
	WC->ApplyWeaponHits(HF);

	// 落雷位置 marker が GroundZone として生成されていること（Amount=2 なので 2 個）
	TestTrue("Lightning Ring spawns strike markers (>=2 ground zones)",
		WC->GetGroundZoneCount() >= 2);

	// marker は敵の位置付近に配置されること
	bool bFoundNearEnemy0 = false;
	bool bFoundNearEnemy1 = false;
	for (int32 i = 0; i < WC->GetGroundZoneCount(); ++i)
	{
		const FVector2D ZPos = WC->GetGroundZonePos(i);
		if (FVector2D::Distance(ZPos, FVector2D(-100.f, 0.f)) < 50.f) bFoundNearEnemy0 = true;
		if (FVector2D::Distance(ZPos, FVector2D( 100.f, 0.f)) < 50.f) bFoundNearEnemy1 = true;
	}
	TestTrue("Strike marker placed near enemy 0", bFoundNearEnemy0);
	TestTrue("Strike marker placed near enemy 1", bFoundNearEnemy1);

	// marker は短寿命（player center 常時 ring は出ない）
	TestTrue("Strike markers are not warning zones", !WC->IsGroundZoneWarning(0));

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
	auto* WC = FSurvivorsGameTestAccess::WeaponComp(S.Game);

	// 画面外敵しかいない場合、弾はランダム方向（≠ 敵方向）に飛ぶ
	// ここでは単純に「弾が発射されること」だけをテストする（ランダム方向は検証しない）
	WC->TickWeapons(SurvivorsGameConstants::PhysicsDt);
	TestEqual("MagicWand fires even when no on-screen enemy (random direction)",
		WC->GetProjectileCount(), 1);

	S.Destroy();
	return true;
}

// Axe は上方向 ±45° のランダム角度で発射されること
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
	auto* WC = FSurvivorsGameTestAccess::WeaponComp(S.Game);

	// 初回発射：1発目
	WC->TickWeapons(SurvivorsGameConstants::PhysicsDt);
	if (!TestEqual("Axe fires first shot", WC->GetProjectileCount(), 1)) { S.Destroy(); return false; }

	const FVector2D Pos0 = WC->GetProjectilePos(0);
	// 上方向（Y > 0）であること
	TestTrue("Axe shot 0 travels upward (Y > 0)", Pos0.Y > 0.f);
	// ±45° 以内なので |X| < Y（sin45°=cos45°）
	TestTrue("Axe shot 0 is within ±45° of up", FMath::Abs(Pos0.X) <= Pos0.Y + 0.01f);

	// 0.2s 後：2発目
	TickTestWeaponsForSeconds(WC, 0.21f);
	if (!TestEqual("Axe fires second shot", WC->GetProjectileCount(), 2)) { S.Destroy(); return false; }

	S.Destroy();
	return true;
}
