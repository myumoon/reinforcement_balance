#include "Misc/AutomationTest.h"
#include "SurvivorsTestHelpers.h"

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

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsResetStateClearsActorRuntimeState,
	"ReinBalance.Survivors.Reset.ResetState_ClearsActorRuntimeState",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FSurvivorsResetStateClearsActorRuntimeState::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!TestTrue("World created", S.Create())) return false;

	ASurvivorsGame* Game = S.Game;

	FSurvivorsGameTestAccess::ActorPlayerPos(Game) = FVector2D(120.f, -45.f);
	FSurvivorsGameTestAccess::ActorPlayerVel(Game) = FVector2D(9.f, 3.f);
	FSurvivorsGameTestAccess::ActorPlayerHP(Game) = 1.f;
	FSurvivorsGameTestAccess::ActorPlayerXP(Game) = 42.f;
	FSurvivorsGameTestAccess::ActorPlayerLevel(Game) = 7;

	FWeaponSlot* ActorWeapons = FSurvivorsGameTestAccess::ActorWeaponSlots(Game);
	ActorWeapons[0].Type = EWeaponType::KingBible;
	ActorWeapons[0].Level = FWeaponLevel(8);
	ActorWeapons[0].Cooldown = FCooldownSeconds(3.f);

	FPassiveSlot* ActorPassives = FSurvivorsGameTestAccess::ActorPassiveSlots(Game);
	ActorPassives[0].Type = EPassiveItemType::Tirajisu;
	ActorPassives[0].Level = 2;

	FSurvivorsGameTestAccess::ActorPassiveEffects(Game).DamageMult = 2.f;
	FSurvivorsGameTestAccess::ActorPassiveEffects(Game).CooldownMult = 0.5f;
	FSurvivorsGameTestAccess::ActorPassiveEffects(Game).MaxRevivalCount = 2;
	FSurvivorsGameTestAccess::ActorGlobalFreezeUntilTime(Game) = 99.f;
	FSurvivorsGameTestAccess::ActorPlayerShieldTimer(Game) = 5.f;
	FSurvivorsGameTestAccess::ActorShieldActive(Game) = true;
	FSurvivorsGameTestAccess::ActorMaxRevivalCount(Game) = 2;
	FSurvivorsGameTestAccess::ActorUsedRevivalCount(Game) = 1;
	FSurvivorsGameTestAccess::ActorNextEnemyId(Game) = 11;
	FSurvivorsGameTestAccess::ActorNextGemId(Game) = 13;
	FSurvivorsGameTestAccess::ActorElapsedTime(Game) = 50.f;
	FSurvivorsGameTestAccess::ActorSpawnAccumulator(Game) = 4.f;
	FSurvivorsGameTestAccess::ActorBossSpawned(Game) = true;
	FSurvivorsGameTestAccess::ActorLastReward(Game) = 7.f;
	FSurvivorsGameTestAccess::ActorEpisodeBaseReward(Game) = 8.f;
	FSurvivorsGameTestAccess::ActorEpisodeStepCount(Game) = 9;
	FSurvivorsGameTestAccess::ActorDone(Game) = true;
	FSurvivorsGameTestAccess::ActorTruncated(Game) = true;
	FSurvivorsGameTestAccess::ActorPhysicsAccumTime(Game) = 0.01f;
	FSurvivorsGameTestAccess::ActorLastSpawnDebug(Game).EnemyCount = 1;
	FSurvivorsGameTestAccess::ActorLastSpawnDebug(Game).SpawnAccumulator = 4.f;

	FEnemyState Enemy;
	Enemy.Pos = FVector2D::ZeroVector;
	Enemy.UniqueId = 101;
	Enemy.CollisionRadius = 10.f;
	FSurvivorsGameTestAccess::ActorEnemies(Game).Add(Enemy);

	FGemState Gem;
	Gem.Pos = FVector2D(20.f, 0.f);
	Gem.UniqueId = 202;
	FSurvivorsGameTestAccess::ActorGems(Game).Add(Gem);
	FSurvivorsGameTestAccess::ActorFloorPickups(Game).Add(FFloorPickupState());
	FSurvivorsGameTestAccess::ActorSpecialPickups(Game).Add(FSpecialPickupState());
	FSurvivorsGameTestAccess::ActorDestructibles(Game).Add(FDestructibleState());

	USurvivorsCollisionComponent* CollComp = FSurvivorsGameTestAccess::CollComp(Game);
	CollComp->BuildEnemyGrid();
	TArray<const FSurvivorsTargetProxy*> ContactsBefore;
	CollComp->QueryEnemyContacts(FVector2D::ZeroVector, 50.f, ContactsBefore);
	TestTrue("Legacy collision grid is populated before reset", ContactsBefore.Num() > 0);

	Game->ResetState(TOptional<int32>(321));

	TestTrue("Actor player position reset", FSurvivorsGameTestAccess::ActorPlayerPos(Game).IsNearlyZero());
	TestTrue("Actor player velocity reset", FSurvivorsGameTestAccess::ActorPlayerVel(Game).IsNearlyZero());
	TestEqual("Actor player HP reset", FSurvivorsGameTestAccess::ActorPlayerHP(Game), Game->MaxPlayerHP);
	TestEqual("Actor player XP reset", FSurvivorsGameTestAccess::ActorPlayerXP(Game), 0.f);
	TestEqual("Actor player level reset", FSurvivorsGameTestAccess::ActorPlayerLevel(Game), 1);
	TestEqual("Actor weapon slot reset", static_cast<int32>(ActorWeapons[0].Type), static_cast<int32>(EWeaponType::None));
	TestEqual("Actor weapon cooldown reset", ActorWeapons[0].Cooldown.Value, 0.f);
	TestEqual("Actor passive slot reset", static_cast<int32>(ActorPassives[0].Type), static_cast<int32>(EPassiveItemType::None));
	TestEqual("Actor passive effects reset", FSurvivorsGameTestAccess::ActorPassiveEffects(Game).DamageMult, 1.f);
	TestEqual("Actor freeze timer reset", FSurvivorsGameTestAccess::ActorGlobalFreezeUntilTime(Game), -1.f);
	TestEqual("Actor shield timer reset", FSurvivorsGameTestAccess::ActorPlayerShieldTimer(Game), 0.f);
	TestFalse("Actor shield flag reset", FSurvivorsGameTestAccess::ActorShieldActive(Game));
	TestEqual("Actor max revival reset", FSurvivorsGameTestAccess::ActorMaxRevivalCount(Game), 0);
	TestEqual("Actor used revival reset", FSurvivorsGameTestAccess::ActorUsedRevivalCount(Game), 0);
	TestEqual("Actor enemy id reset", FSurvivorsGameTestAccess::ActorNextEnemyId(Game), 0);
	TestEqual("Actor gem id reset", FSurvivorsGameTestAccess::ActorNextGemId(Game), 0);
	TestEqual("Actor enemies cleared", FSurvivorsGameTestAccess::ActorEnemies(Game).Num(), 0);
	TestEqual("Actor gems cleared", FSurvivorsGameTestAccess::ActorGems(Game).Num(), 0);
	TestEqual("Actor floor pickups cleared", FSurvivorsGameTestAccess::ActorFloorPickups(Game).Num(), 0);
	TestEqual("Actor special pickups cleared", FSurvivorsGameTestAccess::ActorSpecialPickups(Game).Num(), 0);
	TestEqual("Actor destructibles cleared", FSurvivorsGameTestAccess::ActorDestructibles(Game).Num(), 0);
	TestEqual("Actor elapsed time reset", FSurvivorsGameTestAccess::ActorElapsedTime(Game), 0.f);
	TestEqual("Actor spawn accumulator reset", FSurvivorsGameTestAccess::ActorSpawnAccumulator(Game), 0.f);
	TestFalse("Actor boss flag reset", FSurvivorsGameTestAccess::ActorBossSpawned(Game));
	TestEqual("Actor last reward reset", FSurvivorsGameTestAccess::ActorLastReward(Game), 0.f);
	TestEqual("Actor episode base reward reset", FSurvivorsGameTestAccess::ActorEpisodeBaseReward(Game), 0.f);
	TestEqual("Actor episode step count reset", FSurvivorsGameTestAccess::ActorEpisodeStepCount(Game), 0);
	TestFalse("Actor done flag reset", FSurvivorsGameTestAccess::ActorDone(Game));
	TestFalse("Actor truncated flag reset", FSurvivorsGameTestAccess::ActorTruncated(Game));
	TestEqual("Actor variable-step accumulator reset", FSurvivorsGameTestAccess::ActorPhysicsAccumTime(Game), 0.f);
	TestEqual("Actor spawn debug reset", FSurvivorsGameTestAccess::ActorLastSpawnDebug(Game).EnemyCount, 0);

	TArray<const FSurvivorsTargetProxy*> ContactsAfter;
	CollComp->QueryEnemyContacts(FVector2D::ZeroVector, 50.f, ContactsAfter);
	TestEqual("Legacy collision grid cleared", ContactsAfter.Num(), 0);

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
	FSurvivorsGameTestAccess::RecalcPassiveEffects(S.Game);

	TestTrue("Crown level 5 gives 1.4x Growth",
		FMath::IsNearlyEqual(FSurvivorsGameTestAccess::PassiveEffects(S.Game).GrowthMult, 1.4f, 0.001f));
	TestTrue("Attractorb level 5 uses wiki multiplier",
		FMath::IsNearlyEqual(FSurvivorsGameTestAccess::GemPickupRadius(S.Game), 50.f * 3.980025f, 0.01f));

	FSurvivorsGameTestAccess::ProcessXPGain(S.Game, 10.f);
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
	S.Game->GetLogic()->EquipWeapon(0, EWeaponType::Whip, 8);

	FPassiveSlot* Passives = FSurvivorsGameTestAccess::PassiveSlots(S.Game);
	Passives[0].Type = EPassiveItemType::HollowHeart;
	Passives[0].Level = 1;

	FSpecialPickupState Chest;
	Chest.Pos = FSurvivorsGameTestAccess::PlayerPos(S.Game);
	Chest.Type = ESpecialPickupType::TreasureChest;
	Chest.bActive = true;
	FSurvivorsGameTestAccess::SpecialPickups(S.Game).Add(Chest);
	FSurvivorsGameTestAccess::GemPickupRadius(S.Game) = 50.f;

	FSurvivorsGameTestAccess::CheckSpecialPickups(S.Game);

	TestEqual("Whip evolved into Bloody Tear",
		static_cast<int32>(Weapons[0].Type),
		static_cast<int32>(EWeaponType::BloodyTear));
	TestEqual("Evolved weapon level is 1", Weapons[0].Level.Value, 1);

	S.Destroy();
	return true;
}
