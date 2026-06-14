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
