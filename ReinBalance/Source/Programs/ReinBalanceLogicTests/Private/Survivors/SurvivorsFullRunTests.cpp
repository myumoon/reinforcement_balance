/**
 * Survivors 30分 terminal と RSI loadout validation を検証する。
 * 初心者向け: 完走、短い timeout、不正装備が canonical Logic で区別されることを確認する。
 */
#include "TestHarness.h"

#include "Survivors/SurvivorsGameConstants.h"
#include "Survivors/SurvivorsGameLogic.h"

TEST_CASE("Survivors full run clears exactly at 1800 seconds", "[unit][survivors][full-run]")
{
	FSurvivorsGameLogicConfig Config;
	Config.MaxEpisodeTime = SurvivorsGameConstants::MaxGameTime;
	Config.bHasInitialOverride = true;
	Config.InitialElapsedTime = SurvivorsGameConstants::MaxGameTime - SurvivorsGameConstants::PhysicsDt;

	FSurvivorsGameLogic Logic;
	REQUIRE(Logic.Initialize(Config));
	Logic.Reset(73013);
	const FSurvivorsStepResult Result = Logic.ExecStep({8.f}, 1);

	CHECK(Result.bTruncated);
	CHECK(Result.bStageCleared);
	CHECK(!Result.bTimedOut);
	CHECK(!Result.bDone);
	CHECK(Logic.IsStageClear());
	CHECK(!Logic.IsTimedOut());
	CHECK(Logic.GetSpawnDebugJson().Contains(TEXT("\"stage_cleared\":true")));
}

TEST_CASE("Survivors short curriculum terminal remains timeout", "[unit][survivors][full-run]")
{
	FSurvivorsGameLogicConfig Config;
	Config.MaxEpisodeTime = 300.f;
	Config.bHasInitialOverride = true;
	Config.InitialElapsedTime = 300.f - SurvivorsGameConstants::PhysicsDt;

	FSurvivorsGameLogic Logic;
	REQUIRE(Logic.Initialize(Config));
	Logic.Reset(73019);
	const FSurvivorsStepResult Result = Logic.ExecStep({8.f}, 1);

	CHECK(Result.bTruncated);
	CHECK(!Result.bStageCleared);
	CHECK(Result.bTimedOut);
	CHECK(!Logic.IsStageClear());
	CHECK(Logic.IsTimedOut());
}

TEST_CASE("Survivors invalid initial loadouts are rejected", "[unit][survivors][full-run][loadout]")
{
	FSurvivorsGameLogic Logic;
	FSurvivorsGameLogicConfig Baseline;
	Baseline.bHasInitialOverride = true;
	Baseline.InitialWeaponSlots.Add({1, 2});
	Baseline.InitialPassiveSlots.Add({1, 2});
	REQUIRE(Logic.Initialize(Baseline));

	FSurvivorsGameLogicConfig InvalidWeapon;
	InvalidWeapon.bHasInitialOverride = true;
	InvalidWeapon.InitialWeaponSlots.Add({999, 1});
	CHECK(!Logic.ApplyConfig(InvalidWeapon));

	FSurvivorsGameLogicConfig InvalidElapsed;
	InvalidElapsed.bHasInitialOverride = true;
	InvalidElapsed.InitialElapsedTime = SurvivorsGameConstants::MaxGameTime + 1.f;
	CHECK(!Logic.ApplyConfig(InvalidElapsed));

	FSurvivorsGameLogicConfig DuplicateWeapon;
	DuplicateWeapon.bHasInitialOverride = true;
	DuplicateWeapon.InitialWeaponSlots.Add({1, 1});
	DuplicateWeapon.InitialWeaponSlots.Add({1, 2});
	CHECK(!Logic.ApplyConfig(DuplicateWeapon));

	FSurvivorsGameLogicConfig TooManyWeapons;
	TooManyWeapons.bHasInitialOverride = true;
	for (int32 WeaponId = 1; WeaponId <= SurvivorsGameConstants::MaxWeaponSlots + 1; ++WeaponId)
	{
		TooManyWeapons.InitialWeaponSlots.Add({WeaponId, 1});
	}
	CHECK(!Logic.ApplyConfig(TooManyWeapons));

	FSurvivorsGameLogicConfig InvalidWeaponLevel;
	InvalidWeaponLevel.bHasInitialOverride = true;
	InvalidWeaponLevel.InitialWeaponSlots.Add({1, 9});
	CHECK(!Logic.ApplyConfig(InvalidWeaponLevel));

	FSurvivorsGameLogicConfig InvalidPassiveId;
	InvalidPassiveId.bHasInitialOverride = true;
	InvalidPassiveId.InitialPassiveSlots.Add({999, 1});
	CHECK(!Logic.ApplyConfig(InvalidPassiveId));

	FSurvivorsGameLogicConfig DuplicatePassive;
	DuplicatePassive.bHasInitialOverride = true;
	DuplicatePassive.InitialPassiveSlots.Add({1, 1});
	DuplicatePassive.InitialPassiveSlots.Add({1, 2});
	CHECK(!Logic.ApplyConfig(DuplicatePassive));

	FSurvivorsGameLogicConfig TooManyPassives;
	TooManyPassives.bHasInitialOverride = true;
	for (int32 PassiveId = 1; PassiveId <= SurvivorsGameConstants::MaxPassiveSlots + 1; ++PassiveId)
	{
		TooManyPassives.InitialPassiveSlots.Add({PassiveId, 1});
	}
	CHECK(!Logic.ApplyConfig(TooManyPassives));

	FSurvivorsGameLogicConfig InvalidPassiveLevel;
	InvalidPassiveLevel.bHasInitialOverride = true;
	InvalidPassiveLevel.InitialPassiveSlots.Add({1, 99});
	CHECK(!Logic.ApplyConfig(InvalidPassiveLevel));

	Logic.Reset(73037);
	CHECK(Logic.GetWeaponSlot(0).Type == EWeaponType::Garlic);
	CHECK(Logic.GetWeaponSlot(0).Level.Value == 2);
	CHECK(Logic.GetPassiveSlot(0).Type == EPassiveItemType::Spinach);
	CHECK(Logic.GetPassiveSlot(0).Level == 2);
}

TEST_CASE("Survivors FR0 through FR4 loadouts are accepted", "[unit][survivors][full-run][loadout]")
{
	TArray<FSurvivorsGameLogicConfig> Bands;

	FSurvivorsGameLogicConfig FR0;
	FR0.bHasInitialOverride = true;
	FR0.InitialWeaponSlots.Add({1, 1});
	Bands.Add(FR0);

	FSurvivorsGameLogicConfig FR1;
	FR1.bHasInitialOverride = true;
	FR1.InitialWeaponSlots = {{1, 4}, {7, 4}};
	FR1.InitialPassiveSlots = {{4, 2}, {8, 2}};
	Bands.Add(FR1);

	FSurvivorsGameLogicConfig FR2;
	FR2.bHasInitialOverride = true;
	FR2.InitialWeaponSlots = {{1, 8}, {7, 8}, {3, 6}, {9, 6}};
	FR2.InitialPassiveSlots = {{4, 5}, {8, 5}, {5, 4}, {11, 4}};
	Bands.Add(FR2);

	FSurvivorsGameLogicConfig FR3;
	FR3.bHasInitialOverride = true;
	FR3.InitialWeaponSlots = {{16, 1}, {22, 1}, {18, 1}, {24, 1}, {25, 1}, {26, 1}};
	FR3.InitialPassiveSlots = {{4, 5}, {8, 5}, {5, 5}, {11, 5}, {2, 5}, {9, 2}};
	Bands.Add(FR3);

	FSurvivorsGameLogicConfig FR4;
	Bands.Add(FR4);

	for (const FSurvivorsGameLogicConfig& Band : Bands)
	{
		CHECK(FSurvivorsGameLogic::IsValidInitialLoadout(Band));
	}
}
