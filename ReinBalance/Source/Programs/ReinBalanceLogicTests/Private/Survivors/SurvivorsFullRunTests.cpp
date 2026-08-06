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
	FSurvivorsGameLogicConfig InvalidWeapon;
	InvalidWeapon.bHasInitialOverride = true;
	InvalidWeapon.InitialWeaponSlots.Add({999, 1});
	CHECK(!Logic.Initialize(InvalidWeapon));

	FSurvivorsGameLogicConfig DuplicateWeapon;
	DuplicateWeapon.bHasInitialOverride = true;
	DuplicateWeapon.InitialWeaponSlots.Add({1, 1});
	DuplicateWeapon.InitialWeaponSlots.Add({1, 2});
	CHECK(!Logic.ApplyConfig(DuplicateWeapon));

	FSurvivorsGameLogicConfig InvalidPassive;
	InvalidPassive.bHasInitialOverride = true;
	InvalidPassive.InitialPassiveSlots.Add({9, 3});
	CHECK(!FSurvivorsGameLogic::IsValidInitialLoadout(InvalidPassive));
}

