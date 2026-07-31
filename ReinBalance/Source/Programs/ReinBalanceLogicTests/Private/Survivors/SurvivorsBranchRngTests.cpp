/**
 * validation-only branch RNG と通常 training RNG の分離を検証する。
 * reset/step の既存 stream semantics と candidate 共通 stream binding を純 Logic で確認する。
 */
#include "TestHarness.h"

#include "Survivors/SurvivorsGameLogic.h"

namespace
{
struct FActionTrace
{
	TArray<TArray<float>> Observations;
	TArray<float> Rewards;
	TArray<int32> RngStates;
};

FActionTrace RunTrace(FSurvivorsGameLogic& Logic, int32 StepCount)
{
	FActionTrace Trace;
	for (int32 Step = 0; Step < StepCount; ++Step)
	{
		Logic.PhysicsStep(Step % 9);
		Trace.Observations.Add(Logic.GetObservation());
		Trace.Rewards.Add(Logic.GetReward());
		Trace.RngStates.Add(Logic.GetValidationBranchRngCurrentState());
		if (Logic.IsDone() || Logic.IsTruncated() || Logic.IsLevelUpPending())
		{
			break;
		}
	}
	return Trace;
}

void CheckTraceEqual(const FActionTrace& Left, const FActionTrace& Right)
{
	REQUIRE(Left.Observations.Num() == Right.Observations.Num());
	REQUIRE(Left.Rewards.Num() == Right.Rewards.Num());
	REQUIRE(Left.RngStates.Num() == Right.RngStates.Num());
	for (int32 Index = 0; Index < Left.Observations.Num(); ++Index)
	{
		CHECK(Left.Observations[Index] == Right.Observations[Index]);
		CHECK(Left.Rewards[Index] == Right.Rewards[Index]);
		CHECK(Left.RngStates[Index] == Right.RngStates[Index]);
	}
}
}

TEST_CASE("Survivors validation branch RNG does not change normal reset and step semantics",
	"[unit][survivors][branch-rng][regression]")
{
	FSurvivorsGameLogicConfig Config;
	Config.MinActiveEnemies = 2;
	Config.MaxActiveEnemies = 8;
	Config.ItemSelectionMode = TEXT("auto");

	FSurvivorsGameLogic Expected;
	Expected.Initialize(Config);
	Expected.Reset(12345);
	const FActionTrace ExpectedTrace = RunTrace(Expected, 80);

	FSurvivorsGameLogic AfterValidation;
	AfterValidation.Initialize(Config);
	AfterValidation.Reset(77);
	REQUIRE(AfterValidation.ActivateValidationBranchRng(
		FSurvivorsGameLogic::GetValidationBranchRngSchemaVersion(),
		FString::ChrN(64, TEXT('a')),
		991));
	RunTrace(AfterValidation, 10);

	AfterValidation.Reset(12345);
	CHECK(!AfterValidation.GetValidationBranchRngState().bActive);
	const FActionTrace ActualTrace = RunTrace(AfterValidation, 80);
	CheckTraceEqual(ExpectedTrace, ActualTrace);
}

TEST_CASE("Survivors validation candidates share the same replication stream",
	"[unit][survivors][branch-rng][crn]")
{
	FSurvivorsGameLogicConfig Config;
	Config.MinActiveEnemies = 1;
	Config.MaxActiveEnemies = 4;

	FSurvivorsGameLogic FirstCandidate;
	FSurvivorsGameLogic SecondCandidate;
	FirstCandidate.Initialize(Config);
	SecondCandidate.Initialize(Config);
	FirstCandidate.Reset(2026);
	SecondCandidate.Reset(2026);
	CheckTraceEqual(
		RunTrace(FirstCandidate, 20),
		RunTrace(SecondCandidate, 20));

	const FString ReplicationKey = FString::ChrN(64, TEXT('b'));
	REQUIRE(FirstCandidate.ActivateValidationBranchRng(
		FSurvivorsGameLogic::GetValidationBranchRngSchemaVersion(),
		ReplicationKey,
		314159));
	REQUIRE(SecondCandidate.ActivateValidationBranchRng(
		FSurvivorsGameLogic::GetValidationBranchRngSchemaVersion(),
		ReplicationKey,
		314159));
	CHECK(FirstCandidate.GetValidationBranchRngState().InitialStreamState
		== SecondCandidate.GetValidationBranchRngState().InitialStreamState);
	CheckTraceEqual(
		RunTrace(FirstCandidate, 40),
		RunTrace(SecondCandidate, 40));
}

TEST_CASE("Survivors validation branch RNG rejects unbound schema and key atomically",
	"[unit][survivors][branch-rng][binding]")
{
	FSurvivorsGameLogic Logic;
	Logic.Initialize(FSurvivorsGameLogicConfig());
	Logic.Reset(42);
	const int32 Before = Logic.GetValidationBranchRngCurrentState();

	CHECK(!Logic.ActivateValidationBranchRng(
		TEXT("survivors.branch_rng.v999"),
		FString::ChrN(64, TEXT('c')),
		12));
	CHECK(Logic.GetValidationBranchRngCurrentState() == Before);
	CHECK(!Logic.GetValidationBranchRngState().bActive);

	CHECK(!Logic.ActivateValidationBranchRng(
		FSurvivorsGameLogic::GetValidationBranchRngSchemaVersion(),
		TEXT("not-a-sha256"),
		12));
	CHECK(Logic.GetValidationBranchRngCurrentState() == Before);
	CHECK(!Logic.GetValidationBranchRngState().bActive);
}
