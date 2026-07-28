/**
 * 外部レベルアップ選択の純粋な状態遷移を検証する。
 * 初心者向け: HTTP や画面を使わず、保留・再送・古い ID の扱いだけを高速に確認する。
 */
#include "TestHarness.h"

#include "Survivors/SurvivorsGameLogic.h"
#include "Survivors/SurvivorsLevelUpDecision.h"

namespace
{
FLevelUpChoice MakeWeaponChoice(EWeaponType Type)
{
	FLevelUpChoice Choice;
	Choice.ChoiceType = FLevelUpChoice::EChoiceType::WeaponNew;
	Choice.WeaponType = Type;
	Choice.NewLevel = 1;
	return Choice;
}
}

TEST_CASE("Survivors external level-up decision is deterministic and idempotent",
	"[unit][survivors][level-up][decision]")
{
	FSurvivorsLevelUpDecisionState State;
	State.Reset(41);

	const TArray<FLevelUpChoice> Choices = {
		MakeWeaponChoice(EWeaponType::Whip),
		MakeWeaponChoice(EWeaponType::Axe),
	};
	REQUIRE(State.BeginDecision(2, Choices));
	const FSurvivorsPendingLevelUpDecision First = State.GetPending();

	CHECK(First.DecisionId == TEXT("level-up-41-1-2"));
	REQUIRE(First.Choices.Num() == 2);
	CHECK(First.Choices[0].ChoiceId == TEXT("choice-0"));
	CHECK(First.Choices[1].ChoiceId == TEXT("choice-1"));

	int32 ChoiceIndex = INDEX_NONE;
	CHECK(State.ValidateChoice(First.DecisionId, TEXT("choice-999"), ChoiceIndex)
		== ESurvivorsLevelUpChoiceValidation::InvalidChoice);
	CHECK(State.ValidateChoice(First.DecisionId, TEXT("choice-1"), ChoiceIndex)
		== ESurvivorsLevelUpChoiceValidation::Accepted);
	CHECK(ChoiceIndex == 1);
	State.CommitChoice(First.DecisionId, TEXT("choice-1"));

	CHECK(State.ValidateChoice(First.DecisionId, TEXT("choice-1"), ChoiceIndex)
		== ESurvivorsLevelUpChoiceValidation::Duplicate);
	CHECK(State.ValidateChoice(First.DecisionId, TEXT("choice-0"), ChoiceIndex)
		== ESurvivorsLevelUpChoiceValidation::StaleDecision);
	CHECK(State.ValidateChoice(TEXT("level-up-41-999-2"), TEXT("choice-1"), ChoiceIndex)
		== ESurvivorsLevelUpChoiceValidation::StaleDecision);
}

TEST_CASE("Survivors level-up decision reset rejects old IDs and empty offers",
	"[unit][survivors][level-up][reset]")
{
	FSurvivorsLevelUpDecisionState State;
	State.Reset(1);
	CHECK(!State.BeginDecision(2, {}));

	REQUIRE(State.BeginDecision(2, {MakeWeaponChoice(EWeaponType::Knife)}));
	const FString OldId = State.GetPending().DecisionId;

	State.Reset(2);
	int32 ChoiceIndex = INDEX_NONE;
	CHECK(State.ValidateChoice(OldId, TEXT("choice-0"), ChoiceIndex)
		== ESurvivorsLevelUpChoiceValidation::StaleDecision);

	REQUIRE(State.BeginDecision(2, {MakeWeaponChoice(EWeaponType::Knife)}));
	CHECK(State.GetPending().DecisionId == TEXT("level-up-2-1-2"));
	CHECK(State.GetPending().DecisionId != OldId);
}

TEST_CASE("Survivors auto level-up remains seeded for uniform and weighted pools",
	"[unit][survivors][level-up][auto]")
{
	for (const FString Mode : {FString(TEXT("all_base")), FString(TEXT("weighted"))})
	{
		FSurvivorsGameLogicConfig Config;
		Config.ItemSelectionMode = TEXT("auto");
		Config.WeaponPoolMode = Mode;
		if (Mode == TEXT("weighted"))
		{
			Config.AllowedWeaponTypes = {
				static_cast<int32>(EWeaponType::Whip),
				static_cast<int32>(EWeaponType::Axe),
			};
			Config.WeaponWeights.Add(static_cast<int32>(EWeaponType::Whip), 4.f);
			Config.WeaponWeights.Add(static_cast<int32>(EWeaponType::Axe), 1.f);
		}

		FSurvivorsGameLogic First;
		FSurvivorsGameLogic Second;
		First.Initialize(Config);
		Second.Initialize(Config);
		First.Reset(1234);
		Second.Reset(1234);
		First.AddExperience(100.f);
		Second.AddExperience(100.f);

		CHECK(!First.IsLevelUpPending());
		CHECK(First.GetPlayerLevel() == Second.GetPlayerLevel());
		for (int32 Slot = 0; Slot < SurvivorsGameConstants::MaxWeaponSlots; ++Slot)
		{
			CHECK(First.GetWeaponSlot(Slot).Type == Second.GetWeaponSlot(Slot).Type);
			CHECK(First.GetWeaponSlot(Slot).Level.Value
				== Second.GetWeaponSlot(Slot).Level.Value);
		}
	}
}

TEST_CASE("Survivors external overflow advances one level and freezes simulation",
	"[unit][survivors][level-up][external][pause]")
{
	FSurvivorsGameLogicConfig Config;
	Config.ItemSelectionMode = TEXT("external");
	Config.WeaponPoolMode = TEXT("all_base");
	Config.MinActiveEnemies = 0;
	Config.MaxActiveEnemies = 1;

	FSurvivorsGameLogic Logic;
	Logic.Initialize(Config);
	Logic.Reset(7);
	Logic.AddExperience(1000.f);

	REQUIRE(Logic.IsLevelUpPending());
	CHECK(Logic.GetPlayerLevel() == 2);
	CHECK(Logic.GetLevelUpBacklog() > 0);
	const float TimeBefore = Logic.GetElapsedTime();
	const int32 StepsBefore = Logic.GetEpisodeStepCount();
	const TArray<float> ObsBefore = Logic.GetObservation();

	Logic.PhysicsStep(2);

	CHECK(Logic.GetElapsedTime() == TimeBefore);
	CHECK(Logic.GetEpisodeStepCount() == StepsBefore);
	CHECK(Logic.GetReward() == 0.f);
	CHECK(Logic.GetObservation() == ObsBefore);

	const FSurvivorsPendingLevelUpDecision Pending =
		Logic.GetPendingLevelUpDecision();
	REQUIRE(!Pending.Choices.IsEmpty());
	const FString ChoiceId = Pending.Choices[0].ChoiceId;
	const FSurvivorsLevelUpApplyResult Applied =
		Logic.ApplyExternalLevelUpChoice(Pending.DecisionId, ChoiceId);
	CHECK(Applied.Status == ESurvivorsLevelUpApplyStatus::Applied);
	CHECK(Logic.GetPlayerLevel() == 3);
	CHECK(Logic.IsLevelUpPending());

	const FSurvivorsLevelUpApplyResult Duplicate =
		Logic.ApplyExternalLevelUpChoice(Pending.DecisionId, ChoiceId);
	CHECK(Duplicate.Status == ESurvivorsLevelUpApplyStatus::Applied);
	CHECK(Duplicate.PostChoiceObs == Applied.PostChoiceObs);
	CHECK(Duplicate.PendingAfter.DecisionId == Applied.PendingAfter.DecisionId);

	const FSurvivorsLevelUpApplyResult Stale =
		Logic.ApplyExternalLevelUpChoice(Pending.DecisionId, TEXT("choice-999"));
	CHECK(Stale.Status == ESurvivorsLevelUpApplyStatus::StaleDecision);
	CHECK(Logic.GetPlayerLevel() == 3);
}
