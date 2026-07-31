/**
 * Survivors choice previewのproduction parityと不変性を検証するLLT。
 * 初心者向け: 同じprefixから作った別Logicへ実適用し、preview sandboxのraw observationと比較する。
 */
#include "TestHarness.h"

#include "Math/UnrealMathUtility.h"
#include "Survivors/SurvivorsGameConstants.h"
#include "Survivors/SurvivorsGameLogic.h"

/**
 * preview検証だけに非公開choice setupとidempotency stateを公開するfriend adapter。
 * 初心者向け: production methodを別実装せず、テストscenarioの準備と読取だけを許可する。
 */
struct FSurvivorsChoiceProjectionTestAccess
{
	static void ApplySetupChoice(
		FSurvivorsGameLogic& Logic,
		const FLevelUpChoice& Choice)
	{
		Logic.ApplyLevelUpChoice(Choice);
		Logic.RecalcPassiveEffects();
	}

	static bool BeginDecision(
		FSurvivorsGameLogic& Logic,
		const FLevelUpChoice& Choice)
	{
		return Logic.LevelUpDecisionState.BeginDecision(
			Logic.PlayerLevel, {Choice});
	}

	static const FString& LastAppliedDecisionId(
		const FSurvivorsGameLogic& Logic)
	{
		return Logic.LevelUpDecisionState.GetLastAppliedDecisionId();
	}

	static const FString& LastAppliedChoiceId(
		const FSurvivorsGameLogic& Logic)
	{
		return Logic.LevelUpDecisionState.GetLastAppliedChoiceId();
	}

	static int32 CachedObservationDimension(
		const FSurvivorsGameLogic& Logic)
	{
		return Logic.CachedObsDim;
	}

	static const TOptional<FSurvivorsLevelUpApplyResult>&
	LastAppliedResult(const FSurvivorsGameLogic& Logic)
	{
		return Logic.LastAppliedLevelUpResult;
	}
};

namespace
{
enum class EPreviewScenario : uint8
{
	WeaponNew,
	WeaponUpgrade,
	WeaponEvolve,
	WeaponUnion,
	PassiveNew,
	PassiveUpgrade,
};

/**
 * choice type・weapon・passiveを明示したtest choiceを作る。
 * 初心者向け: setup値だけを組み立て、適用自体は必ずproduction Logicへ渡す。
 */
FLevelUpChoice MakeChoice(
	FLevelUpChoice::EChoiceType ChoiceType,
	EWeaponType WeaponType = EWeaponType::None,
	EPassiveItemType PassiveType = EPassiveItemType::None,
	int32 NewLevel = 1)
{
	FLevelUpChoice Choice;
	Choice.ChoiceType = ChoiceType;
	Choice.WeaponType = WeaponType;
	Choice.PassiveType = PassiveType;
	Choice.NewLevel = NewLevel;
	return Choice;
}

/**
 * RNGとconfigを固定した独立Logic instanceを初期化する。
 * 初心者向け: preview側とoracle側が同じepisode prefixから始まるようseedまで揃える。
 */
void InitializeExternalLogic(FSurvivorsGameLogic& Logic)
{
	FSurvivorsGameLogicConfig Config;
	Config.ItemSelectionMode = TEXT("external");
	Config.StartingWeaponMode = TEXT("garlic");
	Config.MinActiveEnemies = 0;
	Config.MaxActiveEnemies = 0;
	Logic.Initialize(Config);
	Logic.Reset(314159);
}

/**
 * new/upgrade/evolve/union/passive各経路の同一stateと対象choiceを準備する。
 * 初心者向け: unionもWeaponEvolveのproduction分岐へVandalierを渡し、partner消費まで検証する。
 */
FLevelUpChoice PrepareScenario(
	FSurvivorsGameLogic& Logic,
	EPreviewScenario Scenario)
{
	switch (Scenario)
	{
	case EPreviewScenario::WeaponNew:
		return MakeChoice(
			FLevelUpChoice::EChoiceType::WeaponNew,
			EWeaponType::Knife);
	case EPreviewScenario::WeaponUpgrade:
		return MakeChoice(
			FLevelUpChoice::EChoiceType::WeaponUpgrade,
			EWeaponType::Garlic,
			EPassiveItemType::None,
			2);
	case EPreviewScenario::WeaponEvolve:
		FSurvivorsChoiceProjectionTestAccess::ApplySetupChoice(
			Logic,
			MakeChoice(
				FLevelUpChoice::EChoiceType::WeaponUpgrade,
				EWeaponType::Garlic,
				EPassiveItemType::None,
				SurvivorsGameConstants::GetWeaponMaxLevel(
					EWeaponType::Garlic)));
		FSurvivorsChoiceProjectionTestAccess::ApplySetupChoice(
			Logic,
			MakeChoice(
				FLevelUpChoice::EChoiceType::PassiveNew,
				EWeaponType::None,
				EPassiveItemType::Pummarola));
		return MakeChoice(
			FLevelUpChoice::EChoiceType::WeaponEvolve,
			EWeaponType::SoulEater);
	case EPreviewScenario::WeaponUnion:
		for (const EWeaponType Type :
			{EWeaponType::Peachone, EWeaponType::EbonyWings})
		{
			FSurvivorsChoiceProjectionTestAccess::ApplySetupChoice(
				Logic,
				MakeChoice(
					FLevelUpChoice::EChoiceType::WeaponNew,
					Type));
			FSurvivorsChoiceProjectionTestAccess::ApplySetupChoice(
				Logic,
				MakeChoice(
					FLevelUpChoice::EChoiceType::WeaponUpgrade,
					Type,
					EPassiveItemType::None,
					SurvivorsGameConstants::GetWeaponMaxLevel(Type)));
		}
		return MakeChoice(
			FLevelUpChoice::EChoiceType::WeaponEvolve,
			EWeaponType::Vandalier);
	case EPreviewScenario::PassiveNew:
		return MakeChoice(
			FLevelUpChoice::EChoiceType::PassiveNew,
			EWeaponType::None,
			EPassiveItemType::Wings);
	case EPreviewScenario::PassiveUpgrade:
		FSurvivorsChoiceProjectionTestAccess::ApplySetupChoice(
			Logic,
			MakeChoice(
				FLevelUpChoice::EChoiceType::PassiveNew,
				EWeaponType::None,
				EPassiveItemType::Wings));
		return MakeChoice(
			FLevelUpChoice::EChoiceType::PassiveUpgrade,
			EWeaponType::None,
			EPassiveItemType::Wings,
			2);
	default:
		FAIL("unknown preview scenario");
		return FLevelUpChoice();
	}
}

/**
 * 二つのraw observationが指定誤差内で全要素一致することを検証する。
 * 初心者向け: array全体を比較し、特定slotだけ合う不完全なprojectionを見逃さない。
 */
void CheckObservationParity(
	const TArray<float>& Preview,
	const TArray<float>& Actual)
{
	REQUIRE(Preview.Num() == Actual.Num());
	for (int32 Index = 0; Index < Preview.Num(); ++Index)
	{
		CAPTURE(Index, Preview[Index], Actual[Index]);
		CHECK(FMath::Abs(Preview[Index] - Actual[Index]) <= 1.e-6f);
	}
}

/**
 * observable state・RNG・pending/idempotencyを安定した文字列へ直列化する。
 * 初心者向け: preview前後の値を一括比較し、観測に出ない再送用IDのmutationも検出する。
 */
FString SerializePreviewInvariantState(const FSurvivorsGameLogic& Logic)
{
	FString Serialized = Logic.GetObsSchemaHash();
	for (const float Value : Logic.GetObservation())
	{
		Serialized += TEXT("|");
		Serialized += FString::Printf(TEXT("%.9g"), Value);
	}
	Serialized += FString::Printf(
		TEXT("|rng_initial=%d|rng_current=%d|backlog=%d|projectiles=%d|zones=%d"),
		Logic.RandStream.GetInitialSeed(),
		Logic.RandStream.GetCurrentSeed(),
		Logic.GetLevelUpBacklog(),
		Logic.Projectiles.Num(),
		Logic.GroundZones.Num());
	const FSurvivorsPendingLevelUpDecision& Pending =
		Logic.GetPendingLevelUpDecision();
	Serialized += TEXT("|pending=");
	Serialized += Pending.DecisionId;
	Serialized += FString::Printf(TEXT("|pending_level=%d"), Pending.PlayerLevel);
	for (const FSurvivorsLevelUpChoiceOffer& Offer : Pending.Choices)
	{
		Serialized += TEXT("|choice=");
		Serialized += Offer.ChoiceId;
		Serialized += FString::Printf(
			TEXT(":%d:%d:%d:%d:%d"),
			static_cast<int32>(Offer.Choice.ChoiceType),
			static_cast<int32>(Offer.Choice.WeaponType),
			static_cast<int32>(Offer.Choice.PassiveType),
			Offer.Choice.SlotIdx,
			Offer.Choice.NewLevel);
	}
	Serialized += TEXT("|last_decision=");
	Serialized +=
		FSurvivorsChoiceProjectionTestAccess::LastAppliedDecisionId(Logic);
	Serialized += TEXT("|last_choice=");
	Serialized +=
		FSurvivorsChoiceProjectionTestAccess::LastAppliedChoiceId(Logic);
	const TOptional<FSurvivorsLevelUpApplyResult>& LastResult =
		FSurvivorsChoiceProjectionTestAccess::LastAppliedResult(Logic);
	Serialized += LastResult.IsSet()
		? TEXT("|last_result=set")
		: TEXT("|last_result=unset");
	if (LastResult.IsSet())
	{
		const FSurvivorsLevelUpApplyResult& Result = LastResult.GetValue();
		Serialized += FString::Printf(
			TEXT(":%d:%d"),
			static_cast<int32>(Result.Status),
			Result.BacklogAfter);
		Serialized += TEXT(":");
		Serialized += Result.DecisionId;
		Serialized += TEXT(":");
		Serialized += Result.ChoiceId;
		for (const float Value : Result.PostChoiceObs)
		{
			Serialized += TEXT(":obs=");
			Serialized += FString::Printf(TEXT("%.9g"), Value);
		}
		Serialized += TEXT(":pending=");
		Serialized += Result.PendingAfter.DecisionId;
		Serialized += FString::Printf(
			TEXT(":%d"), Result.PendingAfter.PlayerLevel);
		for (const FSurvivorsLevelUpChoiceOffer& Offer :
			Result.PendingAfter.Choices)
		{
			Serialized += TEXT(":offer=");
			Serialized += Offer.ChoiceId;
			Serialized += FString::Printf(
				TEXT(":%d:%d:%d:%d:%d"),
				static_cast<int32>(Offer.Choice.ChoiceType),
				static_cast<int32>(Offer.Choice.WeaponType),
				static_cast<int32>(Offer.Choice.PassiveType),
				Offer.Choice.SlotIdx,
				Offer.Choice.NewLevel);
		}
	}
	return Serialized;
}

/**
 * cooldown・projectile・ground zoneを含む同一attack runtimeを投入する。
 * 初心者向け: slotだけでなく発射済みentityと武器instance内部timerのdeep cloneもparity対象にする。
 */
void SeedAttackRuntimeState(FSurvivorsGameLogic& Logic)
{
	FSurvivorsChoiceProjectionTestAccess::ApplySetupChoice(
		Logic,
		MakeChoice(
			FLevelUpChoice::EChoiceType::WeaponNew,
			EWeaponType::FireWand));
	for (int32 Step = 0; Step < 20; ++Step)
	{
		Logic.PhysicsStep(2);
	}

	FProjectileState Projectile;
	Projectile.Pos = FVector2D(10.f, 20.f);
	Projectile.Vel = FVector2D(3.f, 4.f);
	Projectile.Radius = FSimRadius(5.f);
	Projectile.WeaponType = EWeaponType::Knife;
	Projectile.WeaponSlotIdx = 0;
	Projectile.LifeTime = FProjectileLifeTime(1.f);
	Logic.Projectiles.Add(Projectile);
	FGroundZoneState Zone;
	Zone.Pos = FVector2D(-10.f, 30.f);
	Zone.WeaponType = EWeaponType::FireWand;
	Zone.WeaponSlotIdx = 1;
	Logic.GroundZones.Add(Zone);
}
}

TEST_CASE(
	"Survivors preview matches actual apply for every choice kind",
	"[unit][survivors][level-up][preview][parity]")
{
	const EPreviewScenario Scenarios[] = {
		EPreviewScenario::WeaponNew,
		EPreviewScenario::WeaponUpgrade,
		EPreviewScenario::WeaponEvolve,
		EPreviewScenario::WeaponUnion,
		EPreviewScenario::PassiveNew,
		EPreviewScenario::PassiveUpgrade,
	};
	for (const EPreviewScenario Scenario : Scenarios)
	{
		CAPTURE(static_cast<int32>(Scenario));
		FSurvivorsGameLogic PreviewLogic;
		FSurvivorsGameLogic OracleLogic;
		InitializeExternalLogic(PreviewLogic);
		InitializeExternalLogic(OracleLogic);
		const FLevelUpChoice PreviewChoice =
			PrepareScenario(PreviewLogic, Scenario);
		const FLevelUpChoice OracleChoice =
			PrepareScenario(OracleLogic, Scenario);
		REQUIRE(
			static_cast<int32>(PreviewChoice.ChoiceType)
			== static_cast<int32>(OracleChoice.ChoiceType));
		REQUIRE(
			FSurvivorsChoiceProjectionTestAccess::BeginDecision(
				PreviewLogic, PreviewChoice));
		REQUIRE(
			FSurvivorsChoiceProjectionTestAccess::BeginDecision(
				OracleLogic, OracleChoice));
		const FSurvivorsPendingLevelUpDecision Pending =
			PreviewLogic.GetPendingLevelUpDecision();
		REQUIRE(Pending.Choices.Num() == 1);

		const FSurvivorsChoicePreview Preview =
			PreviewLogic.PreviewLevelUpChoice(
				Pending.DecisionId, Pending.Choices[0].ChoiceId);
		const FSurvivorsLevelUpApplyResult Actual =
			OracleLogic.ApplyExternalLevelUpChoice(
				Pending.DecisionId, Pending.Choices[0].ChoiceId);

		REQUIRE(Preview.IsValid());
		REQUIRE(
			Actual.Status == ESurvivorsLevelUpApplyStatus::Applied);
		CheckObservationParity(
			Preview.ProjectedObservation, Actual.PostChoiceObs);
	}
}

TEST_CASE(
	"Survivors preview const path does not populate the source observation cache",
	"[unit][survivors][level-up][preview][const]")
{
	FSurvivorsGameLogic Logic;
	InitializeExternalLogic(Logic);
	const FLevelUpChoice Choice = MakeChoice(
		FLevelUpChoice::EChoiceType::PassiveNew,
		EWeaponType::None,
		EPassiveItemType::Wings);
	REQUIRE(
		FSurvivorsChoiceProjectionTestAccess::BeginDecision(Logic, Choice));
	const FSurvivorsPendingLevelUpDecision Pending =
		Logic.GetPendingLevelUpDecision();
	const int32 CachedDimensionBefore =
		FSurvivorsChoiceProjectionTestAccess::CachedObservationDimension(Logic);

	const FSurvivorsChoicePreview Preview =
		Logic.PreviewLevelUpChoice(
			Pending.DecisionId, Pending.Choices[0].ChoiceId);

	REQUIRE(Preview.IsValid());
	CHECK(
		FSurvivorsChoiceProjectionTestAccess::CachedObservationDimension(Logic)
		== CachedDimensionBefore);
}

TEST_CASE(
	"Survivors preview preserves passive stat boundaries",
	"[unit][survivors][level-up][preview][passive]")
{
	const EPassiveItemType BoundaryPassives[] = {
		EPassiveItemType::HollowHeart,
		EPassiveItemType::Wings,
		EPassiveItemType::Attractorb,
		EPassiveItemType::Tirajisu,
	};
	for (const EPassiveItemType PassiveType : BoundaryPassives)
	{
		CAPTURE(static_cast<int32>(PassiveType));
		FSurvivorsGameLogic PreviewLogic;
		FSurvivorsGameLogic OracleLogic;
		InitializeExternalLogic(PreviewLogic);
		InitializeExternalLogic(OracleLogic);
		const FLevelUpChoice Choice = MakeChoice(
			FLevelUpChoice::EChoiceType::PassiveNew,
			EWeaponType::None,
			PassiveType);
		REQUIRE(
			FSurvivorsChoiceProjectionTestAccess::BeginDecision(
				PreviewLogic, Choice));
		REQUIRE(
			FSurvivorsChoiceProjectionTestAccess::BeginDecision(
				OracleLogic, Choice));
		const FSurvivorsPendingLevelUpDecision Pending =
			PreviewLogic.GetPendingLevelUpDecision();
		const FSurvivorsChoicePreview Preview =
			PreviewLogic.PreviewLevelUpChoice(
				Pending.DecisionId, Pending.Choices[0].ChoiceId);
		const FSurvivorsLevelUpApplyResult Actual =
			OracleLogic.ApplyExternalLevelUpChoice(
				Pending.DecisionId, Pending.Choices[0].ChoiceId);

		REQUIRE(Preview.IsValid());
		REQUIRE(
			Actual.Status == ESurvivorsLevelUpApplyStatus::Applied);
		CheckObservationParity(
			Preview.ProjectedObservation, Actual.PostChoiceObs);
		CHECK(Preview.ChangedSegments.Contains(TEXT("passive_slots")));
	}
}

TEST_CASE(
	"Survivors preview preserves weapon cooldown projectile and ground zone parity",
	"[unit][survivors][level-up][preview][weapon-runtime]")
{
	FSurvivorsGameLogic PreviewLogic;
	FSurvivorsGameLogic OracleLogic;
	InitializeExternalLogic(PreviewLogic);
	InitializeExternalLogic(OracleLogic);
	SeedAttackRuntimeState(PreviewLogic);
	SeedAttackRuntimeState(OracleLogic);
	REQUIRE(PreviewLogic.Projectiles.Num() > 0);
	REQUIRE(PreviewLogic.GroundZones.Num() > 0);
	REQUIRE(PreviewLogic.Weapons.IsValidIndex(1));
	REQUIRE(PreviewLogic.Weapons[1].Get() != nullptr);
	REQUIRE(
		PreviewLogic.Weapons[1]->GetCooldownRemaining().Value > 0.f);

	const FLevelUpChoice Choice = MakeChoice(
		FLevelUpChoice::EChoiceType::PassiveNew,
		EWeaponType::None,
		EPassiveItemType::EmptyTome);
	REQUIRE(
		FSurvivorsChoiceProjectionTestAccess::BeginDecision(
			PreviewLogic, Choice));
	REQUIRE(
		FSurvivorsChoiceProjectionTestAccess::BeginDecision(
			OracleLogic, Choice));
	const FSurvivorsPendingLevelUpDecision Pending =
		PreviewLogic.GetPendingLevelUpDecision();
	const FSurvivorsChoicePreview Preview =
		PreviewLogic.PreviewLevelUpChoice(
			Pending.DecisionId, Pending.Choices[0].ChoiceId);
	const FSurvivorsLevelUpApplyResult Actual =
		OracleLogic.ApplyExternalLevelUpChoice(
			Pending.DecisionId, Pending.Choices[0].ChoiceId);

	REQUIRE(Preview.IsValid());
	REQUIRE(Actual.Status == ESurvivorsLevelUpApplyStatus::Applied);
	CheckObservationParity(
		Preview.ProjectedObservation, Actual.PostChoiceObs);
	CHECK(Preview.ChangedSegments.Contains(TEXT("weapon_slots")));
	CHECK(Preview.ChangedSegments.Contains(TEXT("passive_slots")));
}

TEST_CASE(
	"Survivors preview one hundred calls leave observable and hidden state unchanged",
	"[unit][survivors][level-up][preview][immutable]")
{
	FSurvivorsGameLogic Logic;
	InitializeExternalLogic(Logic);
	const FLevelUpChoice PriorChoice = MakeChoice(
		FLevelUpChoice::EChoiceType::PassiveNew,
		EWeaponType::None,
		EPassiveItemType::Armor);
	REQUIRE(
		FSurvivorsChoiceProjectionTestAccess::BeginDecision(
			Logic, PriorChoice));
	const FSurvivorsPendingLevelUpDecision PriorPending =
		Logic.GetPendingLevelUpDecision();
	REQUIRE(
		Logic.ApplyExternalLevelUpChoice(
			PriorPending.DecisionId,
			PriorPending.Choices[0].ChoiceId).Status
		== ESurvivorsLevelUpApplyStatus::Applied);
	SeedAttackRuntimeState(Logic);

	const FLevelUpChoice Choice = MakeChoice(
		FLevelUpChoice::EChoiceType::PassiveNew,
		EWeaponType::None,
		EPassiveItemType::EmptyTome);
	REQUIRE(
		FSurvivorsChoiceProjectionTestAccess::BeginDecision(Logic, Choice));
	const FSurvivorsPendingLevelUpDecision Pending =
		Logic.GetPendingLevelUpDecision();
	const FString Before = SerializePreviewInvariantState(Logic);

	for (int32 Iteration = 0; Iteration < 100; ++Iteration)
	{
		CAPTURE(Iteration);
		const FSurvivorsChoicePreview Preview =
			Logic.PreviewLevelUpChoice(
				Pending.DecisionId, Pending.Choices[0].ChoiceId);
		REQUIRE(Preview.IsValid());
	}

	CHECK(SerializePreviewInvariantState(Logic) == Before);
}
