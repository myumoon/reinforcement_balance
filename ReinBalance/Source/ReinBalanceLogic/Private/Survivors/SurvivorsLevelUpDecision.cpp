/**
 * 外部レベルアップ選択の ID 生成と exactly-once 検証を実装する。
 * 初心者向け: 保留中の候補に一致した要求だけを通し、直前の同一要求は再送として扱う。
 */
#include "Survivors/SurvivorsLevelUpDecision.h"

void FSurvivorsLevelUpDecisionState::Reset(uint64 InEpisodeSerial)
{
	EpisodeSerial = InEpisodeSerial;
	DecisionSequence = 0;
	Pending.Reset();
	LastAppliedDecisionId.Empty();
	LastAppliedChoiceId.Empty();
}

bool FSurvivorsLevelUpDecisionState::BeginDecision(
	int32 PlayerLevel, const TArray<FLevelUpChoice>& Choices)
{
	if (Pending.IsSet() || Choices.IsEmpty())
	{
		return false;
	}

	++DecisionSequence;
	Pending.DecisionId = FString::Printf(
		TEXT("level-up-%llu-%llu-%d"),
		static_cast<unsigned long long>(EpisodeSerial),
		static_cast<unsigned long long>(DecisionSequence),
		PlayerLevel);
	Pending.PlayerLevel = PlayerLevel;
	Pending.Choices.Reserve(Choices.Num());
	for (int32 Index = 0; Index < Choices.Num(); ++Index)
	{
		FSurvivorsLevelUpChoiceOffer Offer;
		Offer.ChoiceId = FString::Printf(TEXT("choice-%d"), Index);
		Offer.Choice = Choices[Index];
		Pending.Choices.Add(MoveTemp(Offer));
	}
	return true;
}

ESurvivorsLevelUpChoiceValidation FSurvivorsLevelUpDecisionState::ValidateChoice(
	const FString& DecisionId, const FString& ChoiceId, int32& OutChoiceIndex) const
{
	OutChoiceIndex = INDEX_NONE;
	if (Pending.IsSet() && Pending.DecisionId == DecisionId)
	{
		for (int32 Index = 0; Index < Pending.Choices.Num(); ++Index)
		{
			if (Pending.Choices[Index].ChoiceId == ChoiceId)
			{
				OutChoiceIndex = Index;
				return ESurvivorsLevelUpChoiceValidation::Accepted;
			}
		}
		return ESurvivorsLevelUpChoiceValidation::InvalidChoice;
	}

	if (DecisionId == LastAppliedDecisionId && ChoiceId == LastAppliedChoiceId)
	{
		return ESurvivorsLevelUpChoiceValidation::Duplicate;
	}
	return ESurvivorsLevelUpChoiceValidation::StaleDecision;
}

void FSurvivorsLevelUpDecisionState::CommitChoice(
	const FString& DecisionId, const FString& ChoiceId)
{
	if (!Pending.IsSet() || Pending.DecisionId != DecisionId)
	{
		return;
	}
	LastAppliedDecisionId = DecisionId;
	LastAppliedChoiceId = ChoiceId;
	Pending.Reset();
}

FString SurvivorsLevelUpChoiceTypeToString(FLevelUpChoice::EChoiceType ChoiceType)
{
	switch (ChoiceType)
	{
	case FLevelUpChoice::EChoiceType::WeaponNew:
		return TEXT("weapon_new");
	case FLevelUpChoice::EChoiceType::WeaponUpgrade:
		return TEXT("weapon_upgrade");
	case FLevelUpChoice::EChoiceType::WeaponEvolve:
		return TEXT("weapon_evolve");
	case FLevelUpChoice::EChoiceType::PassiveNew:
		return TEXT("passive_new");
	case FLevelUpChoice::EChoiceType::PassiveUpgrade:
		return TEXT("passive_upgrade");
	case FLevelUpChoice::EChoiceType::NoUpgrade:
		return TEXT("no_upgrade");
	default:
		return TEXT("unknown");
	}
}
