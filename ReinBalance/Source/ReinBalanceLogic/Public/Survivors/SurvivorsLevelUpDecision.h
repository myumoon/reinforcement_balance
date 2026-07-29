#pragma once

/**
 * 外部レベルアップ選択の保留状態と ID 検証を定義する。
 * 初心者向け: ゲーム本体から分離し、同じ選択の再送と古い選択を安全に見分ける。
 */
#include "CoreMinimal.h"
#include "Survivors/SurvivorsTypes.h"

/** HTTP に公開する一つの順序付き候補。 */
struct FSurvivorsLevelUpChoiceOffer
{
	FString ChoiceId;
	FLevelUpChoice Choice;
};

/**
 * 一つのchoiceをproduction適用した直後の反実仮想観測。
 * 初心者向け: Errorが空でない場合はclone不能またはID不一致なので、観測値を使用してはいけない。
 */
struct FSurvivorsChoicePreview
{
	FString ChoiceId;
	TArray<float> ProjectedObservation;
	TArray<FString> ChangedSegments;
	FString Error;

	bool IsValid() const
	{
		return Error.IsEmpty() && !ChoiceId.IsEmpty()
			&& !ProjectedObservation.IsEmpty();
	}
};

/** 現在保留中のレベルアップ選択。 */
struct FSurvivorsPendingLevelUpDecision
{
	FString DecisionId;
	int32 PlayerLevel = 0;
	TArray<FSurvivorsLevelUpChoiceOffer> Choices;

	bool IsSet() const { return !DecisionId.IsEmpty(); }
	void Reset()
	{
		DecisionId.Empty();
		PlayerLevel = 0;
		Choices.Reset();
	}
};

/** mutation 前の fail-closed 検証結果。 */
enum class ESurvivorsLevelUpChoiceValidation : uint8
{
	Accepted,
	Duplicate,
	StaleDecision,
	InvalidChoice,
};

/**
 * pending と直前に適用済みの ID を管理する純粋な状態機械。
 * 初心者向け: 乱数を使わない連番 ID により、HTTP 再送でもアイテムを二重適用しない。
 */
class REINBALANCELOGIC_API FSurvivorsLevelUpDecisionState
{
public:
	void Reset(uint64 InEpisodeSerial);
	bool BeginDecision(int32 PlayerLevel, const TArray<FLevelUpChoice>& Choices);
	ESurvivorsLevelUpChoiceValidation ValidateChoice(
		const FString& DecisionId, const FString& ChoiceId, int32& OutChoiceIndex) const;
	void CommitChoice(const FString& DecisionId, const FString& ChoiceId);

	bool IsPending() const { return Pending.IsSet(); }
	const FSurvivorsPendingLevelUpDecision& GetPending() const { return Pending; }
	const FString& GetLastAppliedDecisionId() const { return LastAppliedDecisionId; }
	const FString& GetLastAppliedChoiceId() const { return LastAppliedChoiceId; }

private:
	uint64 EpisodeSerial = 0;
	uint64 DecisionSequence = 0;
	FSurvivorsPendingLevelUpDecision Pending;
	FString LastAppliedDecisionId;
	FString LastAppliedChoiceId;
};

/** choice type を wire 上の固定文字列へ変換する。 */
REINBALANCELOGIC_API FString SurvivorsLevelUpChoiceTypeToString(
	FLevelUpChoice::EChoiceType ChoiceType);
