#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Survivors/Logic/SurvivorsTypes.h"
#include "Survivors/Logic/SurvivorsGameConstants.h"
#include "SurvivorsPlayerComponent.generated.h"

class ASurvivorsGame;

UCLASS(ClassGroup=(Custom), meta=(BlueprintSpawnableComponent))
class REINBALANCE_API USurvivorsPlayerComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	USurvivorsPlayerComponent();

	void Initialize(ASurvivorsGame* InGame);
	void Reset();
	void ApplyAction(int32 ActionIdx);
	float XPRequiredForLevel(int32 Level) const;
	float CumulativeXPForLevel(int32 Level) const;
	void ProcessXPGain(float Amount);
	void OnLevelUp(int32 NextLevel);

	/** パッシブスロットの変化後に呼ぶ。CachedPassiveEffects / MaxPlayerHP / GemPickupRadius を更新する */
	void RecalcPassiveEffects();

	/** BuildLevelUpChoices: レベルアップ時の選択肢を生成する（3択）*/
	TArray<FLevelUpChoice> BuildLevelUpChoices();

	/** 進化可能な武器スロットを返す */
	TArray<int32> GetEvolvableWeapons() const;

	/** 武器を進化させる */
	void EvolveWeapon(int32 SlotIdx, EWeaponType EvolvedType);

	/** レベルアップ選択肢を適用する（武器・パッシブ兼用） */
	void ApplyLevelUpChoice(const FLevelUpChoice& Choice);

private:
	UPROPERTY()
	TObjectPtr<ASurvivorsGame> Game;

	FPassiveEffects ComputePassiveEffects() const;
};
