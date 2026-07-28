#pragma once

/**
 * legacy PlayerComponent の薄い facade を定義する。
 * 初心者向け: XP と進化は Game 経由で Logic へ渡し、Component 自身にはレベルアップ本体を持たせない。
 */
#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Survivors/SurvivorsTypes.h"
#include "Survivors/SurvivorsGameConstants.h"
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

	/** 進化可能な武器スロットを返す */
	TArray<int32> GetEvolvableWeapons() const;

	/** 武器を進化させる */
	void EvolveWeapon(int32 SlotIdx, EWeaponType EvolvedType);

private:
	UPROPERTY()
	TObjectPtr<ASurvivorsGame> Game;

};
