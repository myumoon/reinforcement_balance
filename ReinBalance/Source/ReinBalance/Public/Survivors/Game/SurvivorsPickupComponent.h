#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Survivors/Game/SurvivorsTypes.h"
#include "SurvivorsPickupComponent.generated.h"

class ASurvivorsGame;

/**
 * フロアアイテム・特殊アイテム収集を処理するコンポーネント。
 * PhysicsStep から CheckFloorPickups() / CheckSpecialPickups() を呼ぶ。
 */
UCLASS(ClassGroup=(Custom), meta=(BlueprintSpawnableComponent))
class REINBALANCE_API USurvivorsPickupComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	USurvivorsPickupComponent();

	void Initialize(ASurvivorsGame* InGame);

	/** フロアアイテム（FloorChicken / LittleHeart）を収集してHP回復 */
	void CheckFloorPickups();

	/** 特殊アイテム（Rosary / Vacuum / Orologion）の効果を発動 */
	void CheckSpecialPickups();

private:
	UPROPERTY()
	TObjectPtr<ASurvivorsGame> Game;
};
