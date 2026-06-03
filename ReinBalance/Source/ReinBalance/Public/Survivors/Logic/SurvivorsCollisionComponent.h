#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "SurvivorsCollisionComponent.generated.h"

class ASurvivorsGame;

UCLASS(ClassGroup=(Custom), meta=(BlueprintSpawnableComponent))
class REINBALANCE_API USurvivorsCollisionComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	USurvivorsCollisionComponent();

	void Initialize(ASurvivorsGame* InGame);
	void CollectWallActors();
	void ResolveWallCollisions();
	float CastRayToObstacles(FVector2D Origin, FVector2D Dir) const;

	/** プロジェクタイルの壁反射（Runetracer 用）。
	 *  壁に当たった場合は InOutPos/InOutVel を修正して true を返す。 */
	bool ReflectOffWall(FVector2D& InOutPos, FVector2D& InOutVel, float Radius) const;

private:
	UPROPERTY()
	TObjectPtr<ASurvivorsGame> Game;
};
