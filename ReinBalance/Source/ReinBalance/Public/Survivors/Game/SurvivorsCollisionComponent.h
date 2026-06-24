#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Survivors/Logic/SurvivorsTypes.h"
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

	// ---- LocalUniformGrid API ----

	UPROPERTY(EditAnywhere, Category="Survivors|Collision")
	float CollisionHalfExtent = 1000.f;

	UPROPERTY(EditAnywhere, Category="Survivors|Collision")
	float CollisionCellSize = 128.f;

	UPROPERTY(EditAnywhere, Category="Survivors|Collision")
	bool bUseFullFieldCollision = false;

	/** EnemyGrid を構築する（BeginCollisionFrame + RegisterEnemyTargets + Rebuild） */
	void BuildEnemyGrid();

	/** PickupGrid を構築する（Clear + RegisterPickupTargets + Rebuild） */
	void BuildPickupGrid();

	/** EnemyGrid に対して接触候補を問い合わせる */
	void QueryEnemyContacts(FVector2D Pos, float Radius, TArray<FSurvivorsTargetProxy const*>& Out) const;

	/** PickupGrid に対して接触候補を問い合わせる */
	void QueryPickupContacts(FVector2D Pos, float Radius, TArray<FSurvivorsTargetProxy const*>& Out) const;

private:
	UPROPERTY()
	TObjectPtr<ASurvivorsGame> Game;

	FSurvivorsTargetGrid EnemyGrid;
	FSurvivorsTargetGrid PickupGrid;

	float GetEffectiveHalfExtent() const;
	void RegisterEnemyTargets();
	void RegisterPickupTargets();
};
