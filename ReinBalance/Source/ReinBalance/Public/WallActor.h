#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Components/StaticMeshComponent.h"
#include "WallActor.generated.h"

/**
 * エディタで配置する壁アクター。
 * StaticMesh (ビジュアル) と BlockAll collision が一体になっている。
 * ASurvivorsGame が BeginPlay でレベル内の全インスタンスを収集し、
 * 衝突解決・レイキャスト観測に使用する。
 */
UCLASS()
class REINBALANCE_API AWallActor : public AActor
{
	GENERATED_BODY()

public:
	AWallActor();

	/** ルートコンポーネント兼ビジュアル。BlockAll collision 設定済み。 */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Components")
	TObjectPtr<UStaticMeshComponent> WallMesh;

	/**
	 * このアクターの bounds をシム座標 (m) の FBox2D で返す。
	 * @param InSimToUE  1 シムメートルあたりの UE ユニット数
	 */
	FBox2D GetSimBounds(float InSimToUE) const;
};
