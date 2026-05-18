#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "TrainingRenderToggleActor.generated.h"

/**
 * 訓練中にキーボードでレンダリングを on/off するユーティリティ Actor。
 * 訓練用レベルに配置するだけで有効になる。
 * デフォルトキー: F10（Details パネルで変更可）
 *
 * 仕組み: Tick ごとにキー状態をポーリングし、立ち上がりエッジで
 * GEngine->GameViewport->bDisableWorldRendering をトグルする。
 * InputComponent / AutoReceiveInput は使用しないため Enhanced Input と競合しない。
 */
UCLASS()
class REINBALANCEEDITOR_API ATrainingRenderToggleActor : public AActor
{
	GENERATED_BODY()

public:
	ATrainingRenderToggleActor();

	/** レンダリングをトグルするキー */
	UPROPERTY(EditAnywhere, Category = "Training")
	FKey ToggleKey;

protected:
	virtual void Tick(float DeltaTime) override;

private:
	void ToggleRendering();

	bool bRenderEnabled = true;
	bool bKeyWasDown    = false;
};
