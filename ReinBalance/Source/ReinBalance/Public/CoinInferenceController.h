#pragma once

#include "CoreMinimal.h"
#include "InferenceControllerBase.h"
#include "CoinGame.h"
#include "CoinInferenceController.generated.h"

/**
 * 訓練済み ONNX モデルを NNERuntimeORT でロードし、
 * ACoinGame を自律制御するアクター。
 *
 * 使い方:
 *   1. python export_onnx.py --game coin でモデルを .onnx にエクスポート
 *   2. UE5 Content Browser に .onnx をインポート
 *   3. レベルにこのアクターを配置し、ModelData・CoinGame を設定
 *   4. PIE (▶) を開始すると自動で推論・制御が始まる
 */
UCLASS()
class REINBALANCE_API ACoinInferenceController : public AInferenceControllerBase
{
	GENERATED_BODY()

public:
	/** 制御対象の ACoinGame アクター */
	UPROPERTY(EditAnywhere, Category = "Inference")
	TObjectPtr<ACoinGame> CoinGame;

protected:
	virtual uint32 GetObsDim() const override
	{
		// BeginPlay でCoinGame の null チェック済みの状態で呼ばれる
		return CoinGame ? static_cast<uint32>(CoinGame->GetObsDim()) : 116u;
	}
	virtual void BeginPlay() override;
	virtual void Tick(float DeltaTime) override;
};
