#pragma once

#include "CoreMinimal.h"
#include "InferenceControllerBase.h"
#include "BalanceCart.h"
#include "BalanceInferenceController.generated.h"

/**
 * 訓練済み ONNX モデルを NNERuntimeORT でロードし、
 * ABalanceCart を自律制御するアクター。
 *
 * 使い方:
 *   1. python export_onnx.py でモデルを .onnx にエクスポート
 *   2. UE5 Content Browser に .onnx をドラッグ&ドロップでインポート
 *   3. レベルにこのアクターを配置し、ModelData・Cart を設定
 *   4. PIE (▶) を開始すると自動で推論・制御が始まる
 */
UCLASS()
class REINBALANCE_API ABalanceInferenceController : public AInferenceControllerBase
{
	GENERATED_BODY()

public:
	/** 制御対象の ABalanceCart アクター */
	UPROPERTY(EditAnywhere, Category = "Inference")
	TObjectPtr<ABalanceCart> Cart;

protected:
	virtual uint32 GetObsDim() const override { return 6; }
	virtual void BeginPlay() override;
	virtual void Tick(float DeltaTime) override;
};
