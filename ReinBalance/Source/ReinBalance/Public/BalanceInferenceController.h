#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "NNEModelData.h"
#include "NNERuntimeCPU.h"
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
class REINBALANCE_API ABalanceInferenceController : public AActor
{
	GENERATED_BODY()

public:
	ABalanceInferenceController();

	/** ONNX モデルデータアセット (Content Browser からインポートした UNNEModelData) */
	UPROPERTY(EditAnywhere, Category = "Inference")
	TObjectPtr<UNNEModelData> ModelData;

	/** 制御対象の ABalanceCart アクター */
	UPROPERTY(EditAnywhere, Category = "Inference")
	TObjectPtr<ABalanceCart> Cart;

	/** エピソード終了時に自動でリセットするか */
	UPROPERTY(EditAnywhere, Category = "Inference")
	bool bAutoReset = true;

protected:
	virtual void BeginPlay() override;
	virtual void Tick(float DeltaTime) override;

private:
	TSharedPtr<UE::NNE::IModelCPU> NNEModel;
	TSharedPtr<UE::NNE::IModelInstanceCPU> NNEModelInstance;
	bool bModelReady = false;
};
