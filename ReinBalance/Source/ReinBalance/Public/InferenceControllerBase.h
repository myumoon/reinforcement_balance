#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "NNEModelData.h"
#include "NNERuntimeCPU.h"
#include "InferenceControllerBase.generated.h"

/**
 * NNERuntimeORT で ONNX モデルをロードしてゲームを自律制御する基底アクター。
 *
 * 派生クラスの実装方針:
 *   1. GetObsDim() をオーバーライドしてモデルの入力次元を返す
 *   2. BeginPlay() で Super::BeginPlay() を呼ぶ（モデルロード実行）
 *   3. Tick() で GetObservation → RunInference → ApplyAction を実装する
 */
UCLASS(Abstract)
class REINBALANCE_API AInferenceControllerBase : public AActor
{
	GENERATED_BODY()

public:
	AInferenceControllerBase();

	/** ONNX モデルデータアセット (Content Browser からインポートした UNNEModelData) */
	UPROPERTY(EditAnywhere, Category = "Inference")
	TObjectPtr<UNNEModelData> ModelData;

	/** エピソード終了時に自動でリセットするか */
	UPROPERTY(EditAnywhere, Category = "Inference")
	bool bAutoReset = true;

protected:
	/** BeginPlay でモデルをロードする。派生クラスは Super::BeginPlay() を呼ぶこと。 */
	virtual void BeginPlay() override;

	/** モデルの観測次元（派生クラスでオーバーライド必須） */
	virtual uint32 GetObsDim() const
	{
		checkNoEntry(); // 派生クラスで必ずオーバーライドすること
		return 0;
	}

	/** モデルの出力次元（デフォルト 1） */
	virtual uint32 GetActionDim() const { return 1; }

	/**
	 * NNE 推論を実行する。
	 * @param Obs      入力観測ベクトル（GetObsDim() 次元）
	 * @param OutAction 出力アクションベクトル（GetActionDim() 次元）
	 * @return 推論成功なら true
	 */
	bool RunInference(const TArray<float>& Obs, TArray<float>& OutAction);

	bool bModelReady = false;

private:
	TSharedPtr<UE::NNE::IModelCPU>         NNEModel;
	TSharedPtr<UE::NNE::IModelInstanceCPU> NNEModelInstance;
	TArray<float>                           ActionBuffer;
};
