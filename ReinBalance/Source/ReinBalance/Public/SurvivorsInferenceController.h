#pragma once

#include "CoreMinimal.h"
#include "InferenceControllerBase.h"
#include "SurvivorsGame.h"
#include "SurvivorsInferenceController.generated.h"

/**
 * 訓練済み ONNX モデルを NNERuntimeORT でロードし、
 * ASurvivorsGame を自律制御するアクター。
 *
 * 使い方:
 *   1. python export_onnx.py --game survivors でモデルを .onnx にエクスポート
 *   2. UE5 Content Browser に .onnx をインポート
 *   3. レベルにこのアクターを配置し、ModelData・SurvivorsGame を設定
 *   4. PIE (▶) を開始すると自動で推論・制御が始まる
 */
UCLASS()
class REINBALANCE_API ASurvivorsInferenceController : public AInferenceControllerBase
{
	GENERATED_BODY()

public:
	/** 制御対象の ASurvivorsGame アクター */
	UPROPERTY(EditAnywhere, Category = "Inference")
	TObjectPtr<ASurvivorsGame> SurvivorsGame;

protected:
	virtual uint32 GetObsDim() const override
	{
		return SurvivorsGame ? static_cast<uint32>(SurvivorsGame->GetObsDim()) : 183u;
	}
	virtual void BeginPlay() override;
	virtual void Tick(float DeltaTime) override;
};
