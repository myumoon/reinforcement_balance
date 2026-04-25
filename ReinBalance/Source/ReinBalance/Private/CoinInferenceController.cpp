#include "CoinInferenceController.h"

void ACoinInferenceController::BeginPlay()
{
	if (!CoinGame)
	{
		UE_LOG(LogTemp, Warning, TEXT("[CoinInference] CoinGame が未設定です。Details パネルで ACoinGame を設定してください。"));
		return;
	}

	Super::BeginPlay(); // NNE モデルロード

	if (bModelReady)
	{
		CoinGame->ResetState(TOptional<int32>());
		UE_LOG(LogTemp, Log, TEXT("[CoinInference] PIE 中は毎フレーム ONNX 推論で CoinGame を制御します。"));
	}
}

void ACoinInferenceController::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
	if (!bModelReady) return;

	if (CoinGame->IsDone())
	{
		if (bAutoReset) CoinGame->ResetState(TOptional<int32>());
		return;
	}

	TArray<float> Obs = CoinGame->GetObservation();
	TArray<float> Action;
	if (RunInference(Obs, Action) && Action.Num() > 0)
	{
		// PPO Discrete の出力は float にキャストされたアクションインデックス
		const int32 ActionIdx = FMath::Clamp(FMath::RoundToInt(Action[0]), 0, 4);
		CoinGame->PhysicsStep(ActionIdx);
	}
}
