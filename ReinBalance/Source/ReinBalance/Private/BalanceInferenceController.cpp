#include "BalanceInferenceController.h"

void ABalanceInferenceController::BeginPlay()
{
	if (!Cart)
	{
		UE_LOG(LogTemp, Warning, TEXT("[BalanceInference] Cart が未設定です。Details パネルで ABalanceCart を設定してください。"));
		return;
	}

	Super::BeginPlay(); // NNE モデルロード

	if (bModelReady)
	{
		Cart->ResetState(TOptional<int32>());
		UE_LOG(LogTemp, Log, TEXT("[BalanceInference] PIE 中は毎フレーム ONNX 推論で Cart を制御します。"));
	}
}

void ABalanceInferenceController::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
	if (!bModelReady) return;

	if (Cart->IsDone())
	{
		if (bAutoReset) Cart->ResetState(TOptional<int32>());
		return;
	}

	TArray<float> Obs = Cart->GetObservation();
	TArray<float> Action;
	if (RunInference(Obs, Action) && Action.Num() > 0)
	{
		Cart->PhysicsStep(FMath::Clamp(Action[0], -1.f, 1.f));
	}
}
