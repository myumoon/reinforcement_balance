#include "SurvivorsInferenceController.h"

void ASurvivorsInferenceController::BeginPlay()
{
	if (!SurvivorsGame)
	{
		UE_LOG(LogTemp, Warning,
			TEXT("[SurvivorsInference] SurvivorsGame が未設定です。Details パネルで ASurvivorsGame を設定してください。"));
		return;
	}

	Super::BeginPlay(); // NNE モデルロード

	if (bModelReady)
	{
		SurvivorsGame->ResetState(TOptional<int32>());
		UE_LOG(LogTemp, Log,
			TEXT("[SurvivorsInference] PIE 中は毎フレーム ONNX 推論で SurvivorsGame を制御します。"));
	}
}

void ASurvivorsInferenceController::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
	if (!bModelReady || !SurvivorsGame) return;

	if (SurvivorsGame->IsDone())
	{
		if (bAutoReset) SurvivorsGame->ResetState(TOptional<int32>());
		return;
	}

	TArray<float> Obs = SurvivorsGame->GetObservation();
	TArray<float> Action;
	if (RunInference(Obs, Action) && Action.Num() > 0)
	{
		const int32 ActionIdx = FMath::Clamp(FMath::RoundToInt(Action[0]), 0, 4);
		SurvivorsGame->PhysicsStep(ActionIdx);
	}
}
