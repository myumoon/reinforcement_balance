#include "BalanceInferenceController.h"
#include "NNE.h"
#include "NNETensor.h"

ABalanceInferenceController::ABalanceInferenceController()
{
	PrimaryActorTick.bCanEverTick = true;
}

void ABalanceInferenceController::BeginPlay()
{
	Super::BeginPlay();

	if (!ModelData)
	{
		UE_LOG(LogTemp, Warning, TEXT("[Inference] ModelData が未設定です。Details パネルで ONNX アセットを設定してください。"));
		return;
	}
	if (!Cart)
	{
		UE_LOG(LogTemp, Warning, TEXT("[Inference] Cart が未設定です。Details パネルで ABalanceCart を設定してください。"));
		return;
	}

	// NNERuntimeORT ランタイムを取得
	TWeakInterfacePtr<INNERuntimeCPU> Runtime =
		UE::NNE::GetRuntime<INNERuntimeCPU>(TEXT("NNERuntimeORT"));
	if (!Runtime.IsValid())
	{
		UE_LOG(LogTemp, Error, TEXT("[Inference] NNERuntimeORT が見つかりません。プラグインが有効か確認してください。"));
		return;
	}

	// モデル生成
	NNEModel = Runtime->CreateModelCPU(ModelData);
	if (!NNEModel)
	{
		UE_LOG(LogTemp, Error, TEXT("[Inference] IModelCPU の生成に失敗しました。"));
		return;
	}

	// モデルインスタンス生成
	NNEModelInstance = NNEModel->CreateModelInstanceCPU();
	if (!NNEModelInstance)
	{
		UE_LOG(LogTemp, Error, TEXT("[Inference] IModelInstanceCPU の生成に失敗しました。"));
		return;
	}

	// 入力テンソル形状を設定: [1, 6] (バッチサイズ=1, 観測次元=6)
	TArray<UE::NNE::FTensorShape> InputShapes;
	InputShapes.Add(UE::NNE::FTensorShape::Make({ 1u, 6u }));

	auto ShapeStatus = NNEModelInstance->SetInputTensorShapes(InputShapes);
	if (ShapeStatus != UE::NNE::IModelInstanceCPU::ESetInputTensorShapesStatus::Ok)
	{
		UE_LOG(LogTemp, Error, TEXT("[Inference] 入力テンソル形状の設定に失敗しました。ONNX の入力次元を確認してください。"));
		return;
	}

	bModelReady = true;
	Cart->ResetState(TOptional<int32>());
	UE_LOG(LogTemp, Log, TEXT("[Inference] 推論準備完了。PIE 中は毎フレーム ONNX 推論で Cart を制御します。"));
}

void ABalanceInferenceController::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
	if (!bModelReady) return;

	if (Cart->IsDone())
	{
		if (bAutoReset)
		{
			Cart->ResetState(TOptional<int32>());
		}
		return;
	}

	// 観測値を取得して推論
	TArray<float> Obs = Cart->GetObservation(); // 6 次元

	float ActionValue = 0.f;

	UE::NNE::FTensorBindingCPU InBind;
	InBind.Data      = Obs.GetData();
	InBind.SizeInBytes = static_cast<uint64>(Obs.Num()) * sizeof(float);

	UE::NNE::FTensorBindingCPU OutBind;
	OutBind.Data       = &ActionValue;
	OutBind.SizeInBytes = sizeof(float);

	auto RunStatus = NNEModelInstance->RunSync(
		TConstArrayView<UE::NNE::FTensorBindingCPU>({ InBind }),
		TConstArrayView<UE::NNE::FTensorBindingCPU>({ OutBind }));

	if (RunStatus != UE::NNE::IModelInstanceCPU::ERunSyncStatus::Ok)
	{
		UE_LOG(LogTemp, Warning, TEXT("[Inference] RunSync に失敗しました。"));
		return;
	}

	// PPO は tanh squash なしのため推論値をクリップして適用
	Cart->PhysicsStep(FMath::Clamp(ActionValue, -1.f, 1.f));
}
