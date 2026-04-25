#include "InferenceControllerBase.h"
#include "NNE.h"
#include "NNETensor.h"

AInferenceControllerBase::AInferenceControllerBase()
{
	PrimaryActorTick.bCanEverTick = true;
}

void AInferenceControllerBase::BeginPlay()
{
	Super::BeginPlay();

	if (!ModelData)
	{
		UE_LOG(LogTemp, Warning, TEXT("[Inference] ModelData が未設定です。Details パネルで ONNX アセットを設定してください。"));
		return;
	}

	TWeakInterfacePtr<INNERuntimeCPU> Runtime =
		UE::NNE::GetRuntime<INNERuntimeCPU>(FString("NNERuntimeORTCpu"));
	if (!Runtime.IsValid())
	{
		UE_LOG(LogTemp, Error, TEXT("[Inference] NNERuntimeORTCpu が見つかりません。NNERuntimeORT プラグインが有効か確認してください。"));
		return;
	}

	NNEModel = Runtime->CreateModelCPU(ModelData);
	if (!NNEModel)
	{
		UE_LOG(LogTemp, Error, TEXT("[Inference] IModelCPU の生成に失敗しました。"));
		return;
	}

	NNEModelInstance = NNEModel->CreateModelInstanceCPU();
	if (!NNEModelInstance)
	{
		UE_LOG(LogTemp, Error, TEXT("[Inference] IModelInstanceCPU の生成に失敗しました。"));
		return;
	}

	TArray<UE::NNE::FTensorShape> InputShapes;
	InputShapes.Add(UE::NNE::FTensorShape::Make({ 1u, GetObsDim() }));

	auto ShapeStatus = NNEModelInstance->SetInputTensorShapes(InputShapes);
	if (ShapeStatus != UE::NNE::IModelInstanceCPU::ESetInputTensorShapesStatus::Ok)
	{
		UE_LOG(LogTemp, Error, TEXT("[Inference] 入力テンソル形状の設定に失敗しました（ObsDim=%u）。"),
			GetObsDim());
		return;
	}

	ActionBuffer.SetNumZeroed(static_cast<int32>(GetActionDim()));
	bModelReady = true;

	UE_LOG(LogTemp, Log, TEXT("[Inference] 推論準備完了 (ObsDim=%u, ActionDim=%u)"),
		GetObsDim(), GetActionDim());
}

bool AInferenceControllerBase::RunInference(const TArray<float>& Obs, TArray<float>& OutAction)
{
	if (!bModelReady) return false;

	UE::NNE::FTensorBindingCPU InBind;
	InBind.Data        = const_cast<float*>(Obs.GetData());
	InBind.SizeInBytes = static_cast<uint64>(Obs.Num()) * sizeof(float);

	UE::NNE::FTensorBindingCPU OutBind;
	OutBind.Data       = ActionBuffer.GetData();
	OutBind.SizeInBytes = static_cast<uint64>(ActionBuffer.Num()) * sizeof(float);

	auto Status = NNEModelInstance->RunSync(
		TConstArrayView<UE::NNE::FTensorBindingCPU>({ InBind }),
		TConstArrayView<UE::NNE::FTensorBindingCPU>({ OutBind }));

	if (Status != UE::NNE::IModelInstanceCPU::ERunSyncStatus::Ok)
	{
		UE_LOG(LogTemp, Warning, TEXT("[Inference] RunSync に失敗しました。"));
		return false;
	}

	OutAction = ActionBuffer;
	return true;
}
