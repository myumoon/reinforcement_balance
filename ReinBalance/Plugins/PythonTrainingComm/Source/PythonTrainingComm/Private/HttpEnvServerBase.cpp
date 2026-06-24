#include "HttpEnvServerBase.h"
#include "HttpServerModule.h"
#include "HttpServerResponse.h"
#include "HttpServerRequest.h"
#include "Dom/JsonObject.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonWriter.h"
#include "Serialization/JsonSerializer.h"

FHttpEnvServerBase::FHttpEnvServerBase() = default;
FHttpEnvServerBase::~FHttpEnvServerBase() { StopServer(); }

void FHttpEnvServerBase::StartServer(uint32 Port)
{
	FHttpServerModule& HttpModule = FHttpServerModule::Get();
	HttpRouter = HttpModule.GetHttpRouter(Port);
	if (!HttpRouter)
	{
		UE_LOG(LogTemp, Error, TEXT("PythonTrainingComm: Failed to get HTTP router on port %d"), Port);
		return;
	}

	ResetRoute = HttpRouter->BindRoute(
		FHttpPath(TEXT("/reset")), EHttpServerRequestVerbs::VERB_POST,
		FHttpRequestHandler::CreateRaw(this, &FHttpEnvServerBase::HandleReset));

	StepRoute = HttpRouter->BindRoute(
		FHttpPath(TEXT("/step")), EHttpServerRequestVerbs::VERB_POST,
		FHttpRequestHandler::CreateRaw(this, &FHttpEnvServerBase::HandleStep));

	CloseRoute = HttpRouter->BindRoute(
		FHttpPath(TEXT("/close")), EHttpServerRequestVerbs::VERB_POST,
		FHttpRequestHandler::CreateRaw(this, &FHttpEnvServerBase::HandleClose));

	RegisterAdditionalRoutes(HttpRouter);

	HttpModule.StartAllListeners();
	UE_LOG(LogTemp, Log, TEXT("PythonTrainingComm: HTTP server started on port %d"), Port);
}

void FHttpEnvServerBase::StopServer()
{
	if (HttpRouter)
	{
		UnregisterAdditionalRoutes(HttpRouter);
		HttpRouter->UnbindRoute(ResetRoute);
		HttpRouter->UnbindRoute(StepRoute);
		HttpRouter->UnbindRoute(CloseRoute);
		FHttpServerModule::Get().StopAllListeners();
		HttpRouter.Reset();
	}
}

FString FHttpEnvServerBase::BuildResetJson(const FEnvResetResult& Result)
{
	FString ObsStr;
	for (int32 i = 0; i < Result.Obs.Num(); ++i)
	{
		ObsStr += FString::SanitizeFloat(Result.Obs[i]);
		if (i < Result.Obs.Num() - 1) ObsStr += TEXT(",");
	}
	return FString::Printf(
		TEXT("{\"obs\":[%s],\"obs_schema_hash\":\"%s\"}"),
		*ObsStr, *Result.ObsSchemaHash);
}

FString FHttpEnvServerBase::BuildStepJson(const FEnvStepResult& Result)
{
	FString ObsStr;
	for (int32 i = 0; i < Result.Obs.Num(); ++i)
	{
		ObsStr += FString::SanitizeFloat(Result.Obs[i]);
		if (i < Result.Obs.Num() - 1) ObsStr += TEXT(",");
	}
	const FString InfoJson = Result.InfoJson.IsEmpty() ? TEXT("{}") : Result.InfoJson;
	return FString::Printf(
		TEXT("{\"obs\":[%s],\"reward\":%f,\"done\":%s,\"truncated\":%s,\"info\":%s}"),
		*ObsStr,
		Result.Reward,
		Result.bDone      ? TEXT("true") : TEXT("false"),
		Result.bTruncated ? TEXT("true") : TEXT("false"),
		*InfoJson);
}

void FHttpEnvServerBase::Tick()
{
	// Reset リクエスト処理
	{
		FResetRequest Req;
		if (ResetQueue.Dequeue(Req))
		{
			FEnvResetResult Result = ProcessReset(Req.Seed);
			Req.Callback(MakeJsonResponse(BuildResetJson(Result)));
		}
	}

	// Step リクエスト処理
	{
		FStepRequest Req;
		if (ActionQueue.Dequeue(Req))
		{
			FEnvStepResult Result = ProcessStep(Req.Action, Req.Steps);
			Req.Callback(MakeJsonResponse(BuildStepJson(Result)));
		}
	}
}

bool FHttpEnvServerBase::TakeStepRequest(
	TArray<float>& OutAction, int32& OutSteps, FHttpResultCallback& OutCallback)
{
	FStepRequest Req;
	if (!ActionQueue.Dequeue(Req)) return false;
	OutAction   = MoveTemp(Req.Action);
	OutSteps    = Req.Steps;
	OutCallback = MoveTemp(Req.Callback);
	return true;
}

bool FHttpEnvServerBase::TakeResetRequest(
	TOptional<int32>& OutSeed, FHttpResultCallback& OutCallback)
{
	FResetRequest Req;
	if (!ResetQueue.Dequeue(Req)) return false;
	OutSeed     = Req.Seed;
	OutCallback = MoveTemp(Req.Callback);
	return true;
}

void FHttpEnvServerBase::CompleteStep(FEnvStepResult Result, FHttpResultCallback Callback)
{
	Callback(MakeJsonResponse(BuildStepJson(Result)));
}

void FHttpEnvServerBase::CompleteReset(FEnvResetResult Result, FHttpResultCallback Callback)
{
	Callback(MakeJsonResponse(BuildResetJson(Result)));
}

bool FHttpEnvServerBase::HandleReset(const FHttpServerRequest& Request, const FHttpResultCallback& OnComplete)
{
	TOptional<int32> Seed;
	FString Body = ParseBodyString(Request);
	TSharedPtr<FJsonObject> JsonObj;
	TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Body);
	if (FJsonSerializer::Deserialize(Reader, JsonObj) && JsonObj.IsValid())
	{
		int32 SeedVal;
		if (JsonObj->TryGetNumberField(TEXT("seed"), SeedVal))
		{
			Seed = SeedVal;
		}
	}

	ResetQueue.Enqueue({Seed, OnComplete});
	return true; // 非同期応答
}

bool FHttpEnvServerBase::HandleStep(const FHttpServerRequest& Request, const FHttpResultCallback& OnComplete)
{
	TArray<float> Action;
	int32 Steps = 1;
	FString Body = ParseBodyString(Request);
	TSharedPtr<FJsonObject> JsonObj;
	TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Body);
	if (FJsonSerializer::Deserialize(Reader, JsonObj) && JsonObj.IsValid())
	{
		const TArray<TSharedPtr<FJsonValue>>* ActionArr;
		if (JsonObj->TryGetArrayField(TEXT("action"), ActionArr))
		{
			for (const TSharedPtr<FJsonValue>& Val : *ActionArr)
			{
				Action.Add(static_cast<float>(Val->AsNumber()));
			}
		}
		double StepsVal;
		if (JsonObj->TryGetNumberField(TEXT("steps"), StepsVal))
			Steps = FMath::Clamp(static_cast<int32>(StepsVal), 1, 100);
	}

	ActionQueue.Enqueue({MoveTemp(Action), Steps, OnComplete});
	return true; // 非同期応答
}

bool FHttpEnvServerBase::HandleClose(const FHttpServerRequest& Request, const FHttpResultCallback& OnComplete)
{
	OnComplete(MakeJsonResponse(TEXT("{\"ok\":true}")));
	return true;
}

FString FHttpEnvServerBase::ParseBodyString(const FHttpServerRequest& Request)
{
	if (Request.Body.IsEmpty()) return {};
	TArray<uint8> Buf(Request.Body);
	Buf.Add(0);
	return FString(UTF8_TO_TCHAR(reinterpret_cast<const char*>(Buf.GetData())));
}

TUniquePtr<FHttpServerResponse> FHttpEnvServerBase::MakeJsonResponse(const FString& Json)
{
	auto Response = FHttpServerResponse::Create(Json, TEXT("application/json"));
	Response->Code = EHttpServerResponseCodes::Ok;
	return Response;
}
