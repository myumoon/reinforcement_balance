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

void FHttpEnvServerBase::Tick()
{
	// Reset リクエスト処理
	{
		FResetRequest Req;
		if (ResetQueue.Dequeue(Req))
		{
			FEnvResetResult Result = ProcessReset(Req.Seed);

			FString ObsStr;
			for (int32 i = 0; i < Result.Obs.Num(); ++i)
			{
				ObsStr += FString::SanitizeFloat(Result.Obs[i]);
				if (i < Result.Obs.Num() - 1) ObsStr += TEXT(",");
			}
			FString Json = FString::Printf(
				TEXT("{\"obs\":[%s],\"obs_schema_hash\":\"%s\"}"),
				*ObsStr, *Result.ObsSchemaHash);
			Req.Callback(MakeJsonResponse(Json));
		}
	}

	// Step リクエスト処理
	{
		FStepRequest Req;
		if (ActionQueue.Dequeue(Req))
		{
			FEnvStepResult Result = ProcessStep(Req.Action);

			FString ObsStr;
			for (int32 i = 0; i < Result.Obs.Num(); ++i)
			{
				ObsStr += FString::SanitizeFloat(Result.Obs[i]);
				if (i < Result.Obs.Num() - 1) ObsStr += TEXT(",");
			}
			FString Json = FString::Printf(
				TEXT("{\"obs\":[%s],\"reward\":%f,\"done\":%s,\"truncated\":%s}"),
				*ObsStr,
				Result.Reward,
				Result.bDone     ? TEXT("true") : TEXT("false"),
				Result.bTruncated ? TEXT("true") : TEXT("false"));
			Req.Callback(MakeJsonResponse(Json));
		}
	}
}

bool FHttpEnvServerBase::HandleReset(const FHttpServerRequest& Request, const FHttpResultCallback& OnComplete)
{
	TOptional<int32> Seed;
	FString Body(UTF8_TO_TCHAR(reinterpret_cast<const char*>(Request.Body.GetData())));
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
	FString Body(UTF8_TO_TCHAR(reinterpret_cast<const char*>(Request.Body.GetData())));
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
	}

	ActionQueue.Enqueue({MoveTemp(Action), OnComplete});
	return true; // 非同期応答
}

bool FHttpEnvServerBase::HandleClose(const FHttpServerRequest& Request, const FHttpResultCallback& OnComplete)
{
	OnComplete(MakeJsonResponse(TEXT("{\"ok\":true}")));
	return true;
}

TUniquePtr<FHttpServerResponse> FHttpEnvServerBase::MakeJsonResponse(const FString& Json)
{
	auto Response = FHttpServerResponse::Create(Json, TEXT("application/json"));
	Response->Code = EHttpServerResponseCodes::Ok;
	return Response;
}
