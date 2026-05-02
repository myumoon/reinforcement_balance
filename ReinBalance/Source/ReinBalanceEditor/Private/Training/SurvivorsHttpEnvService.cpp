#include "Training/SurvivorsHttpEnvService.h"
#include "HttpEnvServerBase.h"
#include "Kismet/GameplayStatics.h"
#include "Dom/JsonObject.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"

class ASurvivorsHttpEnvService::FSurvivorsEnvServer : public FHttpEnvServerBase
{
public:
	explicit FSurvivorsEnvServer(ASurvivorsGame* InGame) : Game(InGame) {}

	virtual FEnvResetResult ProcessReset(TOptional<int32> Seed) override
	{
		FEnvResetResult Result;
		if (Game)
		{
			Game->ResetState(Seed);
			Result.Obs           = Game->GetObservation();
			Result.ObsSchemaHash = Game->GetObsSchemaHash();
		}
		return Result;
	}

	virtual FEnvStepResult ProcessStep(const TArray<float>& Action, int32 Steps) override
	{
		FEnvStepResult Result;
		if (Game)
		{
			const int32 ActionIdx = Action.Num() > 0
				? FMath::Clamp(static_cast<int32>(Action[0]), 0, 4)
				: 4;
			float AccumulatedReward = 0.f;
			for (int32 i = 0; i < Steps; ++i)
			{
				Game->PhysicsStep(ActionIdx);
				AccumulatedReward += Game->GetReward();
				if (Game->IsDone())
				{
					Result.bDone = true;
					break;
				}
			}
			Result.Obs    = Game->GetObservation();
			Result.Reward = AccumulatedReward;
		}
		return Result;
	}

protected:
	virtual void RegisterAdditionalRoutes(TSharedPtr<IHttpRouter> Router) override
	{
		ObsSchemaRoute = Router->BindRoute(
			FHttpPath(TEXT("/obs_schema")), EHttpServerRequestVerbs::VERB_GET,
			FHttpRequestHandler::CreateRaw(this, &FSurvivorsEnvServer::HandleObsSchema));

		ParamsRoute = Router->BindRoute(
			FHttpPath(TEXT("/params")), EHttpServerRequestVerbs::VERB_POST,
			FHttpRequestHandler::CreateRaw(this, &FSurvivorsEnvServer::HandleParams));
	}

	virtual void UnregisterAdditionalRoutes(TSharedPtr<IHttpRouter> Router) override
	{
		if (Router)
		{
			Router->UnbindRoute(ObsSchemaRoute);
			Router->UnbindRoute(ParamsRoute);
		}
	}

private:
	bool HandleObsSchema(const FHttpServerRequest& Request, const FHttpResultCallback& OnComplete)
	{
		if (!Game)
		{
			OnComplete(MakeJsonResponse(TEXT("{\"error\":\"game not set\"}")));
			return true;
		}

		TArray<FSurvivorsObsSegment> Schema = Game->GetObsSchema();
		FString SegmentsStr;
		for (int32 i = 0; i < Schema.Num(); ++i)
		{
			SegmentsStr += FString::Printf(TEXT("{\"name\":\"%s\",\"dim\":%d}"),
				*Schema[i].Name, Schema[i].Dim);
			if (i < Schema.Num() - 1) SegmentsStr += TEXT(",");
		}
		FString Json = FString::Printf(
			TEXT("{\"segments\":[%s],\"total_dim\":%d,\"obs_schema_hash\":\"%s\"}"),
			*SegmentsStr, Game->GetObsDim(), *Game->GetObsSchemaHash());

		OnComplete(MakeJsonResponse(Json));
		return true;
	}

	bool HandleParams(const FHttpServerRequest& Request, const FHttpResultCallback& OnComplete)
	{
		if (!Game)
		{
			OnComplete(MakeJsonResponse(TEXT("{\"error\":\"game not set\"}")));
			return true;
		}

		// リクエストボディを UTF-8 文字列に変換
		FString BodyStr = FString(UTF8_TO_TCHAR(
			reinterpret_cast<const char*>(Request.Body.GetData())));

		TSharedPtr<FJsonObject> JsonObj;
		TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(BodyStr);
		if (!FJsonSerializer::Deserialize(Reader, JsonObj) || !JsonObj.IsValid())
		{
			OnComplete(MakeJsonResponse(TEXT("{\"error\":\"invalid json\"}")));
			return true;
		}

		// 各パラメータを上書き（存在するフィールドのみ）
		int32 MaxActiveEnemies;
		if (JsonObj->TryGetNumberField(TEXT("MaxActiveEnemies"), MaxActiveEnemies))
			Game->MaxActiveEnemies = FMath::Clamp(MaxActiveEnemies, 1, 20);

		double EnemySpeedMult;
		if (JsonObj->TryGetNumberField(TEXT("EnemySpeedMult"), EnemySpeedMult))
			Game->EnemySpeedMult = FMath::Clamp(static_cast<float>(EnemySpeedMult), 0.5f, 5.f);

		double SpawnInterval;
		if (JsonObj->TryGetNumberField(TEXT("SpawnInterval"), SpawnInterval))
			Game->EnemySpawnInterval = FMath::Clamp(static_cast<float>(SpawnInterval), 2.f, 60.f);

		OnComplete(MakeJsonResponse(TEXT("{\"status\":\"ok\"}")));
		return true;
	}

	FHttpRouteHandle ObsSchemaRoute;
	FHttpRouteHandle ParamsRoute;
	ASurvivorsGame*  Game; // non-owning
};

// ---- Actor ------------------------------------------------------------------

ASurvivorsHttpEnvService::ASurvivorsHttpEnvService()
{
	PrimaryActorTick.bCanEverTick = true;
}

void ASurvivorsHttpEnvService::BeginPlay()
{
	Super::BeginPlay();

	if (!SurvivorsGame)
	{
		SurvivorsGame = Cast<ASurvivorsGame>(
			UGameplayStatics::GetActorOfClass(GetWorld(), ASurvivorsGame::StaticClass()));
	}

	if (!SurvivorsGame)
	{
		UE_LOG(LogTemp, Error,
			TEXT("ASurvivorsHttpEnvService: ASurvivorsGame が見つかりません。レベルに配置してください。"));
		return;
	}

	EnvServer = TUniquePtr<FHttpEnvServerBase>(new FSurvivorsEnvServer(SurvivorsGame.Get()));
	EnvServer->StartServer(static_cast<uint32>(ServerPort));
}

void ASurvivorsHttpEnvService::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	if (EnvServer)
	{
		EnvServer->StopServer();
		EnvServer.Reset();
	}
	Super::EndPlay(EndPlayReason);
}

void ASurvivorsHttpEnvService::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
	if (EnvServer) EnvServer->Tick();
}
