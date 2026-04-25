#include "Training/CoinHttpEnvService.h"
#include "HttpEnvServerBase.h"
#include "Kismet/GameplayStatics.h"

class ACoinHttpEnvService::FCoinEnvServer : public FHttpEnvServerBase
{
public:
	explicit FCoinEnvServer(ACoinGame* InGame) : Game(InGame) {}

	virtual FEnvResetResult ProcessReset(TOptional<int32> Seed) override
	{
		FEnvResetResult Result;
		if (Game)
		{
			Game->ResetState(Seed);
			Result.Obs            = Game->GetObservation();
			Result.ObsSchemaHash  = Game->GetObsSchemaHash();
		}
		return Result;
	}

protected:
	virtual void RegisterAdditionalRoutes(TSharedPtr<IHttpRouter> Router) override
	{
		ObsSchemaRoute = Router->BindRoute(
			FHttpPath(TEXT("/obs_schema")), EHttpServerRequestVerbs::VERB_GET,
			FHttpRequestHandler::CreateRaw(this, &FCoinEnvServer::HandleObsSchema));
	}

	virtual void UnregisterAdditionalRoutes(TSharedPtr<IHttpRouter> Router) override
	{
		if (Router) Router->UnbindRoute(ObsSchemaRoute);
	}

private:
	bool HandleObsSchema(const FHttpServerRequest& Request, const FHttpResultCallback& OnComplete)
	{
		if (!Game)
		{
			OnComplete(MakeJsonResponse(TEXT("{\"error\":\"game not set\"}")));
			return true;
		}

		TArray<FObsSegment> Schema = Game->GetObsSchema();

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

	FHttpRouteHandle ObsSchemaRoute;
	ACoinGame* Game; // non-owning、PIE セッション中は有効

public:
	virtual FEnvStepResult ProcessStep(const TArray<float>& Action) override
	{
		FEnvStepResult Result;
		if (Game)
		{
			const int32 ActionIdx = Action.Num() > 0
				? FMath::Clamp(static_cast<int32>(Action[0]), 0, 4)
				: 4; // デフォルト: 静止
			Game->PhysicsStep(ActionIdx);
			Result.Obs       = Game->GetObservation();
			Result.Reward    = Game->GetReward();
			Result.bDone     = Game->IsDone();
			Result.bTruncated = false;
		}
		return Result;
	}
};

ACoinHttpEnvService::ACoinHttpEnvService()
{
	PrimaryActorTick.bCanEverTick = true;
}

void ACoinHttpEnvService::BeginPlay()
{
	Super::BeginPlay();

	if (!CoinGame)
	{
		CoinGame = Cast<ACoinGame>(
			UGameplayStatics::GetActorOfClass(GetWorld(), ACoinGame::StaticClass()));
	}

	if (!CoinGame)
	{
		UE_LOG(LogTemp, Error,
			TEXT("ACoinHttpEnvService: ACoinGame が見つかりません。レベルに配置してください。"));
		return;
	}

	EnvServer = TUniquePtr<FHttpEnvServerBase>(new FCoinEnvServer(CoinGame.Get()));
	EnvServer->StartServer(static_cast<uint32>(ServerPort));
}

void ACoinHttpEnvService::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	if (EnvServer)
	{
		EnvServer->StopServer();
		EnvServer.Reset();
	}
	Super::EndPlay(EndPlayReason);
}

void ACoinHttpEnvService::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
	if (EnvServer) EnvServer->Tick();
}
