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
			Result.Obs = Game->GetObservation();
		}
		return Result;
	}

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

private:
	ACoinGame* Game; // non-owning、PIE セッション中は有効
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
