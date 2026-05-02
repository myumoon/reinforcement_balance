#include "Training/BalanceHttpEnvService.h"
#include "HttpEnvServerBase.h"
#include "Kismet/GameplayStatics.h"

class ABalanceHttpEnvService::FBalanceEnvServer : public FHttpEnvServerBase
{
public:
	explicit FBalanceEnvServer(ABalanceCart* InCart) : Cart(InCart) {}

	virtual FEnvResetResult ProcessReset(TOptional<int32> Seed) override
	{
		FEnvResetResult Result;
		if (Cart)
		{
			Cart->ResetState(Seed);
			Result.Obs = Cart->GetObservation();
		}
		return Result;
	}

	virtual FEnvStepResult ProcessStep(const TArray<float>& Action, int32 Steps) override
	{
		FEnvStepResult Result;
		if (Cart)
		{
			const float Force = Action.Num() > 0 ? FMath::Clamp(Action[0], -1.f, 1.f) : 0.f;
			float AccumulatedReward = 0.f;
			for (int32 i = 0; i < Steps; ++i)
			{
				Cart->PhysicsStep(Force);
				AccumulatedReward += Cart->GetReward();
				if (Cart->IsDone())
				{
					Result.bDone = true;
					break;
				}
			}
			Result.Obs    = Cart->GetObservation();
			Result.Reward = AccumulatedReward;
		}
		return Result;
	}

private:
	ABalanceCart* Cart; // non-owning、PIE セッション中は有効
};

ABalanceHttpEnvService::ABalanceHttpEnvService()
{
	PrimaryActorTick.bCanEverTick = true;
}

void ABalanceHttpEnvService::BeginPlay()
{
	Super::BeginPlay();

	// BalanceCart が未設定の場合はレベルから自動検索
	if (!BalanceCart)
	{
		BalanceCart = Cast<ABalanceCart>(
			UGameplayStatics::GetActorOfClass(GetWorld(), ABalanceCart::StaticClass()));
	}

	if (!BalanceCart)
	{
		UE_LOG(LogTemp, Error,
			TEXT("ABalanceHttpEnvService: ABalanceCart が見つかりません。レベルに配置してください。"));
	}

	EnvServer = TUniquePtr<FHttpEnvServerBase>(new FBalanceEnvServer(BalanceCart.Get()));
	EnvServer->StartServer(static_cast<uint32>(ServerPort));
}

void ABalanceHttpEnvService::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	if (EnvServer)
	{
		EnvServer->StopServer();
		EnvServer.Reset();
	}
	Super::EndPlay(EndPlayReason);
}

void ABalanceHttpEnvService::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
	if (EnvServer)
	{
		EnvServer->Tick();
	}
}
