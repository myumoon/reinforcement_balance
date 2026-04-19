#include "Training/BalanceHttpEnvService.h"
#include "HttpEnvServerBase.h"

/**
 * BalancePole 固有の HTTP サーバー実装。
 *
 * ProcessStep / ProcessReset でゲーム物理を操作する。
 * 現段階では ABalanceCart が未実装のためスタブ観測値を返す。
 * ABalanceCart 実装後に TODO 箇所を置き換える。
 */
class ABalanceHttpEnvService::FBalanceEnvServer : public FHttpEnvServerBase
{
public:
	virtual FEnvResetResult ProcessReset(TOptional<int32> Seed) override
	{
		// TODO: ABalanceCart を取得して物理リセットを行い実際の観測値を返す
		FEnvResetResult Result;
		Result.Obs = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
		return Result;
	}

	virtual FEnvStepResult ProcessStep(float Force) override
	{
		// TODO: ABalanceCart に力を加え、物理ティック後の観測値・報酬・終了フラグを返す
		FEnvStepResult Result;
		Result.Obs = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
		Result.Reward = 1.f;
		Result.bDone = false;
		Result.bTruncated = false;
		return Result;
	}
};

ABalanceHttpEnvService::ABalanceHttpEnvService()
{
	PrimaryActorTick.bCanEverTick = true;
}

void ABalanceHttpEnvService::BeginPlay()
{
	Super::BeginPlay();
	EnvServer = MakeUnique<FBalanceEnvServer>();
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
