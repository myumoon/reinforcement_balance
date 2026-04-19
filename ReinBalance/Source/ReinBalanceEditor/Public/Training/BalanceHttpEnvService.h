#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "HttpEnvServerBase.h"
#include "BalanceHttpEnvService.generated.h"

/**
 * BalancePole 固有の HTTP 環境サービス。
 *
 * このアクターを PIE レベルに配置すると BeginPlay で HTTP サーバーが起動し、
 * Python 側から /reset・/step・/close を受け付ける。
 *
 * 物理操作の実装は後続フェーズ（ABalanceCart 実装後）に追加する。
 * 現フェーズではインフラ検証用のスタブ観測値を返す。
 */
UCLASS()
class REINBALANCEEDITOR_API ABalanceHttpEnvService : public AActor
{
	GENERATED_BODY()

public:
	ABalanceHttpEnvService();

	UPROPERTY(EditAnywhere, Category = "Training")
	int32 ServerPort = 8765;

protected:
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
	virtual void Tick(float DeltaTime) override;

private:
	class FBalanceEnvServer;
	TUniquePtr<FBalanceEnvServer> EnvServer;
};
