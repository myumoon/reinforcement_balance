#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "HttpEnvServerBase.h"
#include "BalanceCart.h"
#include "BalanceHttpEnvService.generated.h"

/**
 * BalancePole 固有の HTTP 環境サービス。
 *
 * このアクターを PIE レベルに配置すると BeginPlay で HTTP サーバーが起動し、
 * Python 側から /reset・/step・/close を受け付ける。
 * BalanceCart を参照して物理操作を行う。
 */
UCLASS()
class REINBALANCEEDITOR_API ABalanceHttpEnvService : public AActor
{
	GENERATED_BODY()

public:
	ABalanceHttpEnvService();

	UPROPERTY(EditAnywhere, Category = "Training")
	int32 ServerPort = 8765;

	/** レベルに配置した ABalanceCart をここに設定する。未設定時は自動検索。 */
	UPROPERTY(EditAnywhere, Category = "Training")
	TObjectPtr<ABalanceCart> BalanceCart;

protected:
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
	virtual void Tick(float DeltaTime) override;

private:
	class FBalanceEnvServer;
	TUniquePtr<FHttpEnvServerBase> EnvServer;
};
