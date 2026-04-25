#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "HttpEnvServerBase.h"
#include "CoinGame.h"
#include "CoinHttpEnvService.generated.h"

/**
 * CoinGame 固有の HTTP 環境サービス。
 *
 * このアクターを PIE レベルに配置すると BeginPlay で HTTP サーバーが起動し、
 * Python 側から /reset・/step・/close を受け付ける。
 * ACoinGame を参照してゲームロジックを操作する。
 */
UCLASS()
class REINBALANCEEDITOR_API ACoinHttpEnvService : public AActor
{
	GENERATED_BODY()

public:
	ACoinHttpEnvService();

	UPROPERTY(EditAnywhere, Category = "Training")
	int32 ServerPort = 8766;

	/** レベルに配置した ACoinGame をここに設定する。未設定時は自動検索。 */
	UPROPERTY(EditAnywhere, Category = "Training")
	TObjectPtr<ACoinGame> CoinGame;

protected:
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
	virtual void Tick(float DeltaTime) override;

private:
	class FCoinEnvServer;
	TUniquePtr<FHttpEnvServerBase> EnvServer;
};
