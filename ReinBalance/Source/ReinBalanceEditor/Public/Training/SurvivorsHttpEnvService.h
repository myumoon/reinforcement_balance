#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "HttpEnvServerBase.h"
#include "SurvivorsGame.h"
#include "SurvivorsHttpEnvService.generated.h"

/**
 * SurvivorsGame 固有の HTTP 環境サービス。
 *
 * /reset・/step・/close の他に以下を提供する:
 *   GET  /obs_schema  — obs セグメント定義と total_dim を返す
 *   POST /params      — カリキュラム用難易度パラメータを受け取る
 *     {"MaxActiveEnemies": 5, "EnemySpeedMult": 1.2, "SpawnInterval": 6.0}
 */
UCLASS()
class REINBALANCEEDITOR_API ASurvivorsHttpEnvService : public AActor
{
	GENERATED_BODY()

public:
	ASurvivorsHttpEnvService();

	UPROPERTY(EditAnywhere, Category = "Training")
	int32 ServerPort = 8767;

	/** レベルに配置した ASurvivorsGame をここに設定する。未設定時は自動検索。 */
	UPROPERTY(EditAnywhere, Category = "Training")
	TObjectPtr<ASurvivorsGame> SurvivorsGame;

protected:
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
	virtual void Tick(float DeltaTime) override;

private:
	class FSurvivorsEnvServer;
	TUniquePtr<FHttpEnvServerBase> EnvServer;
};
