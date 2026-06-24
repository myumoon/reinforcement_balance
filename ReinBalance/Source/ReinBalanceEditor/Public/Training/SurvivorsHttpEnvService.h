#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "HttpEnvServerBase.h"
#include "IHttpEnvServer.h"
#include "Survivors/Game/SurvivorsGame.h"
#include "Survivors/SurvivorsGameLogic.h"
#include "SurvivorsHttpEnvService.generated.h"

/**
 * SurvivorsGame 固有の HTTP 環境サービス。
 *
 * /reset・/step・/close の他に以下を提供する:
 *   GET  /obs_schema  — obs セグメント定義と total_dim を返す
 *   POST /params      — カリキュラム用難易度パラメータを受け取る
 *     {"MaxActiveEnemies": 5, "EnemySpeedMult": 1.2, "SpawnInterval": 6.0}
 *
 *   bManagedExternally=true にすると Tick() がキュー処理をスキップする。
 *   ASurvivorsParallelSetupActor が一元管理する場合に使用する。
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

	/**
	 * true にすると Tick() がキュー処理をスキップする。
	 * ASurvivorsParallelSetupActor が Step/Reset を一元管理する場合に設定する。
	 */
	bool bManagedExternally = false;

	/** ActionQueue から Step リクエストを 1 件取り出す（GameThread 専用） */
	bool TakeStepRequest(TArray<float>& OutAction, int32& OutSteps, FHttpResultCallback& OutCallback);

	/** ResetQueue から Reset リクエストを 1 件取り出す（GameThread 専用） */
	bool TakeResetRequest(TOptional<int32>& OutSeed, FHttpResultCallback& OutCallback);

	/** ParamsQueue から Params リクエストを 1 件取り出す（GameThread 専用） */
	bool TakeParamsRequest(FString& OutJson, FHttpResultCallback& OutCallback);

	/** Step 結果を HTTP レスポンスとして返す */
	void CompleteStep(FEnvStepResult Result, FHttpResultCallback Callback);

	/** Reset 結果を HTTP レスポンスとして返す */
	void CompleteReset(FEnvResetResult Result, FHttpResultCallback Callback);

	/**
	 * JSON を解析してゲームフィールドに適用する（ゲームスレッド専用）。
	 * 戻り値は HTTP レスポンスとして返す JSON 文字列。
	 * エラー時は {"error":"..."} を返す（既存 HandleParams の挙動を維持）。
	 */
	FString ApplyParams(const FString& Json);

	/** ASurvivorsParallelSetupActor が並列 Tick 時に使用するゲームロジックポインタ */
	FSurvivorsGameLogic* GetGameLogic();

protected:
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
	virtual void Tick(float DeltaTime) override;

private:
	class FSurvivorsEnvServer;
	TUniquePtr<FHttpEnvServerBase> EnvServer;
};
