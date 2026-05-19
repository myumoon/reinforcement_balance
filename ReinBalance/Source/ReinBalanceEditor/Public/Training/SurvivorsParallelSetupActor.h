#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "SurvivorsParallelSetupActor.generated.h"

class ASurvivorsGame;
class ASurvivorsHttpEnvService;

/**
 * 並列訓練環境を自動生成・配線するセットアップアクター。
 *
 * レベルに 1 つ配置し、NumTrainEnvs / BasePort / EvalPort を設定するだけで
 * PIE 開始時（BeginPlay）に以下を自動実行する:
 *   - ASurvivorsGame × (NumTrainEnvs + eval有無) のスポーン
 *   - ASurvivorsHttpEnvService × 同数のスポーンとポート・Game 参照の設定
 *   - FollowPlayerCameraActor の Game 変数を eval Game（なければ最後の train Game）に設定
 *
 * 使用手順:
 *   1. レベルから手動配置済みの ASurvivorsGame / ASurvivorsHttpEnvService を全削除
 *   2. このアクターをレベルに 1 つ配置
 *   3. Details パネルで NumTrainEnvs・BasePort・EvalPort を Python config と合わせて設定
 *   4. FollowPlayerCameraActor にレベル上の FollowPlayerCamera を設定
 */
UCLASS()
class REINBALANCEEDITOR_API ASurvivorsParallelSetupActor : public AActor
{
	GENERATED_BODY()

public:
	ASurvivorsParallelSetupActor();

	/** 訓練 env 数（Python の n_envs に合わせる）*/
	UPROPERTY(EditAnywhere, Category = "Training|Parallel")
	int32 NumTrainEnvs = 1;

	/** 訓練 env のベースポート。BasePort〜BasePort+NumTrainEnvs-1 を使用する */
	UPROPERTY(EditAnywhere, Category = "Training|Parallel")
	int32 BasePort = 8767;

	/** 評価 env のポート。0 の場合は評価 env を生成しない */
	UPROPERTY(EditAnywhere, Category = "Training|Parallel")
	int32 EvalPort = 8771;

	/**
	 * プレビューカメラアクター（FollowPlayerCamera）。
	 * BeginPlay で eval Game（なければ最後の train Game）を
	 * このアクターの "Game" 変数に設定する。未設定時はスキップ。
	 */
	UPROPERTY(EditAnywhere, Category = "Training|Parallel")
	TObjectPtr<AActor> FollowPlayerCameraActor;

	/** スポーン済み訓練 Game（デバッグ確認用）*/
	UPROPERTY(VisibleAnywhere, Category = "Training|Parallel|Debug")
	TArray<TObjectPtr<ASurvivorsGame>> TrainGames;

	/** スポーン済み評価 Game（EvalPort=0 の場合は null）*/
	UPROPERTY(VisibleAnywhere, Category = "Training|Parallel|Debug")
	TObjectPtr<ASurvivorsGame> EvalGame;

protected:
	virtual void BeginPlay() override;

private:
	ASurvivorsGame*           SpawnGame();
	ASurvivorsHttpEnvService* SpawnService(ASurvivorsGame* Game, int32 Port);
	void                      BindCameraToGame(ASurvivorsGame* Game);
};
