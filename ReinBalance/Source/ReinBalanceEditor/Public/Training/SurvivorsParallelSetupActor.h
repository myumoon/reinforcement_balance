#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "SurvivorsParallelSetupActor.generated.h"

class ASurvivorsGame;
class ASurvivorsGameView;
class ASurvivorsHttpEnvService;

/**
 * 並列訓練環境を自動生成・配線するセットアップアクター。
 *
 * レベルに 1 つ配置し、NumTrainEnvs / BasePort / EvalPort を設定するだけで
 * PIE 開始時（BeginPlay）に以下を自動実行する:
 *   - ASurvivorsGame × (NumTrainEnvs + eval有無) のスポーン
 *   - ASurvivorsHttpEnvService × 同数のスポーンとポート・Game 参照の設定
 *   - ViewPort で指定したゲームの ASurvivorsGameView を動的スポーン
 *   - FollowPlayerCameraClass が設定されていれば FollowPlayerCamera を動的スポーンして Game を注入
 *
 * 使用手順:
 *   1. レベルから手動配置済みの ASurvivorsGame / ASurvivorsHttpEnvService /
 *      ASurvivorsGameView / FollowPlayerCamera を全削除
 *   2. このアクターをレベルに 1 つ配置
 *   3. Details パネルで各パラメータを Python config と合わせて設定
 */
UCLASS()
class REINBALANCEEDITOR_API ASurvivorsParallelSetupActor : public AActor
{
	GENERATED_BODY()

public:
	ASurvivorsParallelSetupActor();

	// ── 訓練環境 ──────────────────────────────────────────────────────────

	/** 訓練 env 数（Python の n_envs に合わせる）*/
	UPROPERTY(EditAnywhere, Category = "Training|Parallel")
	int32 NumTrainEnvs = 1;

	/** 訓練 env のベースポート。BasePort〜BasePort+NumTrainEnvs-1 を使用する */
	UPROPERTY(EditAnywhere, Category = "Training|Parallel")
	int32 BasePort = 8767;

	/** 評価 env のポート。0 の場合は評価 env を生成しない */
	UPROPERTY(EditAnywhere, Category = "Training|Parallel")
	int32 EvalPort = 8771;

	// ── 表示設定 ──────────────────────────────────────────────────────────

	/**
	 * 表示（View）する Game のポート番号。
	 * BasePort〜BasePort+NumTrainEnvs-1（訓練 env）または EvalPort（評価 env）を指定。
	 * 0 の場合は ASurvivorsGameView を生成しない。
	 */
	UPROPERTY(EditAnywhere, Category = "Training|View")
	int32 ViewPort = 0;

	/**
	 * スポーンする FollowPlayerCamera の Blueprint クラス。
	 * 設定時は動的スポーンし BeginPlay 前に Game を注入するため、
	 * Camera の BeginPlay 時点で Game 参照が正しく設定される（推奨）。
	 * 未設定時は FollowPlayerCameraActor への reflection アプローチにフォールバック。
	 */
	UPROPERTY(EditAnywhere, Category = "Training|View")
	TSubclassOf<AActor> FollowPlayerCameraClass;

	/**
	 * レベル上の FollowPlayerCamera アクター（フォールバック用）。
	 * FollowPlayerCameraClass が未設定の場合にのみ reflection で Game を設定する。
	 * 推奨: FollowPlayerCameraClass を設定して動的スポーンに移行すること。
	 */
	UPROPERTY(EditAnywhere, Category = "Training|View")
	TObjectPtr<AActor> FollowPlayerCameraActor;

	// ── デバッグ ──────────────────────────────────────────────────────────

	/** スポーン済み訓練 Game */
	UPROPERTY(VisibleAnywhere, Category = "Training|Debug")
	TArray<TObjectPtr<ASurvivorsGame>> TrainGames;

	/** スポーン済み評価 Game（EvalPort=0 の場合は null）*/
	UPROPERTY(VisibleAnywhere, Category = "Training|Debug")
	TObjectPtr<ASurvivorsGame> EvalGame;

	/** スポーン済み GameView（ViewPort=0 の場合は null）*/
	UPROPERTY(VisibleAnywhere, Category = "Training|Debug")
	TObjectPtr<ASurvivorsGameView> SpawnedGameView;

	/** 全環境の HTTP サービス（Train + Eval を含む）*/
	UPROPERTY(VisibleAnywhere, Category = "Training|Debug")
	TArray<TObjectPtr<ASurvivorsHttpEnvService>> AllServices;

protected:
	virtual void BeginPlay() override;
	virtual void Tick(float DeltaTime) override;

private:
	ASurvivorsGame*           SpawnGame();
	ASurvivorsHttpEnvService* SpawnService(ASurvivorsGame* Game, int32 Port);
	ASurvivorsGameView*       SpawnGameView(ASurvivorsGame* Game);
	void                      BindCameraToGame(ASurvivorsGame* Game);

	/** ポート番号から対応する Game を返す。見つからなければ nullptr */
	ASurvivorsGame*           FindGameByPort(int32 Port) const;
};
