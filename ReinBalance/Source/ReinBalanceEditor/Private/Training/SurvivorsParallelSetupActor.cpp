#include "Training/SurvivorsParallelSetupActor.h"
#include "Training/SurvivorsHttpEnvService.h"
#include "Survivors/Logic/SurvivorsGame.h"
#include "Kismet/GameplayStatics.h"

ASurvivorsParallelSetupActor::ASurvivorsParallelSetupActor()
{
	PrimaryActorTick.bCanEverTick = false;
}

void ASurvivorsParallelSetupActor::BeginPlay()
{
	Super::BeginPlay();

	if (NumTrainEnvs <= 0)
	{
		UE_LOG(LogTemp, Error,
			TEXT("[SurvivorsParallelSetup] NumTrainEnvs が 0 以下です。"));
		return;
	}

	// 訓練 env をスポーン
	TrainGames.Reset();
	for (int32 i = 0; i < NumTrainEnvs; ++i)
	{
		ASurvivorsGame* Game = SpawnGame();
		if (!Game) continue;

		TrainGames.Add(Game);
		SpawnService(Game, BasePort + i);
	}

	// 評価 env をスポーン（EvalPort > 0 の場合のみ）
	EvalGame = nullptr;
	if (EvalPort > 0)
	{
		EvalGame = SpawnGame();
		if (EvalGame)
		{
			SpawnService(EvalGame, EvalPort);
		}
	}

	UE_LOG(LogTemp, Log,
		TEXT("[SurvivorsParallelSetup] 訓練 env %d 個（ポート %d〜%d）、"
		     "評価 env %s を生成しました。"),
		TrainGames.Num(),
		BasePort, BasePort + TrainGames.Num() - 1,
		EvalGame ? *FString::Printf(TEXT("ポート %d"), EvalPort) : TEXT("なし"));

	// プレビューカメラに Game を設定（eval があれば eval、なければ最後の train）
	ASurvivorsGame* PreviewGame = EvalGame ? EvalGame.Get()
	                            : (TrainGames.Num() > 0 ? TrainGames.Last().Get() : nullptr);
	BindCameraToGame(PreviewGame);
}

ASurvivorsGame* ASurvivorsParallelSetupActor::SpawnGame()
{
	// SurvivorsGame は純粋ロジックアクターなのでスポーン位置は Manager と同じで問題なし
	ASurvivorsGame* Game = GetWorld()->SpawnActor<ASurvivorsGame>(
		ASurvivorsGame::StaticClass(), GetActorTransform());

	if (!Game)
	{
		UE_LOG(LogTemp, Error,
			TEXT("[SurvivorsParallelSetup] ASurvivorsGame のスポーンに失敗しました。"));
	}
	return Game;
}

ASurvivorsHttpEnvService* ASurvivorsParallelSetupActor::SpawnService(
	ASurvivorsGame* Game, int32 Port)
{
	// BeginPlay でサーバーを起動するため SpawnActorDeferred を使い
	// BeginPlay 前に SurvivorsGame と ServerPort を設定する
	ASurvivorsHttpEnvService* Service = GetWorld()->SpawnActorDeferred<ASurvivorsHttpEnvService>(
		ASurvivorsHttpEnvService::StaticClass(), GetActorTransform());

	if (!Service)
	{
		UE_LOG(LogTemp, Error,
			TEXT("[SurvivorsParallelSetup] ASurvivorsHttpEnvService のスポーンに失敗しました。"
			     "（ポート %d）"), Port);
		return nullptr;
	}

	Service->SurvivorsGame = Game;
	Service->ServerPort    = Port;

	UGameplayStatics::FinishSpawningActor(Service, GetActorTransform());
	return Service;
}

void ASurvivorsParallelSetupActor::BindCameraToGame(ASurvivorsGame* Game)
{
	if (!FollowPlayerCameraActor || !Game) return;

	// FollowPlayerCamera は Blueprint アクターのため C++ から直接アクセスできない。
	// リフレクション経由で "Game" 変数を設定する。
	FObjectProperty* Prop = CastField<FObjectProperty>(
		FollowPlayerCameraActor->GetClass()->FindPropertyByName(TEXT("Game")));

	if (!Prop)
	{
		UE_LOG(LogTemp, Warning,
			TEXT("[SurvivorsParallelSetup] FollowPlayerCameraActor (%s) に"
			     " 'Game' プロパティが見つかりません。"),
			*FollowPlayerCameraActor->GetName());
		return;
	}

	Prop->SetObjectPropertyValue(
		Prop->ContainerPtrToValuePtr<void>(FollowPlayerCameraActor), Game);

	UE_LOG(LogTemp, Log,
		TEXT("[SurvivorsParallelSetup] FollowPlayerCamera.Game を %s に設定しました。"),
		*Game->GetName());
}
