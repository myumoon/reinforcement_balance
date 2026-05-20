#include "Training/SurvivorsParallelSetupActor.h"
#include "Training/SurvivorsHttpEnvService.h"
#include "Survivors/Logic/SurvivorsGame.h"
#include "Survivors/View/SurvivorsGameView.h"
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

	// GameView をスポーン（ViewPort > 0 の場合のみ）
	SpawnedGameView = nullptr;
	if (ViewPort > 0)
	{
		ASurvivorsGame* ViewGame = FindGameByPort(ViewPort);
		if (ViewGame)
		{
			SpawnedGameView = SpawnGameView(ViewGame);
			UE_LOG(LogTemp, Log,
				TEXT("[SurvivorsParallelSetup] GameView をポート %d の Game にバインドしました。"),
				ViewPort);
		}
		else
		{
			UE_LOG(LogTemp, Warning,
				TEXT("[SurvivorsParallelSetup] ViewPort %d に対応する Game が見つかりません。"
				     " BasePort=%d〜%d、EvalPort=%d"),
				ViewPort, BasePort, BasePort + TrainGames.Num() - 1, EvalPort);
		}
	}

	// プレビューカメラを ViewPort の Game にバインド。
	// ViewPort が未設定の場合は eval → 最後の train にフォールバック。
	ASurvivorsGame* CameraTarget = nullptr;
	if (ViewPort > 0)
	{
		CameraTarget = FindGameByPort(ViewPort);
	}
	if (!CameraTarget)
	{
		CameraTarget = EvalGame ? EvalGame.Get()
		             : (TrainGames.Num() > 0 ? TrainGames.Last().Get() : nullptr);
	}
	BindCameraToGame(CameraTarget);
}

ASurvivorsGame* ASurvivorsParallelSetupActor::FindGameByPort(int32 Port) const
{
	if (EvalPort > 0 && Port == EvalPort)
	{
		return EvalGame.Get();
	}
	const int32 Idx = Port - BasePort;
	if (TrainGames.IsValidIndex(Idx))
	{
		return TrainGames[Idx].Get();
	}
	return nullptr;
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

ASurvivorsGameView* ASurvivorsParallelSetupActor::SpawnGameView(ASurvivorsGame* Game)
{
	if (!Game) return nullptr;

	// SpawnActorDeferred で BeginPlay 前に Game を設定する
	ASurvivorsGameView* View = GetWorld()->SpawnActorDeferred<ASurvivorsGameView>(
		ASurvivorsGameView::StaticClass(), GetActorTransform());

	if (!View)
	{
		UE_LOG(LogTemp, Error,
			TEXT("[SurvivorsParallelSetup] ASurvivorsGameView のスポーンに失敗しました。"));
		return nullptr;
	}

	View->Game = Game;
	UGameplayStatics::FinishSpawningActor(View, GetActorTransform());
	return View;
}

void ASurvivorsParallelSetupActor::BindCameraToGame(ASurvivorsGame* Game)
{
	if (!Game) return;

	// FollowPlayerCameraClass が設定されていれば動的スポーンして BeginPlay 前に Game を注入する
	if (FollowPlayerCameraClass)
	{
		AActor* CameraActor = GetWorld()->SpawnActorDeferred<AActor>(
			FollowPlayerCameraClass, GetActorTransform());

		if (!CameraActor)
		{
			UE_LOG(LogTemp, Warning,
				TEXT("[SurvivorsParallelSetup] FollowPlayerCamera のスポーンに失敗しました。"));
			return;
		}

		// Blueprint アクターのため reflection 経由で "Game" プロパティを設定してから BeginPlay を呼ぶ
		FObjectProperty* Prop = CastField<FObjectProperty>(
			CameraActor->GetClass()->FindPropertyByName(TEXT("Game")));

		if (Prop)
		{
			Prop->SetObjectPropertyValue(
				Prop->ContainerPtrToValuePtr<void>(CameraActor), Game);
		}
		else
		{
			UE_LOG(LogTemp, Warning,
				TEXT("[SurvivorsParallelSetup] FollowPlayerCameraClass (%s) に"
				     " 'Game' プロパティが見つかりません。"),
				*FollowPlayerCameraClass->GetName());
		}

		UGameplayStatics::FinishSpawningActor(CameraActor, GetActorTransform());

		UE_LOG(LogTemp, Log,
			TEXT("[SurvivorsParallelSetup] FollowPlayerCamera を動的スポーンし"
			     " Game を %s に設定しました。"),
			*Game->GetName());
		return;
	}

	// フォールバック: レベル上の FollowPlayerCameraActor に reflection で設定
	if (!FollowPlayerCameraActor) return;

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
