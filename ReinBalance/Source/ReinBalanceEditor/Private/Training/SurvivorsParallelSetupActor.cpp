/**
 * Survivors の複数 HTTP 環境を game thread と ParallelFor に配線する。
 * 初心者向け: state mutation が必要な params/level-up choice は並列処理より先に安全に適用する。
 */
#include "Training/SurvivorsParallelSetupActor.h"
#include "Training/SurvivorsHttpEnvService.h"
#include "Survivors/Game/SurvivorsGame.h"
#include "Survivors/SurvivorsGameLogic.h"
#include "Survivors/View/SurvivorsGameView.h"
#include "Kismet/GameplayStatics.h"
#include "Async/ParallelFor.h"

ASurvivorsParallelSetupActor::ASurvivorsParallelSetupActor()
{
	PrimaryActorTick.bCanEverTick = true;
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

	// ── 重複ポート検査（設定ミスを BeginPlay 時点で検出）────────────────
	{
		TSet<int32> UsedPorts;

		// 訓練ポート
		for (int32 i = 0; i < NumTrainEnvs; ++i)
			UsedPorts.Add(BasePort + i);

		// 主評価ポート
		if (EvalPort > 0)
		{
			if (UsedPorts.Contains(EvalPort))
			{
				UE_LOG(LogTemp, Error,
					TEXT("[SurvivorsParallelSetup] EvalPort %d が訓練ポートと重複しています。"
					     "生成をスキップします。"), EvalPort);
				return;
			}
			UsedPorts.Add(EvalPort);
		}

		// 追加評価ポート
		for (int32 Port : ExtraEvalPorts)
		{
			if (Port <= 0) continue;
			if (UsedPorts.Contains(Port))
			{
				UE_LOG(LogTemp, Error,
					TEXT("[SurvivorsParallelSetup] ExtraEvalPorts のポート %d が"
					     "他のポートと重複しています。このポートの生成をスキップします。"), Port);
				// スキップするが他のポートは続行（return しない）
				continue;
			}
			UsedPorts.Add(Port);
		}
	}

	AllServices.Reset();

	// 訓練 env をスポーン
	TrainGames.Reset();
	for (int32 i = 0; i < NumTrainEnvs; ++i)
	{
		ASurvivorsGame* Game = SpawnGame();
		if (!Game) continue;

		TrainGames.Add(Game);

		ASurvivorsHttpEnvService* Svc = SpawnService(Game, BasePort + i);
		if (Svc)
		{
			Svc->bManagedExternally = true;
			AllServices.Add(Svc);
		}
	}

	// 評価 env をスポーン（EvalPort > 0 の場合のみ）
	EvalGame = nullptr;
	if (EvalPort > 0)
	{
		EvalGame = SpawnGame();
		if (EvalGame)
		{
			ASurvivorsHttpEnvService* EvalSvc = SpawnService(EvalGame, EvalPort);
			if (EvalSvc)
			{
				EvalSvc->bManagedExternally = true;
				AllServices.Add(EvalSvc);
			}
		}
	}

	UE_LOG(LogTemp, Log,
		TEXT("[SurvivorsParallelSetup] 訓練 env %d 個（ポート %d〜%d）、"
		     "評価 env %s を生成しました。"),
		TrainGames.Num(),
		BasePort, BasePort + TrainGames.Num() - 1,
		EvalGame ? *FString::Printf(TEXT("ポート %d"), EvalPort) : TEXT("なし"));

	// 追加評価 env をスポーン（有効ポートのみ TMap に格納）
	ExtraEvalGameMap.Reset();
	{
		// 既に使用済みのポートセットを再構築（重複検査済みだが念のため）
		TSet<int32> SpawnedPorts;
		for (int32 i = 0; i < NumTrainEnvs; ++i) SpawnedPorts.Add(BasePort + i);
		if (EvalPort > 0) SpawnedPorts.Add(EvalPort);

		for (int32 Port : ExtraEvalPorts)
		{
			if (Port <= 0) continue;
			if (SpawnedPorts.Contains(Port)) continue;  // 重複はスキップ
			SpawnedPorts.Add(Port);

			ASurvivorsGame* ExtraGame = SpawnGame();
			if (!ExtraGame) continue;

			ASurvivorsHttpEnvService* ExtraSvc = SpawnService(ExtraGame, Port);
			if (!ExtraSvc)
			{
				// Service 作成失敗時は orphan の Game actor を破棄する
				ExtraGame->Destroy();
				continue;
			}

			// Game と Service の両方が成功した場合のみ Map に登録
			ExtraEvalGameMap.Add(Port, ExtraGame);
			ExtraSvc->bManagedExternally = true;
			AllServices.Add(ExtraSvc);
		}
	}

	if (ExtraEvalGameMap.Num() > 0)
	{
		TArray<FString> PortStrs;
		for (const auto& Pair : ExtraEvalGameMap)
			PortStrs.Add(FString::FromInt(Pair.Key));
		UE_LOG(LogTemp, Log,
			TEXT("[SurvivorsParallelSetup] 追加評価 env %d 個（ポート: %s）を生成しました。"),
			ExtraEvalGameMap.Num(),
			*FString::Join(PortStrs, TEXT(", ")));
	}

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

namespace
{
	/** FSurvivorsStepResult → FEnvStepResult 変換 */
	static FEnvStepResult ConvertStepResult(
		const FSurvivorsStepResult& Src, const FString& InfoJson)
	{
		FEnvStepResult Dst;
		Dst.Obs       = Src.Obs;
		Dst.Reward    = Src.Reward;
		Dst.bDone     = Src.bDone;
		Dst.bTruncated = Src.bTruncated;
		Dst.InfoJson = InfoJson;
		return Dst;
	}

	/** FSurvivorsResetResult → FEnvResetResult 変換 */
	static FEnvResetResult ConvertResetResult(const FSurvivorsResetResult& Src)
	{
		FEnvResetResult Dst;
		Dst.Obs           = Src.Obs;
		Dst.ObsSchemaHash = Src.ObsSchemaHash;
		return Dst;
	}
} // namespace

void ASurvivorsParallelSetupActor::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	if (AllServices.IsEmpty()) return;

	// ---- 1. Params をゲームスレッドで先に適用（ParallelFor より前）----
	for (ASurvivorsHttpEnvService* Svc : AllServices)
	{
		if (!Svc) continue;
		// previewをchoice mutationより先に処理してpending bindingを保つ。
		// 初心者向け: 同じframeに両requestが来ても、反実仮想値を古いIDへ誤結合しません。
		Svc->ProcessLevelUpPreviewRequests();
		Svc->ProcessLevelUpChoiceRequests();
		FString Json;
		FHttpResultCallback Cb;
		while (Svc->TakeParamsRequest(Json, Cb))
		{
			FString ResponseJson = Svc->ApplyParams(Json);
			Svc->CompleteParams(ResponseJson, MoveTemp(Cb));
		}
	}

	// ---- 2. Step/Reset リクエストを収集 ----

	struct FPendingWork
	{
		ASurvivorsHttpEnvService* Service  = nullptr;
		FSurvivorsGameLogic*      Logic    = nullptr;
		bool                      bIsStep  = true;
		TArray<float>             Action;
		int32                     Steps    = 1;
		FHttpResultCallback       StepCallback;
		TOptional<int32>          Seed;
		FHttpResultCallback       ResetCallback;
	};

	TArray<FPendingWork> Works;
	Works.Reserve(AllServices.Num());

	for (ASurvivorsHttpEnvService* Svc : AllServices)
	{
		if (!Svc) continue;
		FPendingWork W;
		W.Service = Svc;
		W.Logic   = Svc->GetGameLogic();

		// Reset を Step より優先（既存 FHttpEnvServerBase::Tick と同じ順序）
		if (Svc->TakeResetRequest(W.Seed, W.ResetCallback))
		{
			W.bIsStep = false;
			Works.Add(MoveTemp(W));
		}
		else if (Svc->TakeStepRequest(W.Action, W.Steps, W.StepCallback))
		{
			W.bIsStep = true;
			Works.Add(MoveTemp(W));
		}
	}

	if (Works.IsEmpty()) return;

	// ---- 3. Step/Reset を並列実行 ----

	TArray<FSurvivorsStepResult>  LogicStepResults;
	TArray<FSurvivorsResetResult> LogicResetResults;
	LogicStepResults.SetNum(Works.Num());
	LogicResetResults.SetNum(Works.Num());

	ParallelFor(Works.Num(), [&](int32 i)
	{
		FPendingWork& W = Works[i];
		if (!W.Logic) return;
		if (W.bIsStep)
			LogicStepResults[i]  = W.Logic->ExecStep(W.Action, W.Steps);
		else
			LogicResetResults[i] = W.Logic->ExecReset(W.Seed);
	});

	// ---- 4. レスポンス送信（ゲームスレッドで実行）----
	for (int32 i = 0; i < Works.Num(); ++i)
	{
		FPendingWork& W = Works[i];
		if (W.bIsStep)
		{
				FEnvStepResult EnvResult = ConvertStepResult(
					LogicStepResults[i], W.Service->BuildInfoJson());
			W.Service->CompleteStep(MoveTemp(EnvResult), MoveTemp(W.StepCallback));
		}
		else
		{
			FEnvResetResult EnvResult = ConvertResetResult(LogicResetResults[i]);
			W.Service->CompleteReset(MoveTemp(EnvResult), MoveTemp(W.ResetCallback));
		}
	}
}

ASurvivorsGame* ASurvivorsParallelSetupActor::FindGameByPort(int32 Port) const
{
	// 主評価 env
	if (EvalPort > 0 && Port == EvalPort)
	{
		return EvalGame.Get();
	}

	// 追加評価 env（TMap で O(1) 検索）
	if (const TObjectPtr<ASurvivorsGame>* Found = ExtraEvalGameMap.Find(Port))
	{
		return Found->Get();
	}

	// 訓練 env
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

/** Actor またはそのコンポーネントから "Game" ObjectProperty を探し値を設定する。
 *  FollowPlayerComponent のように Blueprint コンポーネント側にプロパティが定義されている
 *  ケースに対応するため、Actor 直接 → 各コンポーネントの順に検索する。
 *  @return 設定に成功した場合 true */
static bool SetGamePropertyOnActorOrComponent(AActor* Target, ASurvivorsGame* Game)
{
	if (!Target || !Game) return false;

	// 1. Actor 自身に "Game" プロパティがあれば設定
	if (FObjectProperty* Prop = CastField<FObjectProperty>(
		Target->GetClass()->FindPropertyByName(TEXT("Game"))))
	{
		Prop->SetObjectPropertyValue(
			Prop->ContainerPtrToValuePtr<void>(Target), Game);
		return true;
	}

	// 2. 見つからなければ全コンポーネントを検索（Blueprint コンポーネント対応）
	TArray<UActorComponent*> Components;
	Target->GetComponents(Components);
	for (UActorComponent* Comp : Components)
	{
		if (!Comp) continue;
		if (FObjectProperty* Prop = CastField<FObjectProperty>(
			Comp->GetClass()->FindPropertyByName(TEXT("Game"))))
		{
			Prop->SetObjectPropertyValue(
				Prop->ContainerPtrToValuePtr<void>(Comp), Game);
			UE_LOG(LogTemp, Log,
				TEXT("[SurvivorsParallelSetup] %s の Game を %s に設定しました。"),
				*Comp->GetName(), *Game->GetName());
			return true;
		}
	}
	return false;
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

		// Actor または FollowPlayerComponent の "Game" プロパティに設定（BeginPlay 前）
		if (!SetGamePropertyOnActorOrComponent(CameraActor, Game))
		{
			UE_LOG(LogTemp, Warning,
				TEXT("[SurvivorsParallelSetup] FollowPlayerCameraClass (%s) および"
				     "そのコンポーネントに 'Game' プロパティが見つかりません。"),
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

	if (!SetGamePropertyOnActorOrComponent(FollowPlayerCameraActor, Game))
	{
		UE_LOG(LogTemp, Warning,
			TEXT("[SurvivorsParallelSetup] FollowPlayerCameraActor (%s) および"
			     "そのコンポーネントに 'Game' プロパティが見つかりません。"),
			*FollowPlayerCameraActor->GetName());
		return;
	}

	UE_LOG(LogTemp, Log,
		TEXT("[SurvivorsParallelSetup] FollowPlayerCamera.Game を %s に設定しました。"),
		*Game->GetName());
}
