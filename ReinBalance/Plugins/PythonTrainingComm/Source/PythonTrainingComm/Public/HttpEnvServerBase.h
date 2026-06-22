#pragma once

#include "CoreMinimal.h"
#include "IHttpEnvServer.h"
#include "IHttpRouter.h"

/**
 * FHttpServerModule を使った HTTP 環境サーバーの基底クラス。
 *
 * /reset・/step・/close のルーティングと JSON 解析を担う。
 * ゲーム固有の処理は ProcessStep / ProcessReset をオーバーライドして実装する。
 *
 * スレッド安全性:
 *   HTTP ハンドラは FHttpServerModule のワーカースレッドで実行される。
 *   ActionQueue / ResetQueue / StepResultQueue / ResetResultQueue は
 *   TQueue<T, EQueueMode::Spsc> で実装し、書き込みと読み出しが別スレッドになることを保証する。
 */
class PYTHONTRAININGCOMM_API FHttpEnvServerBase : public IHttpEnvServer
{
public:
	FHttpEnvServerBase();
	virtual ~FHttpEnvServerBase() override;

	virtual void StartServer(uint32 Port) override;
	virtual void StopServer() override;

	/**
	 * GameThread から毎 Tick 呼び出す。
	 * キューにデータがあれば ProcessStep / ProcessReset を実行して結果をキューに書く。
	 */
	virtual void Tick() override;

	/**
	 * Jsonレスポンスを作成する
	 * @param Json レスポンスボディの JSON 文字列（例: {"obs":[0.0,1.0],"reward":1.0,"done":false}）
	 * @return レスポンス
	 */
	static TUniquePtr<FHttpServerResponse> MakeJsonResponse(const FString& Json);

	// ---- Phase 2: 並列外部制御 API（IHttpEnvServer オーバーライド） ----

	/**
	 * ActionQueue から Step リクエストを 1 件取り出す。
	 * ParallelSetupActor の Tick から呼ぶ（GameThread 専用）。
	 */
	virtual bool TakeStepRequest(
		TArray<float>& OutAction, int32& OutSteps, FHttpResultCallback& OutCallback) override;

	/**
	 * ResetQueue から Reset リクエストを 1 件取り出す。
	 */
	virtual bool TakeResetRequest(
		TOptional<int32>& OutSeed, FHttpResultCallback& OutCallback) override;

	/** Step 結果を JSON に変換して HTTP コールバックを呼ぶ */
	virtual void CompleteStep(FEnvStepResult Result, FHttpResultCallback Callback) override;

	/** Reset 結果を JSON に変換して HTTP コールバックを呼ぶ */
	virtual void CompleteReset(FEnvResetResult Result, FHttpResultCallback Callback) override;

protected:
	/** Step 結果を JSON 文字列に変換する（Tick() と CompleteStep() で共有） */
	static FString BuildStepJson(const FEnvStepResult& Result);

	/** Reset 結果を JSON 文字列に変換する（Tick() と CompleteReset() で共有） */
	static FString BuildResetJson(const FEnvResetResult& Result);

	// HTTP ハンドラ（ワーカースレッド）
	bool HandleReset(const FHttpServerRequest& Request, const FHttpResultCallback& OnComplete);
	bool HandleStep(const FHttpServerRequest& Request, const FHttpResultCallback& OnComplete);
	bool HandleClose(const FHttpServerRequest& Request, const FHttpResultCallback& OnComplete);

	/** 派生クラスが追加ルートを登録するフック（StartServer 末尾で呼ばれる） */
	virtual void RegisterAdditionalRoutes(TSharedPtr<IHttpRouter> Router) {}

	/** 派生クラスが追加ルートを解除するフック（StopServer 冒頭で呼ばれる） */
	virtual void UnregisterAdditionalRoutes(TSharedPtr<IHttpRouter> Router) {}

	/** Request.Body を null 終端保証付きで FString に変換する。 */
	static FString ParseBodyString(const FHttpServerRequest& Request);

private:
	TSharedPtr<IHttpRouter> HttpRouter;
	FHttpRouteHandle ResetRoute;
	FHttpRouteHandle StepRoute;
	FHttpRouteHandle CloseRoute;

	// HTTP スレッド → GameThread
	struct FResetRequest { TOptional<int32> Seed; FHttpResultCallback Callback; };
	struct FStepRequest  { TArray<float> Action; int32 Steps = 1; FHttpResultCallback Callback; };

	TQueue<FResetRequest, EQueueMode::Mpsc> ResetQueue;
	TQueue<FStepRequest,  EQueueMode::Mpsc> ActionQueue;

	// GameThread → HTTP スレッド
	TQueue<FEnvResetResult, EQueueMode::Spsc> ResetResultQueue;
	TQueue<FEnvStepResult,  EQueueMode::Spsc> StepResultQueue;

	// 保留中のコールバック（GameThread で結果が出たら呼ぶ）
	TOptional<FHttpResultCallback> PendingResetCallback;
	TOptional<FHttpResultCallback> PendingStepCallback;
};
