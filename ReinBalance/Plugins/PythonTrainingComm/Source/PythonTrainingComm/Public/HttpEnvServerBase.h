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

protected:
	// HTTP ハンドラ（ワーカースレッド）
	bool HandleReset(const FHttpServerRequest& Request, const FHttpResultCallback& OnComplete);
	bool HandleStep(const FHttpServerRequest& Request, const FHttpResultCallback& OnComplete);
	bool HandleClose(const FHttpServerRequest& Request, const FHttpResultCallback& OnComplete);

	/** 派生クラスが追加ルートを登録するフック（StartServer 末尾で呼ばれる） */
	virtual void RegisterAdditionalRoutes(TSharedPtr<IHttpRouter> Router) {}

	/** 派生クラスが追加ルートを解除するフック（StopServer 冒頭で呼ばれる） */
	virtual void UnregisterAdditionalRoutes(TSharedPtr<IHttpRouter> Router) {}

private:
	TSharedPtr<IHttpRouter> HttpRouter;
	FHttpRouteHandle ResetRoute;
	FHttpRouteHandle StepRoute;
	FHttpRouteHandle CloseRoute;

	// HTTP スレッド → GameThread
	struct FResetRequest { TOptional<int32> Seed; FHttpResultCallback Callback; };
	struct FStepRequest  { TArray<float> Action; FHttpResultCallback Callback; };

	TQueue<FResetRequest, EQueueMode::Mpsc> ResetQueue;
	TQueue<FStepRequest,  EQueueMode::Mpsc> ActionQueue;

	// GameThread → HTTP スレッド
	TQueue<FEnvResetResult, EQueueMode::Spsc> ResetResultQueue;
	TQueue<FEnvStepResult,  EQueueMode::Spsc> StepResultQueue;

	// 保留中のコールバック（GameThread で結果が出たら呼ぶ）
	TOptional<FHttpResultCallback> PendingResetCallback;
	TOptional<FHttpResultCallback> PendingStepCallback;

	static TUniquePtr<FHttpServerResponse> MakeJsonResponse(const FString& Json);
};
