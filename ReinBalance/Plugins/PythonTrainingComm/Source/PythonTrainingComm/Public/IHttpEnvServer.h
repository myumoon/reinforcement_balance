#pragma once

#include "CoreMinimal.h"

/**
 * Step/Reset の結果を格納する構造体。
 * HTTP スレッドと GameThread 間で TQueue 経由でやり取りする。
 */
struct FEnvStepResult
{
	TArray<float> Obs;
	float Reward = 0.f;
	bool bDone = false;
	bool bTruncated = false;
};

struct FEnvResetResult
{
	TArray<float> Obs;
	FString       ObsSchemaHash; // 観測スキーマのハッシュ（Python 側の不一致検出用）
};

/**
 * Python 訓練スクリプトとの HTTP 通信を担う環境サーバーの抽象インターフェース。
 *
 * 実装クラスの責務:
 *   - ActionQueue  : HTTP スレッドが書き込み、GameThread が読み出す
 *   - StepResultQueue : GameThread が書き込み、HTTP スレッドが読み出す
 *   - ResetResultQueue: 同上
 *
 * Tick() を GameThread から毎フレーム呼び出すことで同期が成立する。
 */
class PYTHONTRAININGCOMM_API IHttpEnvServer
{
public:
	virtual ~IHttpEnvServer() = default;

	/** HTTP サーバーを起動する。BeginPlay から呼ぶ。 */
	virtual void StartServer(uint32 Port) = 0;

	/** HTTP サーバーを停止する。EndPlay から呼ぶ。 */
	virtual void StopServer() = 0;

	/**
	 * GameThread から毎 Tick 呼び出す。
	 * ActionQueue にアクションが積まれていれば ProcessStep を呼び出し、
	 * 結果を StepResultQueue に書き込む。
	 * ResetQueue に reset が積まれていれば ProcessReset を呼び出す。
	 */
	virtual void Tick() = 0;

	/** アクション到着時にゲーム固有の物理操作を行い観測値を返す。GameThread で実行される。 */
	virtual FEnvStepResult ProcessStep(const TArray<float>& Action) = 0;

	/** リセット時にゲーム固有の初期化を行い初期観測値を返す。GameThread で実行される。 */
	virtual FEnvResetResult ProcessReset(TOptional<int32> Seed) = 0;
};
