#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "CoinGame.generated.h"

/**
 * コイン収集 + 敵回避ゲームの物理・ゲームロジックアクター（ビジュアルなし）。
 *
 * 2D XY 平面上でプレイヤーがコインを集めながら敵を避ける純粋シミュレーター。
 * ビジュアル表示は ACoinGameView が担う。
 *
 * 行動: 離散5方向 (0=+Y, 1=-Y, 2=-X, 3=+X, 4=静止)
 * 観測: GetObsDim() 次元 = 11 + NumCoinObs*2 + MaxEnemyObs*5
 *   [0-1]                  プレイヤー位置 (x,y) / FieldHalfSize
 *   [2-3]                  プレイヤー速度 (vx,vy)
 *   [4-7]                  壁距離 (上/下/左/右) / FieldHalfSize
 *   [8]                    現在の敵数 / MaxEnemyObs
 *   [9]                    エピソード内コイン収集累計数 / NumCoins
 *   [10]                   次スポーンまでの残り時間 (0~1)
 *   [11 .. 11+N*2-1]       コイン相対位置 dx,dy × NumCoinObs  / (FieldHalfSize*2)
 *   [11+N*2 .. +M*2-1]     敵相対位置  dx,dy × MaxEnemyObs=20 / (FieldHalfSize*2)
 *   [.. +M*2]              敵速度      vx,vy × MaxEnemyObs=20
 *   [.. +M]                敵の種類スカラー × MaxEnemyObs=20  (A=0.0, B=0.5, C=1.0)
 * ※ NumCoinObs を変更すると観測次元が変わるため、モデルの再訓練が必要
 */
/** 観測ベクトルの各セグメントを表す構造体。GetObsSchema() で取得する。 */
struct FObsSegment
{
	FString Name;
	int32   Dim;
};

UCLASS()
class REINBALANCE_API ACoinGame : public AActor
{
	GENERATED_BODY()

public:
	ACoinGame();

	/** 離散行動 (0〜4) を受けて 1 物理ステップ進める。GameThread から呼ぶ。 */
	void PhysicsStep(int32 ActionIdx);

	/** 状態をリセットする。Seed 未指定の場合はランダム。 */
	void ResetState(TOptional<int32> Seed);

	/** 観測ベクトルを返す（次元数は GetObsDim() に一致） */
	TArray<float> GetObservation() const;

	/** 観測ベクトルのセグメント定義を返す。/obs_schema エンドポイントや検証に使用。 */
	TArray<FObsSegment> GetObsSchema() const;

	/** 観測次元数を返す: 11 + NumCoinObs*2 + MaxEnemyObs*5 */
	int32 GetObsDim() const { return 11 + NumCoinObs * 2 + MaxEnemyObs * 5; }

	/**
	 * 観測スキーマのハッシュ文字列を返す。
	 * obs次元に影響するパラメータ（NumCoinObs, MaxEnemyObs）から生成する。
	 * NumCoins はフィールド上の枚数のみに影響するためハッシュに含めない。
	 */
	FString GetObsSchemaHash() const;

	/** ステップ報酬を返す */
	float GetReward() const;

	/** エピソード終了判定（敵接触） */
	bool IsDone() const;

	// ---- ビュー向け状態アクセサー ----

	FVector2D GetPlayerPos() const { return PlayerPos; }
	FVector2D GetPlayerVel() const { return PlayerVel; }

	int32     GetCoinCount()      const { return CoinPositions.Num(); }
	FVector2D GetCoinPos(int32 i) const
	{
		return CoinPositions.IsValidIndex(i) ? CoinPositions[i] : FVector2D::ZeroVector;
	}

	int32     GetEnemyCount()       const { return Enemies.Num(); }
	FVector2D GetEnemyPos(int32 i)  const
	{
		return Enemies.IsValidIndex(i) ? Enemies[i].Pos : FVector2D::ZeroVector;
	}
	FVector2D GetEnemyVel(int32 i)  const
	{
		return Enemies.IsValidIndex(i) ? Enemies[i].Vel : FVector2D::ZeroVector;
	}
	int32     GetEnemyType(int32 i) const
	{
		return Enemies.IsValidIndex(i) ? Enemies[i].Type : 0;
	}
	
	// ---- 報酬設定 ----
	
	/** 生存報酬 */
	UPROPERTY(EditAnywhere, Category = "CoinGame|Train")
	float AliveReward = 0.001f;
	
	/** コイン獲得報酬 */
	UPROPERTY(EditAnywhere, Category = "CoinGame|Train")
	float CoinReward = 5.0f;

	// ---- フィールド設定 ----

	/** フィールド半幅 [m]（正方形 -Half〜+Half） */
	UPROPERTY(EditAnywhere, Category = "CoinGame|Config")
	float FieldHalfSize = 10.f;

	/** フィールド上のコイン枚数（上限なし） */
	UPROPERTY(EditAnywhere, Category = "CoinGame|Config")
	int32 NumCoins = 3;

	/** 敵スポーン間隔 [秒] */
	UPROPERTY(EditAnywhere, Category = "CoinGame|Config")
	float EnemySpawnInterval = 10.f;

	// ---- プレイヤー物理 ----

	/** 入力1方向あたりの加速度 [m/s²] */
	UPROPERTY(EditAnywhere, Category = "CoinGame|Physics")
	float PlayerAccel = 5.f;

	/** 線形ドラッグ係数（終端速度 = Accel / Drag） */
	UPROPERTY(EditAnywhere, Category = "CoinGame|Physics")
	float PlayerDrag = 2.f;

	// ---- コイン・敵 ----

	UPROPERTY(EditAnywhere, Category = "CoinGame|Physics")
	float CoinCollectRadius = 1.0f;

	UPROPERTY(EditAnywhere, Category = "CoinGame|Physics")
	float EnemyCollisionRadius = 0.6f;

	/** 敵タイプA の移動速度 [m/s]（遅い直進追跡） */
	UPROPERTY(EditAnywhere, Category = "CoinGame|Physics")
	float EnemySpeedA = 1.0f;

	/** 敵タイプB の移動速度 [m/s]（速い直進追跡） */
	UPROPERTY(EditAnywhere, Category = "CoinGame|Physics")
	float EnemySpeedB = 2.5f;

	/** 敵タイプC の移動速度 [m/s]（予測追跡） */
	UPROPERTY(EditAnywhere, Category = "CoinGame|Physics")
	float EnemySpeedC = 1.5f;

	/** タイプC の先読み時間 [秒] */
	UPROPERTY(EditAnywhere, Category = "CoinGame|Physics")
	float EnemyPredictTime = 0.75f;

protected:
	virtual void BeginPlay() override;

private:
	// 観測に含めるコインの数
	static constexpr int32 NumCoinObs = 20;
	
	// 最大敵数
	static constexpr int32 MaxEnemyObs = 20;
	
	// フレームレート
	static constexpr float PhysicsDt   = 1.f / 60.f;

	struct FEnemyState
	{
		FVector2D Pos;
		FVector2D Vel;
		int32     Type; // 0=A, 1=B, 2=C
	};

	FVector2D           PlayerPos;
	FVector2D           PlayerVel;
	TArray<FVector2D>   CoinPositions;
	TArray<FEnemyState> Enemies;

	float SpawnTimer     = 0.f;
	float LastReward     = 0.f;
	bool  bDone          = false;
	int32 CoinsCollected = 0;

	FRandomStream RandStream;

	FVector2D RandomInsideField();
	FVector2D RandomOnEdge();
	void      SpawnEnemy();
	void      UpdateEnemies();
	void      CheckCoinCollections();
	bool      CheckEnemyCollisions() const;
	void      ClampPlayerToField();
};
