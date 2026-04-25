#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Components/StaticMeshComponent.h"
#include "CoinGame.generated.h"

/**
 * コイン収集 + 敵回避ゲームの物理・ゲームロジックアクター。
 *
 * 2D XY 平面上でプレイヤーがコインを集めながら敵を避ける。
 * UE5 物理エンジンは使用せず、解析的に積分する。
 *
 * 行動: 離散5方向 (0=+Y, 1=-Y, 2=-X, 3=+X, 4=静止)
 * 観測: 116次元
 *   [0-1]   プレイヤー位置 (x,y) / FieldHalfSize
 *   [2-3]   プレイヤー速度 (vx,vy)
 *   [4-7]   壁距離 (上/下/左/右) / FieldHalfSize
 *   [8]     現在の敵数 / MaxEnemyObs
 *   [9]     次スポーンまでの残り時間 (0~1)
 *   [10-15] コイン相対位置 dx,dy × 3  / (FieldHalfSize*2)
 *   [16-55] 敵相対位置  dx,dy × 20   / (FieldHalfSize*2)
 *   [56-95] 敵速度      vx,vy × 20
 *   [96-115] 敵の種類スカラー × 20  (A=0.0, B=0.5, C=1.0)
 */
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

	/** 観測値 116 次元を返す */
	TArray<float> GetObservation() const;

	/** ステップ報酬を返す */
	float GetReward() const;

	/** エピソード終了判定（敵接触） */
	bool IsDone() const;

	// ---- フィールド設定 ----

	/** フィールド半幅 [m]（正方形 -Half〜+Half） */
	UPROPERTY(EditAnywhere, Category = "CoinGame|Config")
	float FieldHalfSize = 10.f;

	/** コインの枚数（観測次元に影響するため 3 を推奨） */
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

	// ---- ビジュアル ----

	/** シミュレーション座標（m）→ UE5 単位（cm）の変換スケール
	 *  デフォルト 50: FieldHalfSize=10m → 視覚フィールド 1000×1000 UE単位 */
	UPROPERTY(EditAnywhere, Category = "CoinGame|Visual")
	float SimToUE = 50.f;

protected:
	virtual void BeginPlay() override;

private:
	static constexpr int32 MaxEnemyObs = 20;
	static constexpr int32 NumCoinObs  = 3;
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

	float SpawnTimer = 0.f;
	float LastReward = 0.f;
	bool  bDone      = false;

	FRandomStream RandStream;

	// ---- ビジュアルコンポーネント ----

	UPROPERTY(VisibleAnywhere, Category = "Components")
	TObjectPtr<UStaticMeshComponent> PlayerMesh;

	UPROPERTY(VisibleAnywhere, Category = "Components")
	TArray<TObjectPtr<UStaticMeshComponent>> CoinMeshComponents;

	/** 動的に生成・破棄される敵メッシュ（GC 防止のため UPROPERTY） */
	UPROPERTY()
	TArray<TObjectPtr<UStaticMeshComponent>> EnemyMeshComponents;

	/** キャッシュされたアセット参照 */
	UPROPERTY()
	TObjectPtr<UStaticMesh> ConeMeshAsset;
	UPROPERTY()
	TObjectPtr<UStaticMesh> SphereMeshAsset;
	UPROPERTY()
	TObjectPtr<UStaticMesh> CubeMeshAsset;
	UPROPERTY()
	TObjectPtr<UMaterial> BaseMaterialAsset;

	void SetupVisuals();
	void UpdateVisuals();
	UStaticMeshComponent* CreateEnemyVisual(int32 EnemyIndex, int32 Type);
	class UMaterialInstanceDynamic* CreateColorMaterial(const FLinearColor& Color);

	// ---- シミュレーション内部 ----

	FVector2D RandomInsideField();
	FVector2D RandomOnEdge();
	void      SpawnEnemy();
	void      UpdateEnemies();
	void      CheckCoinCollections();
	bool      CheckEnemyCollisions() const;
	void      ClampPlayerToField();
};
