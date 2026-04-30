#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "SurvivorsGame.generated.h"

/** obs スキーマの1セグメント */
struct FSurvivorsObsSegment
{
	FString Name;
	int32   Dim;
};

/**
 * 武器の種類。将来アイテム取得で追加される。
 * MaxWeaponTypeSlots (=8) 分の容量を obs エンコードに確保済み。
 */
UENUM(BlueprintType)
enum class EWeaponType : uint8
{
	None     = 0,
	Aura     = 1,
	Whip     = 2,
	Fireball = 3,
};

/** 武器スロット1つ分のデータ */
struct FWeaponSlot
{
	EWeaponType Type  = EWeaponType::None;
	int32       Level = 0; // 0=未装備, 1-8=レベル
};

/**
 * Vampire Survivors 風サバイバルゲームのロジッククラス（ビジュアルなし）。
 *
 * 2D XY 平面上でプレイヤーが敵を倒しながらアイテムを集めて生き延びる。
 * ビジュアル表示は別クラスが担う（未実装）。
 *
 * 行動: 離散5方向 (0=+Y, 1=-Y, 2=-X, 3=+X, 4=静止)
 * 観測: GetObsDim() 次元 = 23 + NumItemObs*2 + MaxEnemyObs*6
 *   [0-1]    プレイヤー位置 (x,y) / FieldHalfSize
 *   [2-3]    プレイヤー速度 (vx,vy)
 *   [4-11]   8方向レイキャスト壁距離 (0~1)
 *   [12]     プレイヤー HP / MaxPlayerHP
 *   [13-18]  武器スロット × 3: (type_norm, level_norm) × MaxWeaponSlots
 *   [19]     現在の敵数 / MaxEnemyObs
 *   [20]     次スポーンまでの残り時間 (0~1)
 *   [21]     xp_progress (0~1, Phase1=0 固定)
 *   [22]     player_level (0~1, Phase1=0 固定)
 *   [23 .. 23+N*2-1]       アイテム相対位置 dx,dy × NumItemObs
 *   [23+N*2 .. +M*6-1]     敵情報 (dx,dy,vx,vy,type_norm,hp_norm) × MaxEnemyObs
 */
UCLASS()
class REINBALANCE_API ASurvivorsGame : public AActor
{
	GENERATED_BODY()

public:
	ASurvivorsGame();

	/** 離散行動 (0〜4) を受けて 1 物理ステップ進める */
	void PhysicsStep(int32 ActionIdx);

	/** 状態をリセット */
	void ResetState(TOptional<int32> Seed);

	/** 観測ベクトルを返す */
	TArray<float> GetObservation() const;

	/** 観測スキーマを返す */
	TArray<FSurvivorsObsSegment> GetObsSchema() const;

	/** 観測次元数: 23 + NumItemObs*2 + MaxEnemyObs*6 */
	int32 GetObsDim() const { return 23 + NumItemObs * 2 + MaxEnemyObs * 6; }

	/** obs 次元に影響するパラメータから生成するハッシュ */
	FString GetObsSchemaHash() const;

	/** ステップ報酬を返す */
	float GetReward() const;

	/** エピソード終了判定（HP <= 0） */
	bool IsDone() const;

	// ---- ビュー / デバッグ向けアクセサー ----

	FVector2D GetPlayerPos()   const { return PlayerPos; }
	FVector2D GetPlayerVel()   const { return PlayerVel; }
	float     GetPlayerHP()    const { return PlayerHP; }
	float     GetMaxPlayerHP() const { return MaxPlayerHP; }

	int32     GetItemCount()       const { return ItemPositions.Num(); }
	FVector2D GetItemPos(int32 i)  const
	{
		return ItemPositions.IsValidIndex(i) ? ItemPositions[i] : FVector2D::ZeroVector;
	}

	int32     GetEnemyCount()      const { return Enemies.Num(); }
	FVector2D GetEnemyPos(int32 i) const
	{
		return Enemies.IsValidIndex(i) ? Enemies[i].Pos : FVector2D::ZeroVector;
	}

	// ---- 報酬設定 ----

	/** 生存報酬 (毎ステップ) */
	UPROPERTY(EditAnywhere, Category = "Survivors|Train")
	float AliveReward = 0.005f;

	/** アイテム獲得報酬 */
	UPROPERTY(EditAnywhere, Category = "Survivors|Train")
	float ItemReward = 3.0f;

	/** 敵撃破ボーナス */
	UPROPERTY(EditAnywhere, Category = "Survivors|Train")
	float KillReward = 2.0f;

	// ---- フィールド設定 ----

	/** フィールド半幅 [m] */
	UPROPERTY(EditAnywhere, Category = "Survivors|Config")
	float FieldHalfSize = 15.f;

	/** フィールド上のアイテム数 */
	UPROPERTY(EditAnywhere, Category = "Survivors|Config")
	int32 NumItems = 5;

	/** 敵スポーン間隔 [秒]（/params で上書き可能） */
	UPROPERTY(EditAnywhere, Category = "Survivors|Config")
	float EnemySpawnInterval = 8.f;

	/** 同時に存在できる最大敵数（カリキュラム制御用, /params で上書き可能） */
	UPROPERTY(EditAnywhere, Category = "Survivors|Config")
	int32 MaxActiveEnemies = 3;

	/** 敵速度グローバル倍率（カリキュラム制御用, /params で上書き可能） */
	UPROPERTY(EditAnywhere, Category = "Survivors|Config")
	float EnemySpeedMult = 1.0f;

	// ---- プレイヤー ----

	/** プレイヤー最大 HP */
	UPROPERTY(EditAnywhere, Category = "Survivors|Player")
	float MaxPlayerHP = 100.f;

	/** 入力1方向あたりの加速度 [m/s²] */
	UPROPERTY(EditAnywhere, Category = "Survivors|Player")
	float PlayerAccel = 6.f;

	/** 線形ドラッグ係数 */
	UPROPERTY(EditAnywhere, Category = "Survivors|Player")
	float PlayerDrag = 2.f;

	// ---- 武器 (オーラ) ----

	/** オーラ攻撃半径 [m] */
	UPROPERTY(EditAnywhere, Category = "Survivors|Weapon")
	float AuraRadius = 1.5f;

	/** オーラ攻撃 DPS（秒あたりダメージ）: /tick = AuraDPS * PhysicsDt */
	UPROPERTY(EditAnywhere, Category = "Survivors|Weapon")
	float AuraDPS = 15.f;

	// ---- 敵設定 ----

	/** タイプA (低速追跡 Slime) の基本速度 [m/s] */
	UPROPERTY(EditAnywhere, Category = "Survivors|Enemy")
	float EnemySpeedA = 1.0f;

	/** タイプA の接触 DPS */
	UPROPERTY(EditAnywhere, Category = "Survivors|Enemy")
	float EnemyDamageA = 5.f;

	/** タイプA の HP */
	UPROPERTY(EditAnywhere, Category = "Survivors|Enemy")
	float EnemyHPA = 20.f;

	/** タイプB (高速直進 Zombie) の基本速度 [m/s] */
	UPROPERTY(EditAnywhere, Category = "Survivors|Enemy")
	float EnemySpeedB = 2.5f;

	/** タイプB の接触 DPS */
	UPROPERTY(EditAnywhere, Category = "Survivors|Enemy")
	float EnemyDamageB = 10.f;

	/** タイプB の HP */
	UPROPERTY(EditAnywhere, Category = "Survivors|Enemy")
	float EnemyHPB = 50.f;

	/** タイプC (予測追跡 Ghost) の基本速度 [m/s] */
	UPROPERTY(EditAnywhere, Category = "Survivors|Enemy")
	float EnemySpeedC = 1.5f;

	/** タイプC の接触 DPS */
	UPROPERTY(EditAnywhere, Category = "Survivors|Enemy")
	float EnemyDamageC = 8.f;

	/** タイプC の HP */
	UPROPERTY(EditAnywhere, Category = "Survivors|Enemy")
	float EnemyHPC = 30.f;

	/** タイプC の先読み時間 [秒] */
	UPROPERTY(EditAnywhere, Category = "Survivors|Enemy")
	float EnemyPredictTime = 0.75f;

	/** 敵との接触判定半径 [m] */
	UPROPERTY(EditAnywhere, Category = "Survivors|Enemy")
	float EnemyCollisionRadius = 0.6f;

	// ---- アイテム ----

	/** アイテム収集半径 [m] */
	UPROPERTY(EditAnywhere, Category = "Survivors|Item")
	float ItemCollectRadius = 1.0f;

protected:
	virtual void BeginPlay() override;

private:
	// ---- 定数 ----
	static constexpr int32  NumItemObs       = 20;
	static constexpr int32  MaxEnemyObs      = 20;
	static constexpr int32  MaxWeaponSlots   = 3;
	static constexpr int32  MaxWeaponTypeSlots = 8; // obs エンコードの型容量
	static constexpr int32  MaxWeaponLevel   = 8;
	static constexpr float  PhysicsDt        = 1.f / 60.f;
	static const FVector2D  RayDirs[8];              // 8方向レイキャスト方向

	// ---- 敵データ ----
	struct FEnemyState
	{
		FVector2D Pos;
		FVector2D Vel;
		int32     Type;  // 0=A, 1=B, 2=C
		float     HP;
		float     MaxHP;
	};

	// ---- 状態 ----
	FVector2D             PlayerPos;
	FVector2D             PlayerVel;
	float                 PlayerHP    = 100.f;
	FWeaponSlot           WeaponSlots[MaxWeaponSlots];
	TArray<FVector2D>     ItemPositions;
	TArray<FEnemyState>   Enemies;
	float                 SpawnTimer  = 0.f;
	float                 LastReward  = 0.f;
	bool                  bDone       = false;
	FRandomStream         RandStream;

	// ---- 内部メソッド ----
	FVector2D RandomInsideField();
	FVector2D RandomOnEdge();
	void      SpawnEnemy();
	void      UpdateEnemies();
	void      ApplyAuraDamage();
	void      CheckItemCollections();
	void      ApplyEnemyContactDamage();
	void      ClampPlayerToField();
	float     CastRayToBoundary(FVector2D Origin, FVector2D Dir) const;

	// 敵タイプ別パラメータ取得
	float GetEnemySpeed(int32 Type) const;
	float GetEnemyDamagePerTick(int32 Type) const;
	float GetEnemyMaxHP(int32 Type) const;
};
