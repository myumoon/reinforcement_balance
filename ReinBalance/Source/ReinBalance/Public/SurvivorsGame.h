#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "SurvivorsGame.generated.h"

class AWallActor;

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

/** 経験値ジェムの種類（青=1XP / 緑=5XP / 赤=10XP） */
UENUM(BlueprintType)
enum class EGemType : uint8
{
	Blue  = 0, // 雑魚（Bat 系）ドロップ, xp=1
	Green = 1, // 中型敵ドロップ, xp=5
	Red   = 2, // ボスドロップ, xp=10
};

/**
 * 敵1種のパラメーター。EnemyTypeTable に格納し Details パネルやカリキュラムから変更可能。
 * knockback_resistance / is_boss は Plan05 / Plan06 で使用。
 */
USTRUCT(BlueprintType)
struct FEnemyTypeParams
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadOnly)
	FString Name;

	/** 出現直後 HP（時間スケーリング前、Plan08 で乗算） */
	UPROPERTY(EditAnywhere, BlueprintReadOnly)
	float BaseHP = 1.f;

	/** 内部移動速度 [u/s] */
	UPROPERTY(EditAnywhere, BlueprintReadOnly)
	float Speed = 50.f;

	/** 接触 1 ヒットあたりのダメージ（Plan04 で hit_interval と組み合わせ） */
	UPROPERTY(EditAnywhere, BlueprintReadOnly)
	float ContactDamage = 5.f;

	/** 衝突判定半径 [u] */
	UPROPERTY(EditAnywhere, BlueprintReadOnly)
	float CollisionRadius = 10.f;

	/** ノックバック耐性（0=なし, 1=完全耐性）。Plan05 で参照。 */
	UPROPERTY(EditAnywhere, BlueprintReadOnly)
	float KnockbackResistance = 0.f;

	/** ボス扱い（ドロップ・耐性に影響）。Plan06 で参照。 */
	UPROPERTY(EditAnywhere, BlueprintReadOnly)
	bool bIsBoss = false;
};

/** ウェーブ内の敵種別と出現重み */
USTRUCT(BlueprintType)
struct FEnemySpawnWeight
{
	GENERATED_BODY()

	/** EnemyTypeTable のインデックス（type_id） */
	UPROPERTY(EditAnywhere, BlueprintReadOnly)
	int32 TypeId = 0;

	/** 選択重み（相対値、正規化は内部で行う） */
	UPROPERTY(EditAnywhere, BlueprintReadOnly)
	float Weight = 1.f;
};

/** 時刻帯（Wave）ごとのスポーン設定 */
USTRUCT(BlueprintType)
struct FSpawnWave
{
	GENERATED_BODY()

	/** Wave 開始時刻 [秒] */
	UPROPERTY(EditAnywhere, BlueprintReadOnly)
	float TimeStart = 0.f;

	/** Wave 終了時刻 [秒] */
	UPROPERTY(EditAnywhere, BlueprintReadOnly)
	float TimeEnd = 30.f;

	/** スポーンレート [体/秒]（SpawnRateMult で乗算） */
	UPROPERTY(EditAnywhere, BlueprintReadOnly)
	float SpawnRate = 1.f;

	/** この Wave の同時出現上限（MaxActiveEnemies との min を取る） */
	UPROPERTY(EditAnywhere, BlueprintReadOnly)
	int32 MaxEnemies = 50;

	/** 出現する敵種別と重み */
	UPROPERTY(EditAnywhere, BlueprintReadOnly)
	TArray<FEnemySpawnWeight> EnemyWeights;
};

/**
 * Vampire Survivors 風サバイバルゲームのロジッククラス（ビジュアルなし）。
 *
 * 2D XY 平面上でプレイヤーが敵を倒しながらアイテムを集めて生き延びる。
 * ビジュアル表示は別クラスが担う（未実装）。
 *
 * 行動: 離散5方向 (0=+Y, 1=-Y, 2=-X, 3=+X, 4=静止)
 * 観測: GetObsDim() 次元 = 23 + NumGemObs*2 + MaxEnemyObs*6
 *   [0-1]    プレイヤー位置 (x,y) / FieldHalfSize
 *   [2-3]    プレイヤー速度 (vx,vy) / MoveSpeed（-1〜1 に正規化）
 *   [4-11]   8方向レイキャスト壁距離 (0~1)
 *   [12]     プレイヤー HP / MaxPlayerHP
 *   [13-18]  武器スロット × 3: (type_norm, level_norm) × MaxWeaponSlots
 *   [19]     現在の敵数 / MaxEnemyObs
 *   [20]     経過時間 elapsed_time / MaxGameTime (0~1)
 *   [21]     xp_progress (0~1, ジェム回収・レベルアップでリセット)
 *   [22]     player_level (0~1 = level / MaxPlayerLevel, 最大 MaxPlayerLevel=100)
 *   [23 .. 23+N*2-1]       ジェム相対位置 dx,dy × NumGemObs（近い順）
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

	/** 観測次元数: 23 + NumGemObs*2 + MaxEnemyObs*6 */
	int32 GetObsDim() const { return 23 + NumGemObs * 2 + MaxEnemyObs * 6; }

	/** obs 次元に影響するパラメータから生成するハッシュ */
	FString GetObsSchemaHash() const;

	/** ステップ報酬を返す */
	float GetReward() const;

	/** エピソード終了判定（HP <= 0） */
	bool IsDone() const;

	// ---- ビュー / デバッグ向けアクセサー ----

	UFUNCTION(BlueprintPure, Category = "Survivors|Config")
	FVector2D GetPlayerPos()   const { return PlayerPos; }
	
	FVector2D GetPlayerVel()   const { return PlayerVel; }
	float     GetPlayerHP()    const { return PlayerHP; }
	float     GetMaxPlayerHP() const { return MaxPlayerHP; }
	float     GetAuraSize()    const { return AuraRadius; }

	// View との互換維持のため GetItemCount/GetItemPos はジェムデータを返す
	int32     GetItemCount()       const { return Gems.Num(); }
	FVector2D GetItemPos(int32 i)  const;
	EGemType  GetItemGemType(int32 i) const;

	int32     GetEnemyCount()         const { return Enemies.Num(); }
	FVector2D GetEnemyPos(int32 i)    const
	{
		return Enemies.IsValidIndex(i) ? Enemies[i].Pos   : FVector2D::ZeroVector;
	}
	int32     GetEnemyType(int32 i)   const { return Enemies.IsValidIndex(i) ? Enemies[i].TypeId : 0; }
	float     GetEnemyHP(int32 i)     const { return Enemies.IsValidIndex(i) ? Enemies[i].HP    : 0.f; }
	float     GetEnemyMaxHP(int32 i)  const { return Enemies.IsValidIndex(i) ? Enemies[i].MaxHP : 1.f; }

	// ---- 報酬設定 ----

	/** 生存報酬 (毎ステップ) */
	UPROPERTY(EditAnywhere, Category = "Survivors|Train")
	float AliveReward = 0.001f;

	/** ジェム回収報酬（種類問わず固定） */
	UPROPERTY(EditAnywhere, Category = "Survivors|Train")
	float ItemReward = 1.0f;

	/** 敵撃破ボーナス */
	UPROPERTY(EditAnywhere, Category = "Survivors|Train")
	float KillReward = 2.0f;

	// ---- フィールド設定 ----

	/** フィールド半幅 [u]。敵/アイテムスポーン範囲・obs 正規化基準として使用。外側境界は AWallActor で定義する。 */
	UPROPERTY(EditAnywhere, Category = "Survivors|Config")
	float FieldHalfSize = 1000.f;

	/** シム座標(u) ↔ UE5 単位 変換スケール。SurvivorsGameView の SimToUE と一致させること。 */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Survivors|Config")
	float SimToUE = 5.f;

	/** 同時に存在できる最大敵数の上限（カリキュラム制御用, /params で上書き可能）。
	 *  実効値 = min(CurrentWave.MaxEnemies, MaxActiveEnemies)。 */
	UPROPERTY(EditAnywhere, Category = "Survivors|Config")
	int32 MaxActiveEnemies = 6;

	/** スポーンレートグローバル倍率（カリキュラム制御用, /params で上書き可能） */
	UPROPERTY(EditAnywhere, Category = "Survivors|Config")
	float SpawnRateMult = 1.0f;

	// ---- スポーン設定 ----

	/** 円周スポーン: プレイヤーからの最小距離 [u] */
	UPROPERTY(EditAnywhere, Category = "Survivors|Spawn")
	float SpawnMinDistance = 400.f;

	/** 円周スポーン: プレイヤーからの最大距離 [u] */
	UPROPERTY(EditAnywhere, Category = "Survivors|Spawn")
	float SpawnMaxDistance = 600.f;

	/** GiantBat ボスのスポーン時刻 [秒] */
	UPROPERTY(EditAnywhere, Category = "Survivors|Spawn")
	float BossSpawnTime = 600.f;

	/**
	 * 時刻帯別スポーン設定テーブル（Mad Forest 10 Wave）。
	 * TimeStart/TimeEnd は秒。各 Wave の SpawnRate は SpawnRateMult で乗算される。
	 */
	UPROPERTY(EditAnywhere, Category = "Survivors|Spawn")
	TArray<FSpawnWave> SpawnWaves;

	/** 敵速度グローバル倍率（カリキュラム制御用, /params で上書き可能） */
	UPROPERTY(EditAnywhere, Category = "Survivors|Config")
	float EnemySpeedMult = 1.0f;

	// ---- プレイヤー ----

	/** プレイヤー最大 HP（Poe Ratcho） */
	UPROPERTY(EditAnywhere, Category = "Survivors|Player")
	float MaxPlayerHP = 70.f;

	/** プレイヤー移動速度 [u/s]（直接速度モデル, カリキュラム制御可） */
	UPROPERTY(EditAnywhere, Category = "Survivors|Player")
	float MoveSpeed = 80.f;

	/** プレイヤー衝突半径 [u]（AWallActor との押し出し計算に使用） */
	UPROPERTY(EditAnywhere, Category = "Survivors|Player")
	float PlayerRadius = 10.f;

	// ---- 武器 (Garlic オーラ) ----
	// レベル別ステータスは GarlicTable[] (SurvivorsGame.cpp) で管理。
	// Lv1: damage=5, hit_interval=1.30s, area_radius=80u
	// Lv8: damage=20, hit_interval=0.95s, area_radius=185u

	// ---- 敵設定 ----

	/**
	 * 敵11種のパラメーターテーブル（Mad Forest 準拠）。
	 * type_id = インデックス: 0=Bat, 1=Zombie, ..., 10=GiantBat。
	 * カリキュラム用に EditAnywhere・/params エンドポイントで変更可能。
	 */
	UPROPERTY(EditAnywhere, Category = "Survivors|Enemy")
	TArray<FEnemyTypeParams> EnemyTypeTable;

	// ---- ジェム ----

	/** ジェム自動回収半径 [u]（pickup_radius: 仕様 §4.1） */
	UPROPERTY(EditAnywhere, Category = "Survivors|Item")
	float GemPickupRadius = 30.f;

	// ---- 時間スケーリング（仕様: enemies.md §1.1）----

	/** true = 時間経過で敵 HP/ダメージが増加。カリキュラム Phase A では false に設定 */
	UPROPERTY(EditAnywhere, Category = "Survivors|TimeScaling")
	bool bTimeScalingEnabled = true;

	/** HP 増加率 [割合/分]（デフォルト +10%/min、仕様 ★★） */
	UPROPERTY(EditAnywhere, Category = "Survivors|TimeScaling")
	float HPScaleRatePerMin = 0.10f;

	/** 接触ダメージ増加率 [割合/分]（デフォルト +5%/min、仕様 ★） */
	UPROPERTY(EditAnywhere, Category = "Survivors|TimeScaling")
	float DamageScaleRatePerMin = 0.05f;

protected:
	virtual void BeginPlay() override;

private:
	// ---- 定数 ----
	static constexpr int32  NumGemObs          = 20; // obs に含める近傍ジェム数
	static constexpr int32  MaxEnemyObs        = 20;
	static constexpr int32  MaxWeaponSlots     = 3;
	static constexpr int32  MaxWeaponTypeSlots = 8;
	static constexpr int32  MaxWeaponLevel     = 8;
	static constexpr int32  MaxPlayerLevel     = 100; // obs 正規化基準（ハードキャップではない）
	static constexpr float  PhysicsDt          = 1.f / 60.f;
	static constexpr float  MaxGameTime        = 1800.f;
	static constexpr float  ContactHitInterval = 0.5f;  // 敵→プレイヤー接触無敵 [s]
	static const FVector2D  RayDirs[8];

	// ---- 敵データ ----
	struct FEnemyState
	{
		FVector2D Pos;
		FVector2D Vel;
		int32     TypeId;              // 0〜10: EnemyTypeTable のインデックス
		float     HP;
		float     MaxHP;
		float     CollisionRadius;     // スポーン時に EnemyTypeTable からコピー
		float     ContactDamage;       // スポーン時に時間スケーリング済みの接触ダメージ
		float     GarlicLastHitTime;   // 最後に Garlic ヒットを受けた時刻 [s]
		float     PlayerLastHitTime;   // 最後にプレイヤーに接触ダメージを与えた時刻 [s]
	};

	// ---- ジェムデータ ----
	struct FGemState
	{
		FVector2D Pos;
		EGemType  Type;
	};

	// ---- 状態 ----
	FVector2D             PlayerPos;
	FVector2D             PlayerVel;
	float                 PlayerHP    = 100.f;
	float                 PlayerXP    = 0.f;
	int32                 PlayerLevel = 0;
	float                 AuraRadius  = 80.0f; // キャッシュ: GetAuraSize() / View 用
	FWeaponSlot           WeaponSlots[MaxWeaponSlots];
	TArray<FGemState>     Gems;
	TArray<FEnemyState>   Enemies;
	float                 ElapsedTime      = 0.f;
	float                 SpawnAccumulator = 0.f;
	bool                  bBossSpawned     = false;
	float                 LastReward  = 0.f;
	bool                  bDone       = false;
	FRandomStream         RandStream;

	// ---- WallActors (BeginPlay で自動収集) ----
	UPROPERTY()
	TArray<TObjectPtr<AWallActor>> WallActors;

	// ---- 内部メソッド ----
	FVector2D RandomInsideField();
	FVector2D RandomOnEdge();
	FVector2D RandomSpawnPos();
	void      SpawnEnemy(const FSpawnWave& Wave);
	void      SpawnBoss();
	void      UpdateEnemies();
	void      ApplyAuraDamage();
	void      DropGem(int32 TypeId, FVector2D Pos);
	void      CheckGemCollections();
	void      ApplyEnemyContactDamage();
	void      ResolveWallCollisions();
	float     CastRayToObstacles(FVector2D Origin, FVector2D Dir) const;

	// テーブル初期化
	void  InitDefaultEnemyTable();
	void  InitDefaultSpawnWaves();

	// スポーン補助
	const FSpawnWave* GetCurrentWave() const;
	int32             SelectTypeByWeight(const TArray<FEnemySpawnWeight>& Weights);

	// 敵タイプ別パラメータ取得（テーブルルックアップ）
	float GetEnemySpeed(int32 TypeId) const;
	float GetEnemyTypeMaxHP(int32 TypeId) const;

	// XP 処理（仕様: experience.md §1.1 区分線形テーブル）
	float XPRequiredForLevel(int32 Level) const;  // Level-1 → Level に必要な XP
	float CumulativeXPForLevel(int32 Level) const; // Lv1 から Level に達するまでの累計 XP
	void  ProcessXPGain(float Amount);
	void  OnLevelUp(int32 NextLevel);
};

