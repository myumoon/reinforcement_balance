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

/**
 * Vampire Survivors 風サバイバルゲームのロジッククラス（ビジュアルなし）。
 *
 * 2D XY 平面上でプレイヤーが敵を倒しながらアイテムを集めて生き延びる。
 * ビジュアル表示は別クラスが担う（未実装）。
 *
 * 行動: 離散5方向 (0=+Y, 1=-Y, 2=-X, 3=+X, 4=静止)
 * 観測: GetObsDim() 次元 = 23 + NumItemObs*2 + MaxEnemyObs*6
 *   [0-1]    プレイヤー位置 (x,y) / FieldHalfSize
 *   [2-3]    プレイヤー速度 (vx,vy) / MoveSpeed（-1〜1 に正規化）
 *   [4-11]   8方向レイキャスト壁距離 (0~1)
 *   [12]     プレイヤー HP / MaxPlayerHP
 *   [13-18]  武器スロット × 3: (type_norm, level_norm) × MaxWeaponSlots
 *   [19]     現在の敵数 / MaxEnemyObs
 *   [20]     次スポーンまでの残り時間 (0~1)
 *   [21]     xp_progress (0~1, アイテム取得・敵撃破で増加、レベルアップでリセット)
 *   [22]     player_level (0~1 = level / MaxPlayerLevel, 最大 MaxPlayerLevel=100)
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

	UFUNCTION(BlueprintPure, Category = "Survivors|Config")
	FVector2D GetPlayerPos()   const { return PlayerPos; }
	
	FVector2D GetPlayerVel()   const { return PlayerVel; }
	float     GetPlayerHP()    const { return PlayerHP; }
	float     GetMaxPlayerHP() const { return MaxPlayerHP; }
	float     GetAuraSize()    const { return AuraRadius; }

	int32     GetItemCount()       const { return ItemPositions.Num(); }
	FVector2D GetItemPos(int32 i)  const
	{
		return ItemPositions.IsValidIndex(i) ? ItemPositions[i] : FVector2D::ZeroVector;
	}

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

	/** アイテム獲得報酬 */
	UPROPERTY(EditAnywhere, Category = "Survivors|Train")
	float ItemReward = 3.0f;

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

	/** フィールド上のアイテム数 */
	UPROPERTY(EditAnywhere, Category = "Survivors|Config")
	int32 NumItems = 10;

	/** 敵スポーン間隔 [秒]（/params で上書き可能） */
	UPROPERTY(EditAnywhere, Category = "Survivors|Config")
	float EnemySpawnInterval = 5.f;

	/** 同時に存在できる最大敵数（カリキュラム制御用, /params で上書き可能） */
	UPROPERTY(EditAnywhere, Category = "Survivors|Config")
	int32 MaxActiveEnemies = 6;

	/** 敵速度グローバル倍率（カリキュラム制御用, /params で上書き可能） */
	UPROPERTY(EditAnywhere, Category = "Survivors|Config")
	float EnemySpeedMult = 1.0f;

	/** アイテム獲得時の XP 量 */
	UPROPERTY(EditAnywhere, Category = "Survivors|Config")
	float ItemXP = 1.0f;

	/** 敵撃破時の XP 量（ItemXP の倍率） */
	UPROPERTY(EditAnywhere, Category = "Survivors|Config")
	float KillXPRatio = 0.05f;

	/** Lv0→1 に必要な基礎 XP */
	UPROPERTY(EditAnywhere, Category = "Survivors|Config")
	float XPBase = 5.0f;

	/** 1 レベルごとに増加する XP 量（XPRequired(n) = XPBase + XPGrowth * n） */
	UPROPERTY(EditAnywhere, Category = "Survivors|Config")
	float XPGrowth = 3.0f;

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

	/** Garlic Lv1 オーラ攻撃半径 [u]（VS 仕様値） */
	UPROPERTY(EditAnywhere, Category = "Survivors|Weapon")
	float MinAuraRadius = 80.0f;

	/** Garlic Lv8 オーラ攻撃半径 [u]（VS 仕様値） */
	UPROPERTY(EditAnywhere, Category = "Survivors|Weapon")
	float MaxAuraRadius = 185.0f;

	/** Garlic Lv1 の DPS（敵1体あたり）: damage / hit_interval = 5 / 1.3 */
	UPROPERTY(EditAnywhere, Category = "Survivors|Weapon")
	float MinAuraDPS = 3.85f;

	/** Garlic Lv8 の DPS（敵1体あたり）: damage / hit_interval = 20 / 0.95 */
	UPROPERTY(EditAnywhere, Category = "Survivors|Weapon")
	float MaxAuraDPS = 21.05f;

	// ---- 敵設定 ----

	/**
	 * 敵11種のパラメーターテーブル（Mad Forest 準拠）。
	 * type_id = インデックス: 0=Bat, 1=Zombie, ..., 10=GiantBat。
	 * カリキュラム用に EditAnywhere・/params エンドポイントで変更可能。
	 */
	UPROPERTY(EditAnywhere, Category = "Survivors|Enemy")
	TArray<FEnemyTypeParams> EnemyTypeTable;

	// ---- アイテム ----

	/** アイテム収集半径 [u]（Plan06 でジェム pickup_radius=30 に置換） */
	UPROPERTY(EditAnywhere, Category = "Survivors|Item")
	float ItemCollectRadius = 10.0f;

protected:
	virtual void BeginPlay() override;

private:
	// ---- 定数 ----
	static constexpr int32  NumItemObs       = 20;
	static constexpr int32  MaxEnemyObs      = 20;
	static constexpr int32  MaxWeaponSlots   = 3;
	static constexpr int32  MaxWeaponTypeSlots = 8; // obs エンコードの型容量
	static constexpr int32  MaxWeaponLevel   = 8;
	static constexpr int32  MaxPlayerLevel   = 100;
	static constexpr float  PhysicsDt        = 1.f / 60.f;
	static const FVector2D  RayDirs[8];              // 8方向レイキャスト方向

	// ---- 敵データ ----
	struct FEnemyState
	{
		FVector2D Pos;
		FVector2D Vel;
		int32     TypeId;          // 0〜10: EnemyTypeTable のインデックス
		float     HP;
		float     MaxHP;
		float     CollisionRadius; // スポーン時に EnemyTypeTable からコピー
	};

	// ---- 状態 ----
	FVector2D             PlayerPos;
	FVector2D             PlayerVel;
	float                 PlayerHP    = 100.f;
	float                 PlayerXP    = 0.f;
	int32                 PlayerLevel = 0;
	float                 AuraRadius  = 0.0f;
	float                 AuraDPS     = 0.0f;
	FWeaponSlot           WeaponSlots[MaxWeaponSlots];
	TArray<FVector2D>     ItemPositions;
	TArray<FEnemyState>   Enemies;
	float                 SpawnTimer  = 0.f;
	float                 LastReward  = 0.f;
	bool                  bDone       = false;
	FRandomStream         RandStream;

	// ---- WallActors (BeginPlay で自動収集) ----
	UPROPERTY()
	TArray<TObjectPtr<AWallActor>> WallActors;

	// ---- 内部メソッド ----
	FVector2D RandomInsideField();
	FVector2D RandomOnEdge();
	void      SpawnEnemy();
	void      UpdateEnemies();
	void      ApplyAuraDamage();
	void      CheckItemCollections();
	void      ApplyEnemyContactDamage();
	void      ResolveWallCollisions();
	float     CastRayToObstacles(FVector2D Origin, FVector2D Dir) const;

	// 敵テーブル初期化
	void  InitDefaultEnemyTable();

	// 敵タイプ別パラメータ取得（テーブルルックアップ）
	float GetEnemySpeed(int32 TypeId) const;
	float GetEnemyDamagePerTick(int32 TypeId) const;
	float GetEnemyTypeMaxHP(int32 TypeId) const;

	// XP 処理
	float XPRequiredForLevel(int32 Level) const;
	void  ProcessXPGain(float Amount);
	void  OnLevelUp(int32 nextLevel);
	
	// ステータス
	void  SetAuraSizeByLevel(int32 level, int32 maxLevel);
	void  SetAuraDpsByLevel(int32 level, int32 maxLevel);
};

