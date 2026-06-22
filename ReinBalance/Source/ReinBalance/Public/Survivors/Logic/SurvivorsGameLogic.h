#pragma once
// このファイルは UObject 系ヘッダーをインクルードしてはならない。
// CoreUObject.h, UObject/Object.h, Components/ActorComponent.h 等の追加は禁止。
// 新しい依存を追加する場合はチームレビューを必須とする。

#include "CoreMinimal.h"
#include "Survivors/Logic/SurvivorsTypes.h"
#include "Survivors/Logic/SurvivorsGameConstants.h"

// 前方宣言
class FSurvivorsWeaponBase;

// ============================================================
// PythonTrainingComm に依存しないロジック層の結果型
// ============================================================

struct FSurvivorsStepResult
{
	TArray<float> Obs;
	float Reward     = 0.f;
	bool  bDone      = false;
	bool  bTruncated = false;
	FString SpawnDebugJson;  // info_json 構築用
};

struct FSurvivorsResetResult
{
	TArray<float> Obs;
	FString ObsSchemaHash;  // Python 側の /reset レスポンス互換のため必須
};

// ============================================================
// 設定構造体: ASurvivorsGame の UPROPERTY から抽出したフィールドセット
// ============================================================

struct FSurvivorsGameLogicConfig
{
	// ---- フィールド設定 ----
	float  FieldHalfSize      = 1000.f;
	float  SimToUE            = 5.f;
	bool   bVariableFrameRate = false;

	// ---- 敵設定 ----
	int32  MinActiveEnemies   = 0;
	int32  MaxActiveEnemies   = 600;
	float  SpawnRateMult      = 1.0f;
	int32  MaxEnemyTypeId     = 10;
	float  EnemyHPScale       = 1.0f;
	float  EnemyDamageScale   = 1.0f;
	float  EnemySpeedMult     = 1.0f;
	float  SpawnMinDistance   = 500.f;
	float  SpawnMaxDistance   = 700.f;
	float  BossSpawnTime      = 300.f;

	// ---- プレイヤー設定 ----
	float  MaxPlayerHP        = 100.f;
	float  MoveSpeed          = 80.f;
	float  PlayerRadius       = 10.f;
	float  GemPickupRadius    = SurvivorsGameConstants::BaseGemPickupRadius;
	float  FloorPickupRadius  = 30.f;

	// ---- 報酬設定 ----
	float  AliveReward        = 0.001f;
	float  ItemReward         = 1.0f;
	float  KillReward         = 2.0f;
	float  MaxEpisodeTime     = 300.f;

	// ---- 時間スケーリング ----
	bool   bTimeScalingEnabled = false;
	float  HPScaleRatePerMin   = 0.10f;
	float  DamageScaleRatePerMin = 0.05f;

	// ---- 訓練用パラメータ拡張 ----
	FString WeaponPoolMode    = TEXT("all_base");
	TArray<int32> AllowedWeaponTypes;
	TMap<int32, float> WeaponWeights;
	bool   bEnablePassives    = true;
	bool   bEnableEvolutions  = true;
	float  ReplayOldPhaseFraction = 0.0f;
	FString StartingWeaponMode = TEXT("garlic");

	// ---- RSI オーバーライド ----
	struct FWeaponSlotOverride  { int32 WeaponId = 0; int32 Level = 1; };
	struct FPassiveSlotOverride { int32 PassiveId = 0; int32 Level = 1; };
	float                        InitialElapsedTime  = 0.f;
	TArray<FWeaponSlotOverride>  InitialWeaponSlots;
	TArray<FPassiveSlotOverride> InitialPassiveSlots;
	bool                         bHasInitialOverride = false;

	// ---- スポーン/敵テーブル（BeginPlay で初期化済み） ----
	TArray<FBox2D>          WallBounds;    // AWallActorから変換済み（BeginPlayで1回設定）
	TArray<FSpawnWave>      SpawnWaves;
	TArray<FEnemyTypeParams> EnemyTypeTable;
};

// ============================================================
// FSurvivorsGameLogic: UObject 非依存の純粋 C++ ゲームロジック
// ============================================================

class REINBALANCE_API FSurvivorsGameLogic
{
public:
	FSurvivorsGameLogic();
	~FSurvivorsGameLogic();

	// ---- 初期化 ----

	/** 設定を適用して内部テーブルを初期化する */
	void Initialize(const FSurvivorsGameLogicConfig& Config);

	/** params 更新時に呼ぶ（WallBounds/SpawnWaves/EnemyTypeTable は更新しない） */
	void ApplyConfig(const FSurvivorsGameLogicConfig& Config);

	// ---- 訓練 API ----

	/** 離散行動 (0〜8) を受けて 1 物理ステップ進める */
	void PhysicsStep(int32 ActionIdx);

	/** 状態をリセット */
	void Reset(TOptional<int32> Seed);

	/** 観測ベクトルを返す */
	TArray<float> GetObservation() const;

	/** 観測スキーマを返す */
	TArray<FSurvivorsObsSegment> GetObsSchema() const;

	/** obs 次元に影響するパラメータから生成するハッシュ */
	FString GetObsSchemaHash() const;

	/** 観測次元数 */
	int32 GetObsDim() const;

	/** ステップ報酬 */
	float GetReward() const;

	/** エピソード終了判定 */
	bool IsDone() const;
	bool IsTruncated() const;

	/** SpawnDebug JSON */
	FString GetSpawnDebugJson() const;

	// ---- ParallelFor 内で直接呼ぶ API ----

	/** 複数物理ステップを実行して結果を返す */
	FSurvivorsStepResult ExecStep(const TArray<float>& Action, int32 Steps);

	/** リセットして初期 obs を返す */
	FSurvivorsResetResult ExecReset(TOptional<int32> Seed);

	// ---- ビュー / デバッグ向けアクセサ ----

	FVector2D GetPlayerPos()   const { return PlayerPos; }
	FVector2D GetPlayerVel()   const { return PlayerVel; }
	float     GetPlayerHP()    const { return PlayerHP; }
	float     GetMaxPlayerHP() const { return CurrentConfig.MaxPlayerHP; }
	float     GetPlayerXP()    const { return PlayerXP; }
	int32     GetPlayerLevel() const { return PlayerLevel; }
	float     GetElapsedTime() const { return ElapsedTime; }
	float     GetLastReward()  const { return LastReward; }
	float     GetEpisodeBaseReward()  const { return EpisodeBaseReward; }
	int32     GetEpisodeStepCount()   const { return EpisodeStepCount; }
	bool      IsShieldActive() const { return bShieldActive; }
	float     GetPlayerShieldTimer() const { return PlayerShieldTimer; }

	int32     GetEnemyCount()         const { return Enemies.Num(); }
	FVector2D GetEnemyPos(int32 i)    const { return Enemies.IsValidIndex(i) ? Enemies[i].Pos : FVector2D::ZeroVector; }
	int32     GetEnemyType(int32 i)   const { return Enemies.IsValidIndex(i) ? Enemies[i].TypeId : 0; }
	float     GetEnemyHP(int32 i)     const { return Enemies.IsValidIndex(i) ? Enemies[i].HP : 0.f; }
	float     GetEnemyMaxHP(int32 i)  const { return Enemies.IsValidIndex(i) ? Enemies[i].MaxHP : 1.f; }

	int32     GetItemCount()       const { return Gems.Num(); }
	FVector2D GetItemPos(int32 i)  const;
	EGemType  GetItemGemType(int32 i) const;

	int32     GetFloorPickupCount()       const { return FloorPickups.Num(); }
	FVector2D GetFloorPickupPos(int32 i)  const { return FloorPickups.IsValidIndex(i) ? FloorPickups[i].Pos : FVector2D::ZeroVector; }
	EFloorPickupType GetFloorPickupType(int32 i) const { return FloorPickups.IsValidIndex(i) ? FloorPickups[i].Type : EFloorPickupType::FloorChicken; }

	int32     GetSpecialPickupCount()      const { return SpecialPickups.Num(); }
	FVector2D GetSpecialPickupPos(int32 i) const { return SpecialPickups.IsValidIndex(i) ? SpecialPickups[i].Pos : FVector2D::ZeroVector; }
	ESpecialPickupType GetSpecialPickupType(int32 i) const { return SpecialPickups.IsValidIndex(i) ? SpecialPickups[i].Type : ESpecialPickupType::Rosary; }

	int32     GetDestructibleCount()        const { return Destructibles.Num(); }
	FVector2D GetDestructiblePos(int32 i)   const { return Destructibles.IsValidIndex(i) ? Destructibles[i].Pos : FVector2D::ZeroVector; }
	bool      IsDestructibleActive(int32 i) const { return Destructibles.IsValidIndex(i) ? Destructibles[i].bActive : false; }

	// TODO(issue): プロジェクタイル/グラウンドゾーン/軌道オーブのアクセサは
	// Phase 3 武器移植完了後に追加する。現時点では ASurvivorsGame 経由で取得する。
	int32     GetProjectileCount()             const;
	FVector2D GetProjectilePos(int32 i)        const;
	FSimRadius GetProjectileRadius(int32 i)    const;
	EWeaponType GetProjectileWeaponType(int32 i) const;
	float     GetProjectileBoxHalfWidth(int32 i) const;

	int32     GetGroundZoneCount()             const;
	FVector2D GetGroundZonePos(int32 i)        const;
	float     GetGroundZoneRadius(int32 i)     const;
	EWeaponType GetGroundZoneWeaponType(int32 i) const;
	bool      IsGroundZoneWarning(int32 i)     const;

	int32       GetOrbitOrbCount()                  const;
	FVector2D   GetOrbitOrbPos(int32 i)             const;
	EWeaponType GetOrbitOrbWeaponType(int32 i)      const;
	float       GetOrbitOrbVisualRadius(int32 i)    const;

	/** スクリーン内判定（Camera Z=2000 基準） */
	bool IsOnScreen(FVector2D WorldPos) const;

	/** TypeId → 生存数マップ */
	TMap<int32, int32> GetEnemyCountByType() const;

	float GetXPRequiredForNextLevel() const;
	float GetCurrentLevelXP()         const;

	const FWeaponSlot&      GetWeaponSlot(int32 Idx)            const { return WeaponSlots[Idx]; }
	const FPassiveSlot&     GetPassiveSlot(int32 Idx)           const { return PassiveSlots[Idx]; }
	const FPassiveEffects&  GetCachedPassiveEffects()           const { return CachedPassiveEffects; }
	FString                 GetEnemyTypeDebugLabel(int32 TypeId) const;
	int32                   GetPassiveItemMaxLevel(EPassiveItemType Type) const;
	float                   GetAuraSize() const;

	float GetEpisodeActiveScore() const
	{
		return EpisodeBaseReward - CurrentConfig.AliveReward * static_cast<float>(EpisodeStepCount);
	}

	FSurvivorsSpawnDebug GetSpawnDebug() const { return LastSpawnDebug; }

	// ---- 状態データ（コンポーネント・テストヘルパーからのアクセス用） ----
	// NOTE: Phase 3 移行後、テストヘルパーはこれらを直接参照する

	FVector2D             PlayerPos;
	FVector2D             PlayerVel;
	float                 PlayerHP          = 100.f;
	float                 PlayerXP          = 0.f;
	int32                 PlayerLevel       = 0;
	FWeaponSlot           WeaponSlots[SurvivorsGameConstants::MaxWeaponSlots];
	FPassiveSlot          PassiveSlots[SurvivorsGameConstants::MaxPassiveSlots];
	FPassiveEffects       CachedPassiveEffects;
	float                 GlobalFreezeUntilTime = -1.f;
	float                 PlayerShieldTimer = 0.f;
	bool                  bShieldActive     = false;
	int32                 MaxRevivalCount   = 0;
	int32                 UsedRevivalCount  = 0;
	int32                 NextEnemyId       = 0;
	int32                 NextGemId         = 0;
	TArray<FFloorPickupState>   FloorPickups;
	TArray<FSpecialPickupState> SpecialPickups;
	TArray<FDestructibleState>  Destructibles;
	TArray<FGemState>     Gems;
	TArray<FEnemyState>   Enemies;
	float                 ElapsedTime       = 0.f;
	float                 SpawnAccumulator  = 0.f;
	bool                  bBossSpawned      = false;
	float                 LastReward        = 0.f;
	float                 EpisodeBaseReward = 0.f;
	int32                 EpisodeStepCount  = 0;
	bool                  bDone             = false;
	bool                  bTruncated        = false;
	FRandomStream         RandStream;
	FSurvivorsSpawnDebug  LastSpawnDebug;

	// プロジェクタイル・グラウンドゾーン（Phase 3 武器移植後に有効化）
	TArray<FProjectileState> Projectiles;
	TArray<FGroundZoneState> GroundZones;

	// 純 C++ 武器配列（Phase 3 移植後に有効化）
	// TODO(issue): Phase 3 武器移植が完了したら TUniquePtr<FSurvivorsWeaponBase> を使う
	// TArray<TUniquePtr<FSurvivorsWeaponBase>> Weapons;

	// 現在の設定
	FSurvivorsGameLogicConfig CurrentConfig;

	// Phase 2 暫定実装: ASurvivorsGame へのデリゲートポインタ
	// TODO(issue): Phase 3 で Logic に完全移植したら削除する
	// NOTE: SurvivorsGameLogic.h は UObject 系ヘッダーをインクルードしないため前方宣言のみ
	// ASurvivorsGame は UObject 派生だが、ここでは生のポインタとして前方宣言する
	void SetGameFacade(void* InGame) { GameFacade = InGame; }

private:
	// ---- 定数 ----
	static constexpr int32 MaxWeaponSlots  = SurvivorsGameConstants::MaxWeaponSlots;
	static constexpr int32 MaxPassiveSlots = SurvivorsGameConstants::MaxPassiveSlots;
	static constexpr int32 MaxWeaponLevel  = SurvivorsGameConstants::MaxWeaponLevel;
	static constexpr int32 MaxPlayerLevel  = SurvivorsGameConstants::MaxPlayerLevel;
	static constexpr float PhysicsDt       = SurvivorsGameConstants::PhysicsDt;
	static constexpr float MaxGameTime     = SurvivorsGameConstants::MaxGameTime;
	static constexpr float ContactHitInterval = SurvivorsGameConstants::ContactHitInterval;
	static constexpr float BaseMaxPlayerHPConst     = SurvivorsGameConstants::StandardMaxPlayerHP;
	static constexpr float BaseGemPickupRadiusConst = SurvivorsGameConstants::BaseGemPickupRadius;

	mutable int32 CachedObsDim = -1;
	float PhysicsAccumTime = 0.f;

	// Phase 2 暫定: ASurvivorsGame へのデリゲートポインタ（void* で UObject 依存を回避）
	// TODO(issue): Phase 3 で Logic に完全移植したら削除する
	void* GameFacade = nullptr;

	// ---- 内部メソッド（Phase 3 で各コンポーネントから移植する） ----
	// TODO(issue): 以下は Phase 3 で実装する
	FVector2D RandomInsideField();
	FVector2D RandomOnEdge();
	FVector2D RandomSpawnPos();
	void      SpawnEnemy(const FSpawnWave& Wave);
	void      SpawnBoss();
	void      UpdateEnemies();
	void      DropGem(int32 TypeId, FVector2D Pos);
	void      CheckGemCollections();
	void      ApplyEnemyContactDamage();
	void      ResolveWallCollisions();
	float     CastRayToObstacles(FVector2D Origin, FVector2D Dir) const;
	bool      ReflectOffWall(FVector2D& InOutPos, FVector2D& InOutVel, float Radius) const;
	void      FinalizePendingEnemies();
	void      FinalizePickupRemovals();
	void      InitDefaultEnemyTable();
	void      InitDefaultSpawnWaves();
	const FSpawnWave* GetCurrentWave() const;
	int32             GetCurrentWaveIndex() const;
	bool              BuildSpawnWeights(const FSpawnWave& Wave, TArray<FEnemySpawnWeight>& OutWeights, bool& bOutUsedCurriculumPool) const;
	int32             SelectTypeByWeight(const TArray<FEnemySpawnWeight>& Weights);
	float GetEnemySpeed(int32 TypeId) const;
	float GetEnemyTypeMaxHP(int32 TypeId) const;
	float XPRequiredForLevel(int32 Level) const;
	float CumulativeXPForLevel(int32 Level) const;
	void  ProcessXPGain(float Amount);
	void  OnLevelUp(int32 NextLevel);
	void  RecalcPassiveEffects();
	void  ApplyAction(int32 ActionIdx, float Dt);
	void  StepSpawn(float Dt);
	void  CheckFloorPickups();
	void  TickWeapons(float Dt);
	void  BuildEnemyGrid();
	void  BuildPickupGrid();
	void  QueryEnemyContacts(FVector2D Pos, float Radius, TArray<const struct FSurvivorsTargetProxy*>& Out) const;
	void  QueryPickupContacts(FVector2D Pos, float Radius, TArray<const struct FSurvivorsTargetProxy*>& Out) const;
};
