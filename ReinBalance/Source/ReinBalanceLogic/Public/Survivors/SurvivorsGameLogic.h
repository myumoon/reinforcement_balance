#pragma once
// Survivors の canonical production state と外部 level-up API を定義する。
// 初心者向け: UObject から独立したこの型だけがゲーム進行と選択状態を所有する。
// このファイルは UObject 系ヘッダーをインクルードしてはならない。
// CoreUObject.h, UObject/Object.h, Components/ActorComponent.h 等の追加は禁止。
// 新しい依存を追加する場合はチームレビューを必須とする。

#include "CoreMinimal.h"
#include "Survivors/SurvivorsTypes.h"
#include "Survivors/SurvivorsGameConstants.h"
#include "Survivors/SurvivorsLevelUpDecision.h"
#include "Survivors/SurvivorsCollisionTypes.h"  // FSurvivorsTargetGrid (ReinBalanceLogic module)
#include "Survivors/Weapons/SurvivorsWeaponLogic.h"  // FSurvivorsWeaponLogic 完全定義（TUniquePtr が destructor を要求する）

// ASurvivorsGame: UObject ヘッダーは include しない。friend 宣言のみで使用。
class ASurvivorsGame;

// ============================================================
// PythonTrainingComm に依存しないロジック層の結果型
// ============================================================

struct FSurvivorsStepResult
{
	TArray<float> Obs;
	float Reward     = 0.f;
	bool  bDone      = false;
	bool  bTruncated = false;
	bool  bStageCleared = false;
	bool  bTimedOut     = false;
	FString SpawnDebugJson;  // info_json 構築用
};

/** 外部 choice の適用結果と再送用スナップショット。 */
enum class ESurvivorsLevelUpApplyStatus : uint8
{
	Applied,
	StaleDecision,
	InvalidChoice,
};

/**
 * choice 適用直後の応答に必要な immutable snapshot。
 * 初心者向け: 同じ要求が再送されたとき、この値をそのまま返して二重適用を防ぐ。
 */
struct FSurvivorsLevelUpApplyResult
{
	ESurvivorsLevelUpApplyStatus Status = ESurvivorsLevelUpApplyStatus::StaleDecision;
	FString DecisionId;
	FString ChoiceId;
	TArray<float> PostChoiceObs;
	FSurvivorsPendingLevelUpDecision PendingAfter;
	int32 BacklogAfter = 0;
};

struct FSurvivorsResetResult
{
	TArray<float> Obs;
	FString ObsSchemaHash;  // Python 側の /reset レスポンス互換のため必須
};

/**
 * validation-only post-decision RNG stream の immutable binding。
 * 通常 reset/step は bActive=false のまま従来 RandStream semantics を使う。
 */
struct FSurvivorsValidationBranchRngState
{
	bool bActive = false;
	FString SchemaVersion;
	FString ReplicationKey;
	int32 StreamSeed = 0;
	int32 InitialStreamState = 0;
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
	float  SpawnMaxDistance   = 600.f;
	float  EnemyRecycleDistance = 650.f;
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
	FString ItemSelectionMode = TEXT("auto");

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

class REINBALANCELOGIC_API FSurvivorsGameLogic
{
public:
	FSurvivorsGameLogic();
	~FSurvivorsGameLogic();

	// TUniquePtr は move-only のためコピー禁止
	FSurvivorsGameLogic(const FSurvivorsGameLogic&)            = delete;
	FSurvivorsGameLogic& operator=(const FSurvivorsGameLogic&) = delete;
	FSurvivorsGameLogic(FSurvivorsGameLogic&&)                 = default;
	FSurvivorsGameLogic& operator=(FSurvivorsGameLogic&&)      = default;

	// ---- 初期化 ----

	/** 設定を適用して内部テーブルを初期化する */
	bool Initialize(const FSurvivorsGameLogicConfig& Config);

	/** Apply runtime config. Missing static arrays keep the previous values. */
	bool ApplyConfig(const FSurvivorsGameLogicConfig& Config);

	/**
	 * RSI 初期装備の slot 数、ID、level、一意性を検証する。
	 * 初心者向け: 不正な loadout を state へ適用する前に全入口で同じ規則により拒否する。
	 */
	static bool IsValidInitialLoadout(const FSurvivorsGameLogicConfig& Config);

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

	/**
	 * コンテンツ契約を enum/table/config から JSON として一方向 export する。
	 * Python 側が武器 ID や進化組を複製せず、ゲーム本体と同じ定義を監査に使える。
	 */
	FString GetContentSchema() const;

	/**
	 * 行動と時間刻みの契約を実際の Logic 定数・設定から JSON として返す。
	 * 9方向、物理刻み、移動量、pause/level-up timing を監査側へ渡す。
	 */
	FString GetActionTimeSchema() const;

	/** 観測次元数 */
	int32 GetObsDim() const;

	/** ステップ報酬 */
	float GetReward() const;

	/** エピソード終了判定 */
	bool IsDone() const;
	bool IsTruncated() const;

	/** SpawnDebug JSON */
	FString GetSpawnDebugJson() const;

	/**
	 * XP を source of truth である Logic に加算する。
	 * 初心者向け: Component は計算せず、この入口へ渡すだけにして二重実装を防ぐ。
	 */
	void AddExperience(float Amount) { ProcessXPGain(Amount); }

	/** 外部選択の保留状態を返す。 */
	bool IsLevelUpPending() const { return LevelUpDecisionState.IsPending(); }
	const FSurvivorsPendingLevelUpDecision& GetPendingLevelUpDecision() const
	{
		return LevelUpDecisionState.GetPending();
	}
	int32 GetLevelUpBacklog() const { return LevelUpBacklog; }
	const FString& GetItemSelectionMode() const { return CurrentConfig.ItemSelectionMode; }

	/**
	 * game thread 上で choice を一度だけ適用する。
	 * duplicate は直前の immutable result を返し、stale/invalid は mutation しない。
	 */
	FSurvivorsLevelUpApplyResult ApplyExternalLevelUpChoice(
		const FString& DecisionId, const FString& ChoiceId);

	/**
	 * pending choiceをinternal sandboxへ適用し、productionと同じraw observationを返す。
	 * 初心者向け: const method内でdeep cloneだけを変更するため、本物のepisode stateは不変のままです。
	 */
	FSurvivorsChoicePreview PreviewLevelUpChoice(
		const FString& DecisionId, const FString& ChoiceId) const;

	// ---- ParallelFor 内で直接呼ぶ API ----

	/** 複数物理ステップを実行して結果を返す */
	FSurvivorsStepResult ExecStep(const TArray<float>& Action, int32 Steps);

	/** リセットして初期 obs を返す */
	FSurvivorsResetResult ExecReset(TOptional<int32> Seed);

	/**
	 * semantic replay 完了後の validation worker だけが post-decision stream を切り替える。
	 * schema/key 不正時は RandStream を変更せず false を返す。
	 */
	bool ActivateValidationBranchRng(
		const FString& SchemaVersion,
		const FString& ReplicationKey,
		int32 StreamSeed);

	/** validation branch RNG の固定 schema version を返す。 */
	static const TCHAR* GetValidationBranchRngSchemaVersion()
	{
		return TEXT("survivors.branch_rng.v1");
	}

	/** validation stream binding と現在 state を監査する。 */
	const FSurvivorsValidationBranchRngState& GetValidationBranchRngState() const
	{
		return ValidationBranchRngState;
	}
	int32 GetValidationBranchRngCurrentState() const
	{
		return RandStream.GetCurrentSeed();
	}

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
	int32     GetEpisodeGemCount()    const { return EpisodeGemCount; }
	int32     GetEpisodeKillCount()   const { return EpisodeKillCount; }
	bool      IsAlive()               const { return !bDone; }
	bool      IsStageClear()          const { return bStageCleared; }
	bool      IsTimedOut()            const { return bTimedOut; }
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

	// ---- 武器クラスが使用する API (FSurvivorsWeaponLogic サブクラスからアクセス) ----
	void  EquipWeapon(int32 SlotIdx, EWeaponType Type, int32 Level);
	void  UnequipWeapon(int32 SlotIdx);
	void  SpawnProjectile(const FProjectileState& P) { Projectiles.Add(P); }
	void  SpawnGroundZone(const FGroundZoneState& Z) { GroundZones.Add(Z); }
	TArray<FProjectileState>& GetProjectiles() { return Projectiles; }
	void  UpdateProjectilesBySlot(int32 InSlotIdx, float Dt, TFunctionRef<bool(FProjectileState&, float)> Callback);
	TArray<FProjectileObsState> GetProjectileObsView() const;
	int32 GetOrbitOrbSlotIdx(int32 GI) const;
	void  QueryEnemyContacts(FVector2D Pos, float Radius, TArray<const struct FSurvivorsTargetProxy*>& Out) const;
	bool  ReflectOffWall(FVector2D& InOutPos, FVector2D& InOutVel, float Radius) const;

	// ---- 状態データ（テストヘルパーからのアクセス用） ----

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
	int32                 EpisodeGemCount   = 0;
	int32                 EpisodeKillCount  = 0;
	bool                  bDone             = false;
	bool                  bTruncated        = false;
	bool                  bStageCleared     = false;
	bool                  bTimedOut         = false;
	FRandomStream         RandStream;
	FSurvivorsValidationBranchRngState ValidationBranchRngState;
	FSurvivorsSpawnDebug  LastSpawnDebug;

	// プロジェクタイル・グラウンドゾーン
	TArray<FProjectileState> Projectiles;
	TArray<FGroundZoneState> GroundZones;

	// コリジョングリッド（BuildEnemyGrid/BuildPickupGrid で毎ステップ再構築）
	FSurvivorsTargetGrid EnemyGrid;
	FSurvivorsTargetGrid PickupGrid;

	// 純 C++ 武器配列
	TArray<TUniquePtr<FSurvivorsWeaponLogic>> Weapons;

	// 現在の設定
	FSurvivorsGameLogicConfig CurrentConfig;

private:
	friend class ASurvivorsGame;
#if (defined(WITH_AUTOMATION_TESTS) && WITH_AUTOMATION_TESTS) || (defined(WITH_REINBALANCE_LOGIC_TESTS) && WITH_REINBALANCE_LOGIC_TESTS)
	friend struct FSurvivorsGameTestAccess;
#endif

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

	uint64 EpisodeSerial = 0;
	int32 LevelUpBacklog = 0;
	FSurvivorsLevelUpDecisionState LevelUpDecisionState;
	TOptional<FSurvivorsLevelUpApplyResult> LastAppliedLevelUpResult;

	// ---- 内部メソッド ----
	FVector2D RandomInsideField();
	FVector2D RandomOnEdge();
	FVector2D RandomSpawnPos();
	void      SpawnEnemy(const FSpawnWave& Wave);
	void      SpawnBoss();
	void      UpdateEnemies();
	void      RecycleDistantEnemies();
	void      DropGem(int32 TypeId, FVector2D Pos);
	void      CheckGemCollections();
	void      ApplyEnemyContactDamage();
	void      ComputeContactHits(FSurvivorsHitFrame& HitFrame);
	void      ApplyContactHits(FSurvivorsHitFrame& HitFrame);
	void      ComputePickupHits(FSurvivorsHitFrame& HitFrame);
	void      ApplyPickupHits(FSurvivorsHitFrame& HitFrame);
	void      CheckSpecialPickups();
	void      ComputeGroundZoneHits(FSurvivorsHitFrame& HitFrame);
	void      ComputeProjectileHits(FSurvivorsHitFrame& HitFrame);
	void      TickProjectiles(float Dt);
	void      TickGroundZones(float Dt);
	void      RegisterEnemyTargets();
	void      RegisterPickupTargets();
	void      ResolveWallCollisions();
	float     CastRayToObstacles(FVector2D Origin, FVector2D Dir) const;
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
	void  AdvanceEligibleLevels();
	int32 CountEligibleLevelBacklog() const;
	void  RecalcPassiveEffects();
	FPassiveEffects ComputePassiveEffects() const;
	TArray<FLevelUpChoice> BuildLevelUpChoices();
	void  ApplyLevelUpChoice(const FLevelUpChoice& Choice);
	void  EvolveWeapon(int32 SlotIdx, EWeaponType EvolvedType);
	TArray<int32> GetEvolvableWeapons() const;
	void  ApplyAction(int32 ActionIdx, float Dt);
	void  StepSpawn(float Dt);
	void  CheckFloorPickups();
	void  TickWeapons(float Dt);
	void  ComputeAllWeaponHits(FSurvivorsHitFrame& HitFrame);
	void  ApplyWeaponHits(FSurvivorsHitFrame& HitFrame);
	void  BuildEnemyGrid();
	void  BuildPickupGrid();
	void  QueryPickupContacts(FVector2D Pos, float Radius, TArray<const struct FSurvivorsTargetProxy*>& Out) const;
	TUniquePtr<FSurvivorsWeaponLogic> CreateWeaponLogic(EWeaponType Type);
	/**
	 * preview専用に全stateをdeep cloneし、外部pointerをclone側へ再束縛する。
	 * 初心者向け: public snapshot APIにはせず、反実仮想計算の一時objectだけを生成します。
	 */
	TUniquePtr<FSurvivorsGameLogic> CloneForPreview() const;

#if (defined(WITH_AUTOMATION_TESTS) && WITH_AUTOMATION_TESTS) || (defined(WITH_REINBALANCE_LOGIC_TESTS) && WITH_REINBALANCE_LOGIC_TESTS)
	friend struct FSurvivorsGameTestAccess;
	friend struct FSurvivorsChoiceProjectionTestAccess;
#endif
};
