#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Survivors/Logic/SurvivorsGameConstants.h"
#include "Survivors/Logic/SurvivorsTypes.h"
#include "SurvivorsGame.generated.h"

class AWallActor;
class USurvivorsCollisionComponent;
class USurvivorsEnemyComponent;
class USurvivorsGemComponent;
class USurvivorsObservationComponent;
class USurvivorsPickupComponent;
class USurvivorsPlayerComponent;
class USurvivorsSpawnComponent;
class USurvivorsWeaponComponent;

/**
 * Vampire Survivors 風サバイバルゲームのロジッククラス（ビジュアルなし）。
 *
 * 2D XY 平面上でプレイヤーが敵を倒しながらアイテムを集めて生き延びる。
 * ビジュアル表示は別クラスが担う。
 *
 * 行動: 離散9方向 (0=+Y, 1=北東, 2=+X, 3=南東, 4=-Y, 5=南西, 6=-X, 7=北西, 8=静止)
 * 観測: 708次元（GetObsSchema() のセグメント合計）
 */
UCLASS()
class REINBALANCE_API ASurvivorsGame : public AActor
{
	GENERATED_BODY()

public:
	ASurvivorsGame();

	/** 離散行動 (0〜8) を受けて 1 物理ステップ進める */
	void PhysicsStep(int32 ActionIdx);

	/** 状態をリセット */
	void ResetState(TOptional<int32> Seed);

	/** 観測ベクトルを返す */
	TArray<float> GetObservation() const;

	/** 観測スキーマを返す */
	TArray<FSurvivorsObsSegment> GetObsSchema() const;

	/** 観測次元数: GetObsSchema() の dim 合計（初回計算後はキャッシュ） */
	int32 GetObsDim() const;

	/** obs 次元に影響するパラメータから生成するハッシュ */
	FString GetObsSchemaHash() const;

	/** ステップ報酬を返す */
	float GetReward() const;

	/** エピソード終了判定（HP <= 0） */
	bool IsDone() const;
	bool IsTruncated() const;

	FSurvivorsSpawnDebug GetSpawnDebug() const { return LastSpawnDebug; }
	FString GetSpawnDebugJson() const;

	// ---- ビュー / デバッグ向けアクセサー ----

	UFUNCTION(BlueprintPure, Category = "Survivors|Config")
	FVector2D GetPlayerPos()   const { return PlayerPos; }

	FVector2D GetPlayerVel()   const { return PlayerVel; }
	float     GetPlayerHP()    const { return PlayerHP; }
	float     GetMaxPlayerHP() const { return MaxPlayerHP; }

	/** 後方互換: Garlic オーラ半径（WeaponSlots から取得）。View の DrawAura 用 */
	float     GetAuraSize()    const;

	const FWeaponSlot& GetWeaponSlot(int32 Idx) const { return WeaponSlots[Idx]; }
	bool               IsShieldActive()          const { return bShieldActive; }

	// ジェムアクセサ
	int32     GetItemCount()       const { return Gems.Num(); }
	FVector2D GetItemPos(int32 i)  const;
	EGemType  GetItemGemType(int32 i) const;

	// 敵アクセサ
	int32     GetEnemyCount()         const { return Enemies.Num(); }
	FVector2D GetEnemyPos(int32 i)    const { return Enemies.IsValidIndex(i) ? Enemies[i].Pos   : FVector2D::ZeroVector; }
	int32     GetEnemyType(int32 i)   const { return Enemies.IsValidIndex(i) ? Enemies[i].TypeId : 0; }
	float     GetEnemyHP(int32 i)     const { return Enemies.IsValidIndex(i) ? Enemies[i].HP    : 0.f; }
	float     GetEnemyMaxHP(int32 i)  const { return Enemies.IsValidIndex(i) ? Enemies[i].MaxHP : 1.f; }

	// プロジェクタイルアクセサ（WeaponComponent 経由）
	int32     GetProjectileCount()              const;
	FVector2D GetProjectilePos(int32 i)         const;
	FSimRadius GetProjectileRadius(int32 i)     const;
	EWeaponType GetProjectileWeaponType(int32 i)const;

	// グラウンドゾーンアクセサ（WeaponComponent 経由）
	int32     GetGroundZoneCount()              const;
	FVector2D GetGroundZonePos(int32 i)         const;
	float     GetGroundZoneRadius(int32 i)      const;
	EWeaponType GetGroundZoneWeaponType(int32 i)const;

	// フロアアイテムアクセサ
	int32            GetFloorPickupCount()           const { return FloorPickups.Num(); }
	FVector2D        GetFloorPickupPos(int32 i)      const { return FloorPickups.IsValidIndex(i) ? FloorPickups[i].Pos : FVector2D::ZeroVector; }
	EFloorPickupType GetFloorPickupType(int32 i)     const { return FloorPickups.IsValidIndex(i) ? FloorPickups[i].Type : EFloorPickupType::FloorChicken; }

	// 特殊アイテムアクセサ
	int32              GetSpecialPickupCount()           const { return SpecialPickups.Num(); }
	FVector2D          GetSpecialPickupPos(int32 i)      const { return SpecialPickups.IsValidIndex(i) ? SpecialPickups[i].Pos : FVector2D::ZeroVector; }
	ESpecialPickupType GetSpecialPickupType(int32 i)     const { return SpecialPickups.IsValidIndex(i) ? SpecialPickups[i].Type : ESpecialPickupType::Rosary; }

	// 破壊可能オブジェクトアクセサ
	int32     GetDestructibleCount()          const { return Destructibles.Num(); }
	FVector2D GetDestructiblePos(int32 i)     const { return Destructibles.IsValidIndex(i) ? Destructibles[i].Pos : FVector2D::ZeroVector; }
	bool      IsDestructibleActive(int32 i)   const { return Destructibles.IsValidIndex(i) ? Destructibles[i].bActive : false; }

	// ---- 報酬設定 ----

	UPROPERTY(EditAnywhere, Category = "Survivors|Train")
	float AliveReward = 0.001f;

	UPROPERTY(EditAnywhere, Category = "Survivors|Train")
	float ItemReward = 1.0f;

	UPROPERTY(EditAnywhere, Category = "Survivors|Train")
	float KillReward = 2.0f;

	UPROPERTY(EditAnywhere, Category = "Survivors|Train")
	float MaxEpisodeTime = 300.f;

	// ---- フィールド設定 ----

	UPROPERTY(EditAnywhere, Category = "Survivors|Config")
	float FieldHalfSize = 1000.f;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Survivors|Config")
	float SimToUE = 5.f;

	UPROPERTY(EditAnywhere, Category = "Survivors|Config")
	int32 MinActiveEnemies = 4;

	UPROPERTY(EditAnywhere, Category = "Survivors|Config")
	int32 MaxActiveEnemies = 6;

	UPROPERTY(EditAnywhere, Category = "Survivors|Config")
	float SpawnRateMult = 1.0f;

	UPROPERTY(EditAnywhere, Category = "Survivors|Config")
	int32 MaxEnemyTypeId = 10;

	UPROPERTY(EditAnywhere, Category = "Survivors|Config")
	float EnemyHPScale = 1.0f;

	UPROPERTY(EditAnywhere, Category = "Survivors|Config")
	float EnemyDamageScale = 1.0f;

	// ---- スポーン設定 ----

	UPROPERTY(EditAnywhere, Category = "Survivors|Spawn")
	float SpawnMinDistance = 400.f;

	UPROPERTY(EditAnywhere, Category = "Survivors|Spawn")
	float SpawnMaxDistance = 600.f;

	UPROPERTY(EditAnywhere, Category = "Survivors|Spawn")
	float BossSpawnTime = 600.f;

	UPROPERTY(EditAnywhere, Category = "Survivors|Spawn")
	TArray<FSpawnWave> SpawnWaves;

	UPROPERTY(EditAnywhere, Category = "Survivors|Config")
	float EnemySpeedMult = 1.0f;

	// ---- プレイヤー設定 ----

	UPROPERTY(EditAnywhere, Category = "Survivors|Player")
	float MaxPlayerHP = 70.f;

	UPROPERTY(EditAnywhere, Category = "Survivors|Player")
	float MoveSpeed = 80.f;

	UPROPERTY(EditAnywhere, Category = "Survivors|Player")
	float PlayerRadius = 10.f;

	// ---- 敵設定 ----

	UPROPERTY(EditAnywhere, Category = "Survivors|Enemy")
	TArray<FEnemyTypeParams> EnemyTypeTable;

	// ---- ジェム設定 ----

	UPROPERTY(EditAnywhere, Category = "Survivors|Item")
	float GemPickupRadius = 30.f;

	// ---- 時間スケーリング ----

	UPROPERTY(EditAnywhere, Category = "Survivors|TimeScaling")
	bool bTimeScalingEnabled = true;

	// ---- 訓練用パラメータ拡張（/params エンドポイント経由で設定） ----

	/** 武器プール制御モード: "all" / "garlic_only" / "custom" */
	UPROPERTY(EditAnywhere, Category = "Survivors|Train")
	FString WeaponPoolMode = TEXT("garlic_only");

	/** カスタムモード時の許可武器タイプ ID リスト */
	UPROPERTY(EditAnywhere, Category = "Survivors|Train")
	TArray<int32> AllowedWeaponTypes;

	/** パッシブアイテムを有効にするか */
	UPROPERTY(EditAnywhere, Category = "Survivors|Train")
	bool bEnablePassives = false;

	/** 進化システムを有効にするか */
	UPROPERTY(EditAnywhere, Category = "Survivors|Train")
	bool bEnableEvolutions = false;

	/** リプレイ旧フェーズ比率（0.0〜1.0） */
	UPROPERTY(EditAnywhere, Category = "Survivors|Train")
	float ReplayOldPhaseFraction = 0.0f;

	/** 開始武器選択モード: "garlic" / "random" / "custom" */
	UPROPERTY(EditAnywhere, Category = "Survivors|Train")
	FString StartingWeaponMode = TEXT("garlic");

	UPROPERTY(EditAnywhere, Category = "Survivors|TimeScaling")
	float HPScaleRatePerMin = 0.10f;

	UPROPERTY(EditAnywhere, Category = "Survivors|TimeScaling")
	float DamageScaleRatePerMin = 0.05f;

protected:
	virtual void BeginPlay() override;

private:
	friend class USurvivorsCollisionComponent;
	friend class USurvivorsEnemyComponent;
	friend class USurvivorsGemComponent;
	friend class USurvivorsObservationComponent;
	friend class USurvivorsPickupComponent;
	friend class USurvivorsPlayerComponent;
	friend class USurvivorsSpawnComponent;
	friend class USurvivorsWeaponComponent;
	friend class USurvivorsWeaponBase;
	friend class USurvivorsGarlicWeapon;
	friend class USurvivorsWhipWeapon;
	friend class USurvivorsMagicWandWeapon;
	friend class USurvivorsKnifeWeapon;
	friend class USurvivorsAxeWeapon;
	friend class USurvivorsCrossWeapon;
	friend class USurvivorsKingBibleWeapon;
	friend class USurvivorsFireWandWeapon;
	friend class USurvivorsSantaWaterWeapon;
	friend class USurvivorsRunetracerWeapon;
	friend class USurvivorsLightningRingWeapon;
	friend class USurvivorsPentagramWeapon;
	friend class USurvivorsPeachoneWeapon;
	friend class USurvivorsEbonyWingsWeapon;
	friend class USurvivorsVandalierWeapon;
	friend class USurvivorsLaurelWeapon;
#if WITH_AUTOMATION_TESTS
	friend struct FSurvivorsGameTestAccess;
#endif

	// ---- 定数 ----
	static constexpr int32 MaxEnemyObs    = SurvivorsGameConstants::MaxEnemyObs;
	static constexpr int32 MaxWeaponSlots = SurvivorsGameConstants::MaxWeaponSlots;
	static constexpr int32 MaxPassiveSlots= SurvivorsGameConstants::MaxPassiveSlots;
	static constexpr int32 MaxWeaponLevel = SurvivorsGameConstants::MaxWeaponLevel;
	static constexpr int32 MaxPlayerLevel = SurvivorsGameConstants::MaxPlayerLevel;
	static constexpr float PhysicsDt     = SurvivorsGameConstants::PhysicsDt;
	static constexpr float MaxGameTime   = SurvivorsGameConstants::MaxGameTime;
	static constexpr float ContactHitInterval = SurvivorsGameConstants::ContactHitInterval;

	// パッシブ再計算用ベース値（累積増幅を防ぐため MaxPlayerHP / GemPickupRadius の初期値を保持）
	static constexpr float BaseMaxPlayerHPConst    = 70.f;  // MaxPlayerHP UPROPERTY のデフォルト値
	static constexpr float BaseGemPickupRadiusConst= 30.f;  // GemPickupRadius UPROPERTY のデフォルト値

	// ---- 状態 ----
	FVector2D             PlayerPos;
	FVector2D             PlayerVel;
	float                 PlayerHP         = 100.f;
	float                 PlayerXP         = 0.f;
	int32                 PlayerLevel      = 0;
	FWeaponSlot           WeaponSlots[SurvivorsGameConstants::MaxWeaponSlots];
	FPassiveSlot          PassiveSlots[SurvivorsGameConstants::MaxPassiveSlots];
	FPassiveEffects       CachedPassiveEffects;

	// シールド状態（Laurel 用）
	float                 PlayerShieldTimer = 0.f;
	bool                  bShieldActive     = false;

	// リバイバル（Tirajisú 用）
	int32                 MaxRevivalCount   = 0;
	int32                 UsedRevivalCount  = 0;

	// 敵 UniqueId カウンタ（スポーン時に採番）
	int32                 NextEnemyId       = 0;

	// ジェム UniqueId カウンタ（DropGem 時に採番）
	int32                 NextGemId         = 0;

	// フロアアイテム・特殊アイテム・破壊可能オブジェクト（PR2 で本実装）
	TArray<FFloorPickupState>   FloorPickups;
	TArray<FSpecialPickupState> SpecialPickups;
	TArray<FDestructibleState>  Destructibles;

	TArray<FGemState>     Gems;
	TArray<FEnemyState>   Enemies;
	float                 ElapsedTime      = 0.f;
	float                 SpawnAccumulator = 0.f;
	bool                  bBossSpawned     = false;
	float                 LastReward       = 0.f;
	bool                  bDone            = false;
	bool                  bTruncated       = false;
	FRandomStream         RandStream;
	FSurvivorsSpawnDebug  LastSpawnDebug;

	// ---- WallActors (BeginPlay で自動収集) ----
	UPROPERTY()
	TArray<TObjectPtr<AWallActor>> WallActors;

	UPROPERTY(VisibleAnywhere, Category = "Survivors|Components")
	TObjectPtr<USurvivorsPlayerComponent> PlayerComponent;

	UPROPERTY(VisibleAnywhere, Category = "Survivors|Components")
	TObjectPtr<USurvivorsGemComponent> GemComponent;

	UPROPERTY(VisibleAnywhere, Category = "Survivors|Components")
	TObjectPtr<USurvivorsEnemyComponent> EnemyComponent;

	UPROPERTY(VisibleAnywhere, Category = "Survivors|Components")
	TObjectPtr<USurvivorsSpawnComponent> SpawnComponent;

	UPROPERTY(VisibleAnywhere, Category = "Survivors|Components")
	TObjectPtr<USurvivorsCollisionComponent> CollisionComponent;

	UPROPERTY(VisibleAnywhere, Category = "Survivors|Components")
	TObjectPtr<USurvivorsObservationComponent> ObservationComponent;

	UPROPERTY(VisibleAnywhere, Category = "Survivors|Components")
	TObjectPtr<USurvivorsWeaponComponent> WeaponComponent;

	UPROPERTY(VisibleAnywhere, Category = "Survivors|Components")
	TObjectPtr<USurvivorsPickupComponent> PickupComponent;

	// ---- 内部メソッド ----
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

	// 後方互換（既存の ApplyAuraDamage は WeaponComponent に移管されたが宣言は残す）
	void      ApplyAuraDamage();

	// HitFrame Finalize
	void FinalizePendingEnemies();
	void FinalizePickupRemovals();

	// テーブル初期化
	void  InitDefaultEnemyTable();
	void  InitDefaultSpawnWaves();

	// スポーン補助
	const FSpawnWave* GetCurrentWave() const;
	int32             GetCurrentWaveIndex() const;
	bool              BuildSpawnWeights(const FSpawnWave& Wave, TArray<FEnemySpawnWeight>& OutWeights, bool& bOutUsedCurriculumPool) const;
	int32             SelectTypeByWeight(const TArray<FEnemySpawnWeight>& Weights);

	// 敵タイプ別パラメータ取得
	float GetEnemySpeed(int32 TypeId) const;
	float GetEnemyTypeMaxHP(int32 TypeId) const;

	// XP 処理
	float XPRequiredForLevel(int32 Level) const;
	float CumulativeXPForLevel(int32 Level) const;
	void  ProcessXPGain(float Amount);
	void  OnLevelUp(int32 NextLevel);

	mutable int32 CachedObsDim = -1;
};
