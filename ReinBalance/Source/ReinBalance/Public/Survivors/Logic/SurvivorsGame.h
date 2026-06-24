#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Survivors/Logic/SurvivorsGameConstants.h"
#include "Survivors/Logic/SurvivorsGameLogic.h"
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
 * 観測: 740次元（GetObsSchema() のセグメント合計）
 */
UCLASS()
class REINBALANCE_API ASurvivorsGame : public AActor
{
	GENERATED_BODY()

public:
	ASurvivorsGame();

	/** 離散行動 (0〜8) を受けて 1 物理ステップ進める */
	UFUNCTION(BlueprintCallable, Category = "Survivors|Control")
	void PhysicsStep(int32 ActionIdx);

	/**
	 * 可変フレームレート対応版。bVariableFrameRate が true の場合、
	 * DeltaTime を蓄積して PhysicsDt 単位でステップを進める。
	 * false の場合は PhysicsStep を 1 回呼ぶだけ（従来動作）。
	 */
	UFUNCTION(BlueprintCallable, Category = "Survivors|Control")
	void StepWithDeltaTime(int32 ActionIdx, float DeltaTime);

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

	FSurvivorsSpawnDebug GetSpawnDebug() const { return Logic.GetSpawnDebug(); }
	FString GetSpawnDebugJson() const;

	// ---- ビュー / デバッグ向けアクセサー ----

	UFUNCTION(BlueprintPure, Category = "Survivors|Config")
	FVector2D GetPlayerPos()   const { return Logic.GetPlayerPos(); }
	/** Camera Z=2000 基準のスクリーン内判定（±400u×±225u）
	 *  ターゲット武器は画面内の敵のみを狙い、画面外敵は対象外にする。 */
	bool IsOnScreen(FVector2D WorldPos) const;

	FVector2D GetPlayerVel()   const { return Logic.GetPlayerVel(); }
	float     GetPlayerHP()    const { return Logic.GetPlayerHP(); }
	float     GetMaxPlayerHP() const { return Logic.GetMaxPlayerHP(); }

	/** 後方互換: Garlic オーラ半径（WeaponSlots から取得）。View の DrawAura 用 */
	float     GetAuraSize()    const;

	const FWeaponSlot& GetWeaponSlot(int32 Idx) const { return Logic.GetWeaponSlot(Idx); }
	bool               IsShieldActive()          const { return Logic.IsShieldActive(); }

	// ジェムアクセサ
	int32     GetItemCount()       const { return Logic.GetItemCount(); }
	FVector2D GetItemPos(int32 i)  const;
	EGemType  GetItemGemType(int32 i) const;

	// 敵アクセサ
	int32     GetEnemyCount()         const { return Logic.GetEnemyCount(); }
	FVector2D GetEnemyPos(int32 i)    const { return Logic.GetEnemyPos(i); }
	int32     GetEnemyType(int32 i)   const { return Logic.GetEnemyType(i); }
	float     GetEnemyHP(int32 i)     const { return Logic.GetEnemyHP(i); }
	float     GetEnemyMaxHP(int32 i)  const { return Logic.GetEnemyMaxHP(i); }

	// プロジェクタイルアクセサ（WeaponComponent 経由）
	int32     GetProjectileCount()              const;
	FVector2D GetProjectilePos(int32 i)         const;
	FSimRadius GetProjectileRadius(int32 i)     const;
	EWeaponType GetProjectileWeaponType(int32 i)const;
	float     GetProjectileBoxHalfWidth(int32 i) const;

	// グラウンドゾーンアクセサ（WeaponComponent 経由）
	int32     GetGroundZoneCount()              const;
	FVector2D GetGroundZonePos(int32 i)         const;
	float     GetGroundZoneRadius(int32 i)      const;
	EWeaponType GetGroundZoneWeaponType(int32 i)const;
	bool      IsGroundZoneWarning(int32 i)      const;

	// 軌道オーブアクセサ（KingBible / Peachone / EbonyWings / Vandalier）
	int32       GetOrbitOrbCount()                    const;
	FVector2D   GetOrbitOrbPos(int32 i)               const;
	EWeaponType GetOrbitOrbWeaponType(int32 i)        const;
	float       GetOrbitOrbVisualRadius(int32 i)      const;

	// フロアアイテムアクセサ
	int32            GetFloorPickupCount()           const { return Logic.GetFloorPickupCount(); }
	FVector2D        GetFloorPickupPos(int32 i)      const { return Logic.GetFloorPickupPos(i); }
	EFloorPickupType GetFloorPickupType(int32 i)     const { return Logic.GetFloorPickupType(i); }

	// 特殊アイテムアクセサ
	int32              GetSpecialPickupCount()           const { return Logic.GetSpecialPickupCount(); }
	FVector2D          GetSpecialPickupPos(int32 i)      const { return Logic.GetSpecialPickupPos(i); }
	ESpecialPickupType GetSpecialPickupType(int32 i)     const { return Logic.GetSpecialPickupType(i); }

	// 破壊可能オブジェクトアクセサ
	int32     GetDestructibleCount()          const { return Logic.GetDestructibleCount(); }
	FVector2D GetDestructiblePos(int32 i)     const { return Logic.GetDestructiblePos(i); }
	bool      IsDestructibleActive(int32 i)   const { return Logic.IsDestructibleActive(i); }

	// ---- ビュー / デバッグ向けアクセサー ----

	float              GetPlayerXP()              const { return Logic.GetPlayerXP(); }
	int32              GetPlayerLevel()            const { return Logic.GetPlayerLevel(); }
	float              GetPlayerShieldTimer()      const { return Logic.GetPlayerShieldTimer(); }
	float              GetElapsedTime()            const { return Logic.GetElapsedTime(); }
	float              GetLastReward()             const { return Logic.GetLastReward(); }
	float              GetEpisodeBaseReward()      const { return Logic.GetEpisodeBaseReward(); }
	int32              GetEpisodeStepCount()       const { return Logic.GetEpisodeStepCount(); }
	/** AliveReward 分を除いた撃破・収集による実質スコア（active_score に相当） */
	float              GetEpisodeActiveScore()     const { return Logic.GetEpisodeActiveScore(); }
	/** Idx は 0..MaxPassiveSlots-1。範囲チェックは呼び出し側で行う */
	const FPassiveSlot& GetPassiveSlot(int32 Idx) const { return Logic.GetPassiveSlot(Idx); }

	/** パッシブ効果キャッシュ（View からの参照用） */
	const FPassiveEffects& GetCachedPassiveEffects() const { return Logic.GetCachedPassiveEffects(); }

	/** TypeId → 生存数のマップ。デバッグ表示用（毎フレームの呼び出しを想定） */
	TMap<int32, int32> GetEnemyCountByType() const;

	/** 現在レベルから次レベルへ必要な XP。デバッグ表示用 */
	float GetXPRequiredForNextLevel() const;

	/** 現在レベル内で取得済みの XP（累積 XP から現在レベル開始時の XP を引いた値）。デバッグ表示用 */
	float GetCurrentLevelXP() const;

	/** パッシブアイテムの最大レベル。デバッグ表示用 */
	int32 GetPassiveItemMaxLevel(EPassiveItemType Type) const;

	/** デバッグ表示用ラベル "Name(ID:n)" を返す。未設定・範囲外なら "ID:n" */
	FString GetEnemyTypeDebugLabel(int32 TypeId) const;

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

	/**
	 * true にすると StepWithDeltaTime() が DeltaTime を蓄積してステップ数を調整し、
	 * FPS によらず一定の実ゲーム速度を保つ。
	 * false（デフォルト）は従来の「毎フレーム 1 ステップ」動作を維持する。
	 */
	UPROPERTY(EditAnywhere, Category = "Survivors|Config")
	bool bVariableFrameRate = false;

	/** WallActorがある場合はBeginPlay時に自動計算される（sim units）。WallActorがない場合のフォールバック値。 */
	UPROPERTY(EditAnywhere, Category = "Survivors|Config")
	float FieldHalfSize = 1000.f;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Survivors|Config")
	float SimToUE = 5.f;

	UPROPERTY(EditAnywhere, Category = "Survivors|Config")
	int32 MinActiveEnemies = 0;

	UPROPERTY(EditAnywhere, Category = "Survivors|Config")
	int32 MaxActiveEnemies = 600;

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
	float SpawnMinDistance = 500.f;

	UPROPERTY(EditAnywhere, Category = "Survivors|Spawn")
	float SpawnMaxDistance = 700.f;

	UPROPERTY(EditAnywhere, Category = "Survivors|Spawn")
	float BossSpawnTime = 300.f;

	UPROPERTY(EditAnywhere, Category = "Survivors|Spawn")
	TArray<FSpawnWave> SpawnWaves;

	UPROPERTY(EditAnywhere, Category = "Survivors|Config")
	float EnemySpeedMult = 1.0f;

	// ---- プレイヤー設定 ----

	UPROPERTY(EditAnywhere, Category = "Survivors|Player")
	float MaxPlayerHP = 100.f;

	UPROPERTY(EditAnywhere, Category = "Survivors|Player")
	float MoveSpeed = 80.f;

	UPROPERTY(EditAnywhere, Category = "Survivors|Player")
	float PlayerRadius = 10.f;

	// ---- 敵設定 ----

	UPROPERTY(EditAnywhere, Category = "Survivors|Enemy")
	TArray<FEnemyTypeParams> EnemyTypeTable;

	// ---- 取得設定 ----

	UPROPERTY(BlueprintReadOnly, Category = "Survivors|Item")
	float GemPickupRadius = ASurvivorsGame::BaseGemPickupRadiusConst;
	
	UPROPERTY(EditAnywhere, Category = "Survivors|Item")
	float FloorPickupRadius = 30.f;

	// ---- 時間スケーリング ----

	UPROPERTY(EditAnywhere, Category = "Survivors|TimeScaling")
	bool bTimeScalingEnabled = false;

	// ---- 訓練用パラメータ拡張（/params エンドポイント経由で設定） ----

	/** 武器プール制御モード: "all" / "garlic_only" / "custom" */
	UPROPERTY(EditAnywhere, Category = "Survivors|Train")
	FString WeaponPoolMode = TEXT("all_base");

	/** カスタムモード時の許可武器タイプ ID リスト */
	UPROPERTY(EditAnywhere, Category = "Survivors|Train")
	TArray<int32> AllowedWeaponTypes;

	/** weighted モード時の武器 ID → 重み マップ（重み > 0 の武器のみ保持） */
	UPROPERTY(EditAnywhere, Category = "Survivors|Train")
	TMap<int32, float> WeaponWeights;

	/** パッシブアイテムを有効にするか */
	UPROPERTY(EditAnywhere, Category = "Survivors|Train")
	bool bEnablePassives = true;

	/** 進化システムを有効にするか */
	UPROPERTY(EditAnywhere, Category = "Survivors|Train")
	bool bEnableEvolutions = true;

	/** リプレイ旧フェーズ比率（0.0〜1.0） */
	UPROPERTY(EditAnywhere, Category = "Survivors|Train")
	float ReplayOldPhaseFraction = 0.0f;

	/** 開始武器選択モード: "garlic" / "random" / "custom" */
	UPROPERTY(EditAnywhere, Category = "Survivors|Train")
	FString StartingWeaponMode = TEXT("garlic");

	// RSI: 武器スロット初期化用の構造体
	struct FWeaponSlotOverride
	{
		int32 WeaponId = 0;
		int32 Level    = 1;
	};

	// RSI: パッシブスロット初期化用の構造体
	struct FPassiveSlotOverride
	{
		int32 PassiveId = 0;
		int32 Level     = 1;
	};

	// RSI: リセット時に適用する初期状態オーバーライド（SurvivorsHttpEnvService から設定）
	float                        InitialElapsedTime  = 0.f;
	TArray<FWeaponSlotOverride>  InitialWeaponSlots;
	TArray<FPassiveSlotOverride> InitialPassiveSlots;
	bool                         bHasInitialOverride = false;

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
	friend class USurvivorsWeaponGarlic;
	friend class USurvivorsWeaponWhip;
	friend class USurvivorsWeaponMagicWand;
	friend class USurvivorsWeaponKnife;
	friend class USurvivorsWeaponAxe;
	friend class USurvivorsWeaponCross;
	friend class USurvivorsWeaponKingBible;
	friend class USurvivorsWeaponFireWand;
	friend class USurvivorsWeaponSantaWater;
	friend class USurvivorsWeaponRunetracer;
	friend class USurvivorsWeaponLightningRing;
	friend class USurvivorsWeaponPentagram;
	friend class USurvivorsWeaponPeachone;
	friend class USurvivorsWeaponEbonyWings;
	friend class USurvivorsWeaponVandalier;
	friend class USurvivorsWeaponLaurel;
#if WITH_AUTOMATION_TESTS
	friend struct FSurvivorsGameTestAccess;
#endif
#if WITH_EDITOR
	friend class UDebugSurvivorsSlotComponent;
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
	static constexpr float BaseMaxPlayerHPConst    = SurvivorsGameConstants::StandardMaxPlayerHP;
	static constexpr float BaseGemPickupRadiusConst= SurvivorsGameConstants::BaseGemPickupRadius;

	// ---- 状態 ----
	FVector2D             PlayerPos;
	FVector2D             PlayerVel;
	float                 PlayerHP         = 100.f;
	float                 PlayerXP         = 0.f;
	int32                 PlayerLevel      = 0;
	FWeaponSlot           WeaponSlots[SurvivorsGameConstants::MaxWeaponSlots];
	FPassiveSlot          PassiveSlots[SurvivorsGameConstants::MaxPassiveSlots];
	FPassiveEffects       CachedPassiveEffects;

	/** Orologion グローバルフリーズ終了時刻（-1 = フリーズなし） */
	float                 GlobalFreezeUntilTime = -1.f;

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
	float                 EpisodeBaseReward = 0.f;
	int32                 EpisodeStepCount  = 0;
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

	// StepWithDeltaTime() 用の時間蓄積バッファ
	float PhysicsAccumTime = 0.f;

public:
	// ---- FSurvivorsGameLogic ファサード API ----

	/** ゲームロジックへのポインタ（ParallelFor / テストから参照用） */
	FSurvivorsGameLogic* GetLogic() { return &Logic; }
	const FSurvivorsGameLogic* GetLogic() const { return &Logic; }

	/**
	 * 全 UPROPERTY フィールドを FSurvivorsGameLogicConfig に変換して Logic に同期する。
	 * /params 適用後・ResetState() 前に呼ぶ（ゲームスレッド専用）。
	 */
	void SyncConfigToLogic();

private:
	FSurvivorsGameLogic Logic;
};
