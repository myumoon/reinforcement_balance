#include "Survivors/Logic/SurvivorsGame.h"
#include "Survivors/Logic/SurvivorsCollisionComponent.h"
#include "Survivors/Logic/SurvivorsEnemyComponent.h"
#include "Survivors/Logic/SurvivorsGemComponent.h"
#include "Survivors/Logic/SurvivorsObservationComponent.h"
#include "Survivors/Logic/SurvivorsPickupComponent.h"
#include "Survivors/Logic/SurvivorsPlayerComponent.h"
#include "Survivors/Logic/SurvivorsSpawnComponent.h"
#include "Survivors/Logic/Weapons/SurvivorsWeaponComponent.h"
#include "Survivors/View/WallActor.h"

// ---- 加重サンプリングヘルパー -----------------------------------------------

/**
 * WeaponWeights から加重ランダムサンプリング（1つの武器IDを返す）。
 * weights が空または合計 0 の場合は -1 を返す。
 */
static int32 WeightedSampleWeaponId(const TMap<int32, float>& Weights, FRandomStream& RandStream)
{
	float Total = 0.f;
	for (const auto& Pair : Weights) Total += Pair.Value;
	if (Total <= 0.f) return -1;

	float Rand = RandStream.FRandRange(0.f, Total);
	float Cumulative = 0.f;
	for (const auto& Pair : Weights)
	{
		Cumulative += Pair.Value;
		if (Rand <= Cumulative) return Pair.Key;
	}
	// フォールバック（浮動小数点誤差対策）
	return Weights.CreateConstIterator()->Key;
}

// -----------------------------------------------------------------------------

ASurvivorsGame::ASurvivorsGame()
{
	PrimaryActorTick.bCanEverTick = false;

	PlayerComponent  = CreateDefaultSubobject<USurvivorsPlayerComponent>(TEXT("PlayerComponent"));
	GemComponent     = CreateDefaultSubobject<USurvivorsGemComponent>(TEXT("GemComponent"));
	EnemyComponent   = CreateDefaultSubobject<USurvivorsEnemyComponent>(TEXT("EnemyComponent"));
	SpawnComponent   = CreateDefaultSubobject<USurvivorsSpawnComponent>(TEXT("SpawnComponent"));
	CollisionComponent = CreateDefaultSubobject<USurvivorsCollisionComponent>(TEXT("CollisionComponent"));
	ObservationComponent = CreateDefaultSubobject<USurvivorsObservationComponent>(TEXT("ObservationComponent"));
	WeaponComponent  = CreateDefaultSubobject<USurvivorsWeaponComponent>(TEXT("WeaponComponent"));
	PickupComponent  = CreateDefaultSubobject<USurvivorsPickupComponent>(TEXT("PickupComponent"));

	PlayerComponent->Initialize(this);
	GemComponent->Initialize(this);
	EnemyComponent->Initialize(this);
	SpawnComponent->Initialize(this);
	CollisionComponent->Initialize(this);
	ObservationComponent->Initialize(this);
	WeaponComponent->Initialize(this);
	PickupComponent->Initialize(this);

	SpawnComponent->InitDefaultEnemyTable();
	SpawnComponent->InitDefaultSpawnWaves();
}

void ASurvivorsGame::InitDefaultSpawnWaves()
{
	SpawnComponent->InitDefaultSpawnWaves();
}

void ASurvivorsGame::InitDefaultEnemyTable()
{
	SpawnComponent->InitDefaultEnemyTable();
}

void ASurvivorsGame::BeginPlay()
{
	Super::BeginPlay();

	CollisionComponent->CollectWallActors();

	// Logic を初期化: WallActors を FBox2D に変換して Config に渡す
	{
		FSurvivorsGameLogicConfig Cfg;
		Cfg.FieldHalfSize  = FieldHalfSize;
		Cfg.SimToUE        = SimToUE;
		Cfg.SpawnWaves     = SpawnWaves;
		Cfg.EnemyTypeTable = EnemyTypeTable;

		// WallActors → FBox2D 変換
		for (const TObjectPtr<AWallActor>& Wall : WallActors)
		{
			if (Wall)
				Cfg.WallBounds.Add(Wall->GetSimBounds(SimToUE));
		}

		Logic.Initialize(Cfg);
	}

	ResetState(TOptional<int32>());
}

// ---- スキーマ ----------------------------------------------------------------

TArray<FSurvivorsObsSegment> ASurvivorsGame::GetObsSchema() const
{
	return Logic.GetObsSchema();
}

FString ASurvivorsGame::GetObsSchemaHash() const
{
	return Logic.GetObsSchemaHash();
}

int32 ASurvivorsGame::GetObsDim() const
{
	if (CachedObsDim >= 0) return CachedObsDim;
	CachedObsDim = Logic.GetObsDim();
	return CachedObsDim;
}

void ASurvivorsGame::ResetState(TOptional<int32> Seed)
{
	// Phase 3: Logic に完全委譲
	SyncConfigToLogic();
	Logic.Reset(Seed);
	PhysicsAccumTime = 0.f;  // variable frame rate 用アキュムレータはASurvivorsGame側でリセット
}


void ASurvivorsGame::PhysicsStep(int32 ActionIdx)
{
	// Phase 3: Logic に完全委譲
	Logic.PhysicsStep(ActionIdx);
}

void ASurvivorsGame::StepWithDeltaTime(int32 ActionIdx, float DeltaTime)
{
	if (!bVariableFrameRate)
	{
		PhysicsStep(ActionIdx);
		return;
	}

	PhysicsAccumTime += DeltaTime;
	while (PhysicsAccumTime >= SurvivorsGameConstants::PhysicsDt)
	{
		PhysicsStep(ActionIdx);
		PhysicsAccumTime -= SurvivorsGameConstants::PhysicsDt;
	}
}

void ASurvivorsGame::FinalizePendingEnemies()
{
	for (int32 i = Enemies.Num() - 1; i >= 0; --i)
	{
		if (Enemies[i].bPendingRemove)
		{
			const bool bDropsChest = EnemyTypeTable.IsValidIndex(Enemies[i].TypeId)
				&& EnemyTypeTable[Enemies[i].TypeId].bIsBoss;
			if (bDropsChest)
			{
				FSpecialPickupState Chest;
				Chest.Pos = Enemies[i].Pos;
				Chest.Type = ESpecialPickupType::TreasureChest;
				Chest.bActive = true;
				SpecialPickups.Add(Chest);
			}
			else
			{
				GemComponent->DropGem(Enemies[i].TypeId, Enemies[i].Pos);
			}
			Enemies.RemoveAt(i);
		}
	}
}

void ASurvivorsGame::FinalizePickupRemovals()
{
	for (int32 i = Gems.Num() - 1; i >= 0; --i)
	{
		if (Gems[i].bPendingRemove) Gems.RemoveAt(i);
	}
}

TArray<float> ASurvivorsGame::GetObservation() const { return Logic.GetObservation(); }
float ASurvivorsGame::GetReward()     const { return Logic.GetReward(); }
bool  ASurvivorsGame::IsDone()        const { return Logic.IsDone(); }
bool  ASurvivorsGame::IsTruncated()   const { return Logic.IsTruncated(); }

FString ASurvivorsGame::GetSpawnDebugJson() const
{
	return FString::Printf(
		TEXT("{\"elapsed_time\":%.3f,\"max_episode_time\":%.3f,\"enemy_count\":%d,\"current_wave_index\":%d,"
			 "\"min_active_enemies\":%d,\"max_active_enemies\":%d,"
			 "\"effective_min_enemies\":%d,\"effective_max_enemies\":%d,"
			 "\"max_enemy_type_id\":%d,\"allowed_spawn_type_count\":%d,"
			 "\"spawn_accumulator\":%.3f,\"has_current_wave\":%s,"
			 "\"used_curriculum_enemy_pool\":%s,\"spawn_blocked\":%s,\"truncated\":%s}"),
		LastSpawnDebug.ElapsedTime,
		LastSpawnDebug.MaxEpisodeTime,
		LastSpawnDebug.EnemyCount,
		LastSpawnDebug.CurrentWaveIndex,
		LastSpawnDebug.MinActiveEnemies,
		LastSpawnDebug.MaxActiveEnemies,
		LastSpawnDebug.EffectiveMinEnemies,
		LastSpawnDebug.EffectiveMaxEnemies,
		LastSpawnDebug.MaxEnemyTypeId,
		LastSpawnDebug.AllowedSpawnTypeCount,
		LastSpawnDebug.SpawnAccumulator,
		LastSpawnDebug.bHasCurrentWave ? TEXT("true") : TEXT("false"),
		LastSpawnDebug.bUsedCurriculumEnemyPool ? TEXT("true") : TEXT("false"),
		LastSpawnDebug.bSpawnBlocked ? TEXT("true") : TEXT("false"),
		LastSpawnDebug.bTruncated ? TEXT("true") : TEXT("false"));
}

FVector2D ASurvivorsGame::GetItemPos(int32 i) const { return Logic.GetItemPos(i); }
EGemType  ASurvivorsGame::GetItemGemType(int32 i) const { return Logic.GetItemGemType(i); }

float ASurvivorsGame::GetAuraSize() const
{
	// 後方互換: スロット0〜5 から Garlic/SoulEater スロットを探して半径を返す
	for (int32 s = 0; s < MaxWeaponSlots; ++s)
	{
		const FWeaponSlot& Slot = WeaponSlots[s];
		if (Slot.Type == EWeaponType::Garlic || Slot.Type == EWeaponType::SoulEater)
		{
			const int32 Lv = FMath::Clamp(Slot.Level.Value, 1, MaxWeaponLevel);
			if (Slot.Type == EWeaponType::SoulEater)
				return SurvivorsGameConstants::SoulEaterTable[Lv - 1].AreaRadius.Value;
			else
				return SurvivorsGameConstants::GarlicTable[Lv - 1].AreaRadius.Value;
		}
	}
	return 0.f;
}

// ---- プロジェクタイル / GroundZone アクセサ --------------------------------

int32 ASurvivorsGame::GetProjectileCount() const
{
	return WeaponComponent ? WeaponComponent->GetProjectileCount() : 0;
}

FVector2D ASurvivorsGame::GetProjectilePos(int32 i) const
{
	return WeaponComponent ? WeaponComponent->GetProjectilePos(i) : FVector2D::ZeroVector;
}

FSimRadius ASurvivorsGame::GetProjectileRadius(int32 i) const
{
	return WeaponComponent ? WeaponComponent->GetProjectileRadius(i) : FSimRadius(0.f);
}

EWeaponType ASurvivorsGame::GetProjectileWeaponType(int32 i) const
{
	return WeaponComponent ? WeaponComponent->GetProjectileWeaponType(i) : EWeaponType::None;
}

float ASurvivorsGame::GetProjectileBoxHalfWidth(int32 i) const
{
	return WeaponComponent ? WeaponComponent->GetProjectileBoxHalfWidth(i) : 0.f;
}

int32 ASurvivorsGame::GetGroundZoneCount() const
{
	return WeaponComponent ? WeaponComponent->GetGroundZoneCount() : 0;
}

FVector2D ASurvivorsGame::GetGroundZonePos(int32 i) const
{
	return WeaponComponent ? WeaponComponent->GetGroundZonePos(i) : FVector2D::ZeroVector;
}

float ASurvivorsGame::GetGroundZoneRadius(int32 i) const
{
	return WeaponComponent ? WeaponComponent->GetGroundZoneRadius(i) : 0.f;
}

EWeaponType ASurvivorsGame::GetGroundZoneWeaponType(int32 i) const
{
	return WeaponComponent ? WeaponComponent->GetGroundZoneWeaponType(i) : EWeaponType::None;
}

bool ASurvivorsGame::IsGroundZoneWarning(int32 i) const
{
	return WeaponComponent ? WeaponComponent->IsGroundZoneWarning(i) : false;
}

int32 ASurvivorsGame::GetOrbitOrbCount() const
{
	return WeaponComponent ? WeaponComponent->GetOrbitOrbCount() : 0;
}

FVector2D ASurvivorsGame::GetOrbitOrbPos(int32 i) const
{
	return WeaponComponent ? WeaponComponent->GetOrbitOrbPos(i) : FVector2D::ZeroVector;
}

EWeaponType ASurvivorsGame::GetOrbitOrbWeaponType(int32 i) const
{
	return WeaponComponent ? WeaponComponent->GetOrbitOrbWeaponType(i) : EWeaponType::None;
}

float ASurvivorsGame::GetOrbitOrbVisualRadius(int32 i) const
{
	return WeaponComponent ? WeaponComponent->GetOrbitOrbVisualRadius(i) : 0.f;
}

// ---- 内部ユーティリティ -------------------------------------------------------

float ASurvivorsGame::GetEnemySpeed(int32 TypeId) const
{
	return EnemyComponent->GetEnemySpeed(TypeId);
}

float ASurvivorsGame::GetEnemyTypeMaxHP(int32 TypeId) const
{
	return EnemyComponent->GetEnemyTypeMaxHP(TypeId);
}

float ASurvivorsGame::XPRequiredForLevel(int32 Level) const
{
	return PlayerComponent->XPRequiredForLevel(Level);
}

float ASurvivorsGame::CumulativeXPForLevel(int32 Level) const
{
	return PlayerComponent->CumulativeXPForLevel(Level);
}

void ASurvivorsGame::ProcessXPGain(float Amount)
{
	PlayerComponent->ProcessXPGain(Amount);
}

void ASurvivorsGame::OnLevelUp(int32 NextLevel)
{
	PlayerComponent->OnLevelUp(NextLevel);
}

FVector2D ASurvivorsGame::RandomInsideField()
{
	return SpawnComponent->RandomInsideField();
}

FVector2D ASurvivorsGame::RandomOnEdge()
{
	return SpawnComponent->RandomOnEdge();
}

FVector2D ASurvivorsGame::RandomSpawnPos()
{
	return SpawnComponent->RandomSpawnPos();
}

int32 ASurvivorsGame::SelectTypeByWeight(const TArray<FEnemySpawnWeight>& Weights)
{
	return SpawnComponent->SelectTypeByWeight(Weights);
}

const FSpawnWave* ASurvivorsGame::GetCurrentWave() const
{
	return SpawnComponent->GetCurrentWave();
}

int32 ASurvivorsGame::GetCurrentWaveIndex() const
{
	return SpawnComponent->GetCurrentWaveIndex();
}

bool ASurvivorsGame::BuildSpawnWeights(const FSpawnWave& Wave, TArray<FEnemySpawnWeight>& OutWeights, bool& bOutUsedCurriculumPool) const
{
	return SpawnComponent->BuildSpawnWeights(Wave, OutWeights, bOutUsedCurriculumPool);
}

void ASurvivorsGame::SpawnEnemy(const FSpawnWave& Wave)
{
	SpawnComponent->SpawnEnemy(Wave);
}

void ASurvivorsGame::SpawnBoss()
{
	SpawnComponent->SpawnBoss();
}

void ASurvivorsGame::UpdateEnemies()
{
	EnemyComponent->UpdateEnemies();
}

void ASurvivorsGame::ApplyAuraDamage()
{
	ensureMsgf(false, TEXT("ApplyAuraDamage() is deprecated. Use WeaponComponent->TickAllWeapons() directly."));
	if (WeaponComponent)
	{
		WeaponComponent->TickAllWeapons(SurvivorsGameConstants::PhysicsDt);
	}
}

void ASurvivorsGame::DropGem(int32 TypeId, FVector2D Pos)
{
	GemComponent->DropGem(TypeId, Pos);
}

void ASurvivorsGame::CheckGemCollections()
{
	GemComponent->CheckCollections();
}

void ASurvivorsGame::ApplyEnemyContactDamage()
{
	EnemyComponent->ApplyContactDamage();
}

void ASurvivorsGame::ResolveWallCollisions()
{
	CollisionComponent->ResolveWallCollisions();
}

float ASurvivorsGame::CastRayToObstacles(FVector2D Origin, FVector2D Dir) const
{
	return CollisionComponent->CastRayToObstacles(Origin, Dir);
}

TMap<int32, int32> ASurvivorsGame::GetEnemyCountByType() const
{
	TMap<int32, int32> Result;
	for (const FEnemyState& E : Enemies)
	{
		if (!E.bPendingRemove)
		{
			Result.FindOrAdd(E.TypeId)++;
		}
	}
	return Result;
}

float ASurvivorsGame::GetXPRequiredForNextLevel() const
{
	return XPRequiredForLevel(PlayerLevel + 1);
}

float ASurvivorsGame::GetCurrentLevelXP() const
{
	return PlayerXP - CumulativeXPForLevel(PlayerLevel);
}

int32 ASurvivorsGame::GetPassiveItemMaxLevel(EPassiveItemType Type) const
{
	const int32 TypeIndex = static_cast<int32>(Type);
	if (TypeIndex >= 0 && TypeIndex < UE_ARRAY_COUNT(SurvivorsGameConstants::PassiveMaxLevel))
	{
		return SurvivorsGameConstants::PassiveMaxLevel[TypeIndex];
	}
	return 0;
}

FString ASurvivorsGame::GetEnemyTypeDebugLabel(int32 TypeId) const
{
	if (EnemyTypeTable.IsValidIndex(TypeId) && !EnemyTypeTable[TypeId].Name.IsEmpty())
	{
		return FString::Printf(TEXT("%s(ID:%d)"), *EnemyTypeTable[TypeId].Name, TypeId);
	}
	return FString::Printf(TEXT("ID:%d"), TypeId);
}

bool ASurvivorsGame::IsOnScreen(FVector2D WorldPos) const
{
	const FVector2D Rel = WorldPos - PlayerPos;
	return FMath::Abs(Rel.X) <= SurvivorsGameConstants::ScreenHalfWidthU
		&& FMath::Abs(Rel.Y) <= SurvivorsGameConstants::ScreenHalfHeightU;
}

// ---- FSurvivorsGameLogic ファサード -----------------------------------------

void ASurvivorsGame::SyncConfigToLogic()
{
	FSurvivorsGameLogicConfig Cfg;

	// ---- フィールド設定 ----
	Cfg.FieldHalfSize       = FieldHalfSize;
	Cfg.SimToUE             = SimToUE;
	Cfg.bVariableFrameRate  = bVariableFrameRate;

	// ---- 敵設定 ----
	Cfg.MinActiveEnemies    = MinActiveEnemies;
	Cfg.MaxActiveEnemies    = MaxActiveEnemies;
	Cfg.SpawnRateMult       = SpawnRateMult;
	Cfg.MaxEnemyTypeId      = MaxEnemyTypeId;
	Cfg.EnemyHPScale        = EnemyHPScale;
	Cfg.EnemyDamageScale    = EnemyDamageScale;
	Cfg.EnemySpeedMult      = EnemySpeedMult;
	Cfg.SpawnMinDistance    = SpawnMinDistance;
	Cfg.SpawnMaxDistance    = SpawnMaxDistance;
	Cfg.BossSpawnTime       = BossSpawnTime;

	// ---- プレイヤー設定 ----
	Cfg.MaxPlayerHP         = MaxPlayerHP;
	Cfg.MoveSpeed           = MoveSpeed;
	Cfg.PlayerRadius        = PlayerRadius;
	Cfg.GemPickupRadius     = GemPickupRadius;
	Cfg.FloorPickupRadius   = FloorPickupRadius;

	// ---- 報酬設定 ----
	Cfg.AliveReward         = AliveReward;
	Cfg.ItemReward          = ItemReward;
	Cfg.KillReward          = KillReward;
	Cfg.MaxEpisodeTime      = MaxEpisodeTime;

	// ---- 時間スケーリング ----
	Cfg.bTimeScalingEnabled  = bTimeScalingEnabled;
	Cfg.HPScaleRatePerMin    = HPScaleRatePerMin;
	Cfg.DamageScaleRatePerMin = DamageScaleRatePerMin;

	// ---- 訓練用パラメータ拡張 ----
	Cfg.WeaponPoolMode       = WeaponPoolMode;
	Cfg.AllowedWeaponTypes   = AllowedWeaponTypes;
	Cfg.WeaponWeights        = WeaponWeights;
	Cfg.bEnablePassives      = bEnablePassives;
	Cfg.bEnableEvolutions    = bEnableEvolutions;
	Cfg.ReplayOldPhaseFraction = ReplayOldPhaseFraction;
	Cfg.StartingWeaponMode   = StartingWeaponMode;

	// ---- RSI オーバーライド ----
	Cfg.InitialElapsedTime   = InitialElapsedTime;
	Cfg.bHasInitialOverride  = bHasInitialOverride;
	for (const FWeaponSlotOverride& W : InitialWeaponSlots)
		Cfg.InitialWeaponSlots.Add({ W.WeaponId, W.Level });
	for (const FPassiveSlotOverride& P : InitialPassiveSlots)
		Cfg.InitialPassiveSlots.Add({ P.PassiveId, P.Level });

	// ---- 静的テーブル（BeginPlay で設定済みのもの） ----
	Cfg.SpawnWaves     = SpawnWaves;
	Cfg.EnemyTypeTable = EnemyTypeTable;

	// WallBounds は BeginPlay 時に一度だけ設定する（ここでは更新しない）
	// WallActors → FBox2D 変換は BeginPlay 時に Logic.Initialize() で行う

	Logic.ApplyConfig(Cfg);
}
