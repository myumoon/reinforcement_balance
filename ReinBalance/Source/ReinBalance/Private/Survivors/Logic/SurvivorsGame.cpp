#include "Survivors/Logic/SurvivorsGame.h"
#include "Survivors/Logic/SurvivorsCollisionComponent.h"
#include "Survivors/Logic/SurvivorsEnemyComponent.h"
#include "Survivors/Logic/SurvivorsGemComponent.h"
#include "Survivors/Logic/SurvivorsObservationComponent.h"
#include "Survivors/Logic/SurvivorsPickupComponent.h"
#include "Survivors/Logic/SurvivorsPlayerComponent.h"
#include "Survivors/Logic/SurvivorsSpawnComponent.h"
#include "Survivors/Logic/Weapons/SurvivorsWeaponComponent.h"

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
	ResetState(TOptional<int32>());
}

// ---- スキーマ ----------------------------------------------------------------

TArray<FSurvivorsObsSegment> ASurvivorsGame::GetObsSchema() const
{
	return ObservationComponent->GetObsSchema();
}

FString ASurvivorsGame::GetObsSchemaHash() const
{
	return ObservationComponent->GetObsSchemaHash();
}

int32 ASurvivorsGame::GetObsDim() const
{
	if (CachedObsDim >= 0) return CachedObsDim;
	if (!ObservationComponent) return 0;
	int32 Total = 0;
	for (const FSurvivorsObsSegment& Seg : ObservationComponent->GetObsSchema())
		Total += Seg.Dim;
	CachedObsDim = Total;
	return CachedObsDim;
}

void ASurvivorsGame::ResetState(TOptional<int32> Seed)
{
	if (Seed.IsSet())
		RandStream.Initialize(Seed.GetValue());
	else
		RandStream.GenerateNewSeed();

	// 武器スロットをリセット
	for (int32 i = 0; i < MaxWeaponSlots; ++i)
	{
		WeaponSlots[i].Type    = EWeaponType::None;
		WeaponSlots[i].Level   = FWeaponLevel(0);
		WeaponSlots[i].Cooldown= FCooldownSeconds(0.f);
	}
	// パッシブスロットをリセット
	for (int32 i = 0; i < MaxPassiveSlots; ++i)
	{
		PassiveSlots[i].Type  = EPassiveItemType::None;
		PassiveSlots[i].Level = 0;
	}
	CachedPassiveEffects = FPassiveEffects();

	// パッシブリセット後にベース値を復元（累積増幅防止）
	MaxPlayerHP      = BaseMaxPlayerHPConst;
	GemPickupRadius  = BaseGemPickupRadiusConst;

	// シールド・リバイバルリセット
	PlayerShieldTimer = 0.f;
	bShieldActive     = false;
	MaxRevivalCount   = 0;
	UsedRevivalCount  = 0;

	// 敵 UniqueId カウンタリセット
	NextEnemyId = 0;

	// ジェム UniqueId カウンタリセット
	NextGemId = 0;

	// フロアアイテムリセット（PR2 で本実装）
	FloorPickups.Empty();
	SpecialPickups.Empty();
	Destructibles.Empty();

	PhysicsAccumTime = 0.f;

	PlayerComponent->Reset();
	GemComponent->Reset();
	EnemyComponent->Reset();
	SpawnComponent->Reset();
	WeaponComponent->Reset();

	// 開始武器: StartingWeaponMode / WeaponPoolMode に基づいて選択
	// WeaponPoolMode の受け付け値（SurvivorsHttpEnvService で正規化済み）:
	//   "garlic_only"         → Garlic のみ
	//   "fixed_subset"        → AllowedWeaponTypes を使用
	//   "all_base"            → 全基本武器（Garlic〜Laurel）
	//   "all_with_evolutions" → all_base と同じ（進化後武器は進化システムで処理）
	//   "weighted"            → fixed_subset 扱い（weights=0 の武器は Python 側で除外済み）
	{
		// 初期武器選択プール: Pentagram・Laurel は経験値ドロップなし/攻撃判定なしのため除外。
		// （レベルアップ選択肢には出現する。PlayerComponent 側の AllowedPool には含まれる）
		static const TArray<EWeaponType> AllBaseWeapons = {
			EWeaponType::Garlic,  EWeaponType::Whip,   EWeaponType::MagicWand,
			EWeaponType::Knife,   EWeaponType::Axe,    EWeaponType::Cross,
			EWeaponType::KingBible, EWeaponType::FireWand, EWeaponType::SantaWater,
			EWeaponType::Runetracer, EWeaponType::LightningRing,
			EWeaponType::Peachone, EWeaponType::EbonyWings,
		};

		EWeaponType StartWeapon = EWeaponType::Garlic;

		// "fixed_subset" と "weighted" は同じプール（AllowedWeaponTypes）を使用
		const bool bUseAllowedSubset =
			(WeaponPoolMode == TEXT("fixed_subset") || WeaponPoolMode == TEXT("weighted"))
			&& AllowedWeaponTypes.Num() > 0;

		// weighted モードかつ重みが設定済みかどうか
		const bool bWeightedMode =
			WeaponPoolMode == TEXT("weighted") && !WeaponWeights.IsEmpty();

		// "pool_random" は "random" と同等に扱う（Python W3/W4/W5/W6 で使用）
		if (StartingWeaponMode.Equals(TEXT("random"), ESearchCase::IgnoreCase) ||
			StartingWeaponMode.Equals(TEXT("pool_random"), ESearchCase::IgnoreCase))
		{
			// weapon_pool_mode に従ったプールからランダム選択
			if (WeaponPoolMode == TEXT("garlic_only"))
			{
				StartWeapon = EWeaponType::Garlic;
			}
			else if (bWeightedMode)
			{
				// weighted モード: WeaponWeights による加重サンプリング
				const int32 SampledId = WeightedSampleWeaponId(WeaponWeights, RandStream);
				StartWeapon = (SampledId > 0)
					? static_cast<EWeaponType>(SampledId)
					: EWeaponType::Garlic;
			}
			else if (bUseAllowedSubset)
			{
				const int32 Idx = RandStream.RandRange(0, AllowedWeaponTypes.Num() - 1);
				StartWeapon = static_cast<EWeaponType>(AllowedWeaponTypes[Idx]);
			}
			else  // "all_base" / "all_with_evolutions" / デフォルト
			{
				const int32 Idx = RandStream.RandRange(0, AllBaseWeapons.Num() - 1);
				StartWeapon = AllBaseWeapons[Idx];
			}
		}
		else if (StartingWeaponMode.Equals(TEXT("garlic"), ESearchCase::IgnoreCase) ||
			WeaponPoolMode == TEXT("garlic_only"))
		{
			StartWeapon = EWeaponType::Garlic;
		}
		else if (StartingWeaponMode.Equals(TEXT("whip"), ESearchCase::IgnoreCase))
		{
			StartWeapon = EWeaponType::Whip;
		}
		else if (bWeightedMode)
		{
			// weighted モード: WeaponWeights による加重サンプリング（deterministic でなく重み比例）
			const int32 SampledId = WeightedSampleWeaponId(WeaponWeights, RandStream);
			StartWeapon = (SampledId > 0)
				? static_cast<EWeaponType>(SampledId)
				: EWeaponType::Garlic;
		}
		else if (bUseAllowedSubset)
		{
			// fixed_subset の先頭を開始武器とする（deterministic）
			StartWeapon = static_cast<EWeaponType>(AllowedWeaponTypes[0]);
		}
		// else: "all_base" / "all_with_evolutions" / デフォルト → Garlic

		WeaponSlots[0].Type  = StartWeapon;
		WeaponSlots[0].Level = FWeaponLevel(1);
		WeaponComponent->EquipWeapon(0, StartWeapon, 1);
	}

	ElapsedTime           = 0.f;
	GlobalFreezeUntilTime = -1.f;
	LastReward        = 0.f;
	EpisodeBaseReward = 0.f;
	EpisodeStepCount  = 0;
	bDone       = false;
	bTruncated  = false;
}

void ASurvivorsGame::PhysicsStep(int32 ActionIdx)
{
	if (bDone || bTruncated) return;
	LastReward = 0.f;

	PlayerComponent->ApplyAction(ActionIdx);
	CollisionComponent->ResolveWallCollisions();

	// Pummarola: HP 再生（RegenPerSec → 毎フレーム回復）
	if (CachedPassiveEffects.RegenPerSec > 0.f)
	{
		PlayerHP = FMath::Min(PlayerHP + CachedPassiveEffects.RegenPerSec * SurvivorsGameConstants::PhysicsDt, MaxPlayerHP);
	}

	ElapsedTime += SurvivorsGameConstants::PhysicsDt;
	EnemyComponent->UpdateEnemies();
	WeaponComponent->TickWeapons(SurvivorsGameConstants::PhysicsDt);

	// WeaponHits（StepSpawn 前の敵のみ）
	CollisionComponent->BuildEnemyGrid();
	{
		FSurvivorsHitFrame HitFrame;
		WeaponComponent->ComputeAllWeaponHits(CollisionComponent, HitFrame);
		WeaponComponent->ApplyWeaponHits(HitFrame);
	}
	FinalizePendingEnemies();

	SpawnComponent->StepSpawn();

	// ContactHits（Spawn 後の全敵）
	CollisionComponent->BuildEnemyGrid();
	{
		FSurvivorsHitFrame HitFrame;
		EnemyComponent->ComputeContactHits(CollisionComponent, HitFrame);
		EnemyComponent->ApplyContactHits(HitFrame);
	}

	// PickupHits（DropGem 済みジェムを含む）
	CollisionComponent->BuildPickupGrid();
	{
		FSurvivorsHitFrame HitFrame;
		GemComponent->ComputePickupHits(CollisionComponent, HitFrame);
		GemComponent->ApplyPickupHits(HitFrame);
	}
	FinalizePickupRemovals();

	// フロアアイテム・特殊アイテム収集
	if (PickupComponent)
	{
		PickupComponent->CheckFloorPickups();
		PickupComponent->CheckSpecialPickups();
	}

	if (PlayerHP <= 0.f && UsedRevivalCount < MaxRevivalCount)
	{
		++UsedRevivalCount;
		PlayerHP = MaxPlayerHP * 0.5f;
	}

	if (PlayerHP <= 0.f)
	{
		bDone = true;
		EpisodeBaseReward += LastReward;
		EpisodeStepCount++;
		return;
	}

	LastReward += AliveReward;
	EpisodeBaseReward += LastReward;
	EpisodeStepCount++;

	if (MaxEpisodeTime > 0.f && ElapsedTime >= MaxEpisodeTime)
	{
		bTruncated = true;
		LastSpawnDebug.bTruncated = true;
	}
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

TArray<float> ASurvivorsGame::GetObservation() const
{
	return ObservationComponent->GetObservation();
}

float ASurvivorsGame::GetReward()     const { return LastReward; }
bool  ASurvivorsGame::IsDone()        const { return bDone; }
bool  ASurvivorsGame::IsTruncated()   const { return bTruncated; }

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

FVector2D ASurvivorsGame::GetItemPos(int32 i) const
{
	return Gems.IsValidIndex(i) ? Gems[i].Pos : FVector2D::ZeroVector;
}

EGemType ASurvivorsGame::GetItemGemType(int32 i) const
{
	return Gems.IsValidIndex(i) ? Gems[i].Type : EGemType::Blue;
}

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
