#include "Survivors/Logic/SurvivorsGameLogic.h"

// Phase 2 暫定実装: FSurvivorsGameLogic は ASurvivorsGame の薄いラッパーとして機能する。
// SetGameFacade() で ASurvivorsGame* を void* として受け取り、ここで再キャストして使う。
// TODO(issue): Phase 3 で Logic に完全移植したら GameFacade 参照を除去する。

// Misc/SecureHash は GetObsSchemaHash で使用
#include "Misc/SecureHash.h"

// Phase 2 暫定: ASurvivorsGame のフルヘッダーをここでのみインクルード
#include "Survivors/Logic/SurvivorsGame.h"

FSurvivorsGameLogic::FSurvivorsGameLogic() = default;
FSurvivorsGameLogic::~FSurvivorsGameLogic() = default;

// ---- 初期化 ---------------------------------------------------------------

void FSurvivorsGameLogic::Initialize(const FSurvivorsGameLogicConfig& Config)
{
	CurrentConfig = Config;
	CachedObsDim  = -1;

	// SpawnWaves / EnemyTypeTable が設定済みならそのまま使う
	// TODO(issue): Phase 3 で InitDefaultEnemyTable/InitDefaultSpawnWaves を自己保有する
}

void FSurvivorsGameLogic::ApplyConfig(const FSurvivorsGameLogicConfig& Config)
{
	// WallBounds / SpawnWaves / EnemyTypeTable は変更しない（BeginPlay 時のみ）
	TArray<FBox2D>           SavedWalls  = MoveTemp(CurrentConfig.WallBounds);
	TArray<FSpawnWave>       SavedWaves  = MoveTemp(CurrentConfig.SpawnWaves);
	TArray<FEnemyTypeParams> SavedTable  = MoveTemp(CurrentConfig.EnemyTypeTable);

	CurrentConfig = Config;

	CurrentConfig.WallBounds    = MoveTemp(SavedWalls);
	CurrentConfig.SpawnWaves    = MoveTemp(SavedWaves);
	CurrentConfig.EnemyTypeTable = MoveTemp(SavedTable);

	CachedObsDim = -1;
}

// ---- 訓練 API（Phase 2: 骨格のみ / Phase 3 で移植） ----------------------

void FSurvivorsGameLogic::Reset(TOptional<int32> Seed)
{
	// TODO(issue): Phase 3 で各コンポーネントのロジックをここに移植する。
	// 現時点ではロジックが ASurvivorsGame 側にあるため、この関数は空。
	// ASurvivorsGame::ResetState() が SyncConfigToLogic() → Logic.Reset() を呼ぶため
	// 状態フィールドの更新は ASurvivorsGame 側で行われる。
}

void FSurvivorsGameLogic::PhysicsStep(int32 ActionIdx)
{
	// TODO(issue): Phase 3 で各コンポーネントのロジックをここに移植する。
}

TArray<float> FSurvivorsGameLogic::GetObservation() const
{
	// TODO(issue): Phase 3 で ObservationComponent のロジックを移植する。
	return {};
}

TArray<FSurvivorsObsSegment> FSurvivorsGameLogic::GetObsSchema() const
{
	using namespace SurvivorsGameConstants;
	// ObservationComponent と同じスキーマを返す（Phase 3 で統合）
	return {
		{ TEXT("player_pos"),                 2  },
		{ TEXT("player_vel"),                 2  },
		{ TEXT("wall_rays"),                  8  },
		{ TEXT("player_hp"),                  1  },
		{ TEXT("shield_active"),              1  },
		{ TEXT("shield_timer_norm"),          1  },
		{ TEXT("revival_remaining_norm"),     1  },
		{ TEXT("armor_flat_norm"),            1  },
		{ TEXT("regen_per_sec_norm"),         1  },
		{ TEXT("passive_effect_summary"),     5  },
		{ TEXT("weapon_slots"),               MaxWeaponSlots * 3 },
		{ TEXT("passive_slots"),              MaxPassiveSlots * 2 },
		{ TEXT("enemy_count"),                1  },
		{ TEXT("elapsed_time"),               1  },
		{ TEXT("xp_progress"),               1  },
		{ TEXT("player_level"),               1  },
		{ TEXT("stage_id_norm"),              1  },
		{ TEXT("red_gem_rel_pos"),            MaxRedGemObs * 2 },
		{ TEXT("green_gem_rel_pos"),          MaxGreenGemObs * 2 },
		{ TEXT("blue_gem_rel_pos"),           MaxBlueGemObs * 2 },
		{ TEXT("gem_pickup_radius"),          1  },
		{ TEXT("enemy_rel_pos"),              MaxEnemyObs * 2 },
		{ TEXT("enemy_vel"),                  MaxEnemyObs * 2 },
		{ TEXT("enemy_type"),                 MaxEnemyObs },
		{ TEXT("enemy_hp"),                   MaxEnemyObs },
		{ TEXT("enemy_frozen"),               MaxEnemyObs },
		{ TEXT("enemy_nearest_dist_16dir"),   EnemyDensityDirCount },
		{ TEXT("enemy_density_near_16dir"),   EnemyDensityDirCount },
		{ TEXT("enemy_density_mid_16dir"),    EnemyDensityDirCount },
		{ TEXT("gem_density_all_16dir"),      GemDensityDirCount * 3 },
		{ TEXT("red_green_gem_density_16dir"),GemDensityDirCount * 3 },
		{ TEXT("projectiles"),                MaxProjectileObs * ProjectileObsStride },
		{ TEXT("floor_pickups"),              MaxFloorPickupObs * 3 },
		{ TEXT("special_pickups"),            MaxSpecialPickupObs * 3 },
		{ TEXT("destructibles"),              MaxDestructibleObs * 2 },
		{ TEXT("weapon_attack_range_norm"),   MaxWeaponSlots },
		{ TEXT("weapon_is_directional"),      MaxWeaponSlots },
		{ TEXT("weapon_category_onehot"),     MaxWeaponSlots * 7 },
	};
}

FString FSurvivorsGameLogic::GetObsSchemaHash() const
{
	using namespace SurvivorsGameConstants;
	FString Schema = FString::Printf(
		TEXT("SurvivorsGame_v794"
		     ",MaxEnemyObs=%d,MaxWeaponSlots=%d,MaxPassiveSlots=%d"
		     ",MaxProjectileObs=%d,ProjectileObsStride=%d,MaxRedGemObs=%d,MaxGreenGemObs=%d,MaxBlueGemObs=%d"
		     ",MaxFloorPickupObs=%d,MaxSpecialPickupObs=%d,MaxDestructibleObs=%d"
		     ",MaxWeaponTypeCountReserved=%d,MaxPassiveTypeCountReserved=%d"
		     ",EnemyDensityDirCount=%d,GemDensityDirCount=%d"
		     ",EnemyNearestDistanceMax=%.0f,GemNearestDistanceMax=%.0f"),
		MaxEnemyObs, MaxWeaponSlots, MaxPassiveSlots,
		MaxProjectileObs, ProjectileObsStride, MaxRedGemObs, MaxGreenGemObs, MaxBlueGemObs,
		MaxFloorPickupObs, MaxSpecialPickupObs, MaxDestructibleObs,
		MaxWeaponTypeCountReserved, MaxPassiveTypeCountReserved,
		EnemyDensityDirCount, GemDensityDirCount,
		EnemyNearestDistanceMax, GemNearestDistanceMax);
	return FMD5::HashAnsiString(*Schema);
}

int32 FSurvivorsGameLogic::GetObsDim() const
{
	if (CachedObsDim >= 0) return CachedObsDim;
	int32 Total = 0;
	for (const FSurvivorsObsSegment& Seg : GetObsSchema())
		Total += Seg.Dim;
	CachedObsDim = Total;
	return CachedObsDim;
}

float FSurvivorsGameLogic::GetReward()      const { return LastReward; }
bool  FSurvivorsGameLogic::IsDone()         const { return bDone; }
bool  FSurvivorsGameLogic::IsTruncated()    const { return bTruncated; }

FString FSurvivorsGameLogic::GetSpawnDebugJson() const
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

// ---- ParallelFor API -------------------------------------------------------

FSurvivorsStepResult FSurvivorsGameLogic::ExecStep(const TArray<float>& Action, int32 Steps)
{
	FSurvivorsStepResult Result;

	// Phase 2 暫定実装: ASurvivorsGame に委譲
	// TODO(issue): Phase 3 で PhysicsStep をここに完全移植する（GameFacade 参照を除去）
	ASurvivorsGame* Game = static_cast<ASurvivorsGame*>(GameFacade);
	if (!Game)
	{
		Result.bDone      = bDone;
		Result.bTruncated = bTruncated;
		Result.Reward     = LastReward;
		return Result;
	}

	const int32 ActionIdx = Action.Num() > 0
		? FMath::Clamp(static_cast<int32>(Action[0]), 0, 8)
		: 8;

	float AccumulatedReward = 0.f;
	for (int32 i = 0; i < Steps; ++i)
	{
		Game->PhysicsStep(ActionIdx);
		AccumulatedReward += Game->GetReward();
		if (Game->IsDone())     { Result.bDone      = true; break; }
		if (Game->IsTruncated()){ Result.bTruncated = true; break; }
	}

	Result.Obs           = Game->GetObservation();
	Result.Reward        = AccumulatedReward;
	Result.SpawnDebugJson = Game->GetSpawnDebugJson();
	return Result;
}

FSurvivorsResetResult FSurvivorsGameLogic::ExecReset(TOptional<int32> Seed)
{
	FSurvivorsResetResult Result;

	// Phase 2 暫定実装: ASurvivorsGame に委譲
	// TODO(issue): Phase 3 で Reset をここに完全移植する（GameFacade 参照を除去）
	ASurvivorsGame* Game = static_cast<ASurvivorsGame*>(GameFacade);
	if (!Game)
	{
		Result.ObsSchemaHash = GetObsSchemaHash();
		return Result;
	}

	Game->ResetState(Seed);
	Result.Obs           = Game->GetObservation();
	Result.ObsSchemaHash = Game->GetObsSchemaHash();
	return Result;
}

// ---- アクセサ ---------------------------------------------------------------

FVector2D FSurvivorsGameLogic::GetItemPos(int32 i) const
{
	return Gems.IsValidIndex(i) ? Gems[i].Pos : FVector2D::ZeroVector;
}

EGemType FSurvivorsGameLogic::GetItemGemType(int32 i) const
{
	return Gems.IsValidIndex(i) ? Gems[i].Type : EGemType::Blue;
}

bool FSurvivorsGameLogic::IsOnScreen(FVector2D WorldPos) const
{
	// Camera Z=2000 基準（ASurvivorsGame と同じ定数）
	static constexpr float HalfW = 400.f;
	static constexpr float HalfH = 225.f;
	const FVector2D Rel = WorldPos - PlayerPos;
	return FMath::Abs(Rel.X) <= HalfW && FMath::Abs(Rel.Y) <= HalfH;
}

TMap<int32, int32> FSurvivorsGameLogic::GetEnemyCountByType() const
{
	TMap<int32, int32> Result;
	for (const FEnemyState& E : Enemies)
	{
		if (!E.bPendingRemove)
			Result.FindOrAdd(E.TypeId)++;
	}
	return Result;
}

float FSurvivorsGameLogic::GetXPRequiredForNextLevel() const
{
	return XPRequiredForLevel(PlayerLevel + 1);
}

float FSurvivorsGameLogic::GetCurrentLevelXP() const
{
	return PlayerXP - CumulativeXPForLevel(PlayerLevel);
}

// TODO(issue): 以下のアクセサは Phase 3 で武器移植が完了したら実装する
// 現時点では Stub（0 / ZeroVector）を返す

int32     FSurvivorsGameLogic::GetProjectileCount()              const { return Projectiles.Num(); }
FVector2D FSurvivorsGameLogic::GetProjectilePos(int32 i)         const { return Projectiles.IsValidIndex(i) ? Projectiles[i].Pos : FVector2D::ZeroVector; }
FSimRadius FSurvivorsGameLogic::GetProjectileRadius(int32 i)     const { return Projectiles.IsValidIndex(i) ? Projectiles[i].Radius : FSimRadius(0.f); }
EWeaponType FSurvivorsGameLogic::GetProjectileWeaponType(int32 i) const { return Projectiles.IsValidIndex(i) ? Projectiles[i].WeaponType : EWeaponType::None; }
float     FSurvivorsGameLogic::GetProjectileBoxHalfWidth(int32 i) const { return 0.f; }

int32     FSurvivorsGameLogic::GetGroundZoneCount()              const { return GroundZones.Num(); }
FVector2D FSurvivorsGameLogic::GetGroundZonePos(int32 i)         const { return GroundZones.IsValidIndex(i) ? GroundZones[i].Pos : FVector2D::ZeroVector; }
float     FSurvivorsGameLogic::GetGroundZoneRadius(int32 i)      const { return GroundZones.IsValidIndex(i) ? GroundZones[i].Radius : 0.f; }
EWeaponType FSurvivorsGameLogic::GetGroundZoneWeaponType(int32 i) const { return GroundZones.IsValidIndex(i) ? GroundZones[i].WeaponType : EWeaponType::None; }
bool      FSurvivorsGameLogic::IsGroundZoneWarning(int32 i)      const { return GroundZones.IsValidIndex(i) ? GroundZones[i].bIsWarning : false; }

int32       FSurvivorsGameLogic::GetOrbitOrbCount()               const { return 0; }
FVector2D   FSurvivorsGameLogic::GetOrbitOrbPos(int32 i)          const { return FVector2D::ZeroVector; }
EWeaponType FSurvivorsGameLogic::GetOrbitOrbWeaponType(int32 i)   const { return EWeaponType::None; }
float       FSurvivorsGameLogic::GetOrbitOrbVisualRadius(int32 i) const { return 0.f; }

FString FSurvivorsGameLogic::GetEnemyTypeDebugLabel(int32 TypeId) const
{
	if (CurrentConfig.EnemyTypeTable.IsValidIndex(TypeId))
	{
		const FString& Name = CurrentConfig.EnemyTypeTable[TypeId].Name;
		if (!Name.IsEmpty())
			return FString::Printf(TEXT("%s(ID:%d)"), *Name, TypeId);
	}
	return FString::Printf(TEXT("ID:%d"), TypeId);
}

int32 FSurvivorsGameLogic::GetPassiveItemMaxLevel(EPassiveItemType Type) const
{
	// TODO(issue): Phase 3 で PlayerComponent のロジックを移植する
	// 現時点では PlayerComponent と同じテーブルを参照できないため、
	// ASurvivorsGame 経由で取得するか、ここに定数テーブルをコピーする
	// 暫定: 既知のパッシブ最大レベルを返す（SurvivorsWikiSpec から移植）
	switch (Type)
	{
	case EPassiveItemType::Spinach:       return 5;
	case EPassiveItemType::Armor:         return 5;
	case EPassiveItemType::HollowHeart:   return 5;
	case EPassiveItemType::Pummarola:     return 5;
	case EPassiveItemType::EmptyTome:     return 5;
	case EPassiveItemType::Candelabrador: return 5;
	case EPassiveItemType::Bracer:        return 5;
	case EPassiveItemType::Spellbinder:   return 5;
	case EPassiveItemType::Duplicator:    return 5;
	case EPassiveItemType::Wings:         return 5;
	case EPassiveItemType::Attractorb:    return 5;
	case EPassiveItemType::Clover:        return 5;
	case EPassiveItemType::Crown:         return 5;
	case EPassiveItemType::StoneMask:     return 0;  // 未実装
	case EPassiveItemType::SkullOManiac:  return 5;
	case EPassiveItemType::Tirajisu:      return 3;
	case EPassiveItemType::TorronasBox:   return 5;
	default:                              return 0;
	}
}

float FSurvivorsGameLogic::GetAuraSize() const
{
	// TODO(issue): Phase 3 で武器スロットから Garlic/SoulEater の AreaRadius を取得する
	return 0.f;
}

// ---- 内部メソッドのスタブ（Phase 3 で実装） ---------------------------------

FVector2D FSurvivorsGameLogic::RandomInsideField()
{
	const float H = CurrentConfig.FieldHalfSize;
	return FVector2D(RandStream.FRandRange(-H, H), RandStream.FRandRange(-H, H));
}

FVector2D FSurvivorsGameLogic::RandomOnEdge()
{
	// TODO(issue): Phase 3 で SurvivorsSpawnComponent のロジックを移植する
	return FVector2D::ZeroVector;
}

FVector2D FSurvivorsGameLogic::RandomSpawnPos()
{
	return FVector2D::ZeroVector;
}

void FSurvivorsGameLogic::SpawnEnemy(const FSpawnWave& Wave)
{
	// TODO(issue): Phase 3 で実装
}

void FSurvivorsGameLogic::SpawnBoss()
{
	// TODO(issue): Phase 3 で実装
}

void FSurvivorsGameLogic::UpdateEnemies()
{
	// TODO(issue): Phase 3 で実装
}

void FSurvivorsGameLogic::DropGem(int32 TypeId, FVector2D Pos)
{
	// TODO(issue): Phase 3 で実装
}

void FSurvivorsGameLogic::CheckGemCollections()
{
	// TODO(issue): Phase 3 で実装
}

void FSurvivorsGameLogic::ApplyEnemyContactDamage()
{
	// TODO(issue): Phase 3 で実装
}

void FSurvivorsGameLogic::ResolveWallCollisions()
{
	// TODO(issue): Phase 3 で実装
}

float FSurvivorsGameLogic::CastRayToObstacles(FVector2D Origin, FVector2D Dir) const
{
	// TODO(issue): Phase 3 で実装
	return CurrentConfig.FieldHalfSize * 2.f;
}

bool FSurvivorsGameLogic::ReflectOffWall(FVector2D& InOutPos, FVector2D& InOutVel, float Radius) const
{
	// TODO(issue): Phase 3 で実装
	return false;
}

void FSurvivorsGameLogic::FinalizePendingEnemies()
{
	// TODO(issue): Phase 3 で実装
}

void FSurvivorsGameLogic::FinalizePickupRemovals()
{
	// TODO(issue): Phase 3 で実装
}

void FSurvivorsGameLogic::InitDefaultEnemyTable()
{
	// TODO(issue): Phase 3 で SurvivorsSpawnComponent のロジックを移植する
}

void FSurvivorsGameLogic::InitDefaultSpawnWaves()
{
	// TODO(issue): Phase 3 で SurvivorsSpawnComponent のロジックを移植する
}

const FSpawnWave* FSurvivorsGameLogic::GetCurrentWave() const
{
	return nullptr;
}

int32 FSurvivorsGameLogic::GetCurrentWaveIndex() const
{
	return INDEX_NONE;
}

bool FSurvivorsGameLogic::BuildSpawnWeights(
	const FSpawnWave& Wave, TArray<FEnemySpawnWeight>& OutWeights, bool& bOutUsedCurriculumPool) const
{
	bOutUsedCurriculumPool = false;
	return false;
}

int32 FSurvivorsGameLogic::SelectTypeByWeight(const TArray<FEnemySpawnWeight>& Weights)
{
	return 0;
}

float FSurvivorsGameLogic::GetEnemySpeed(int32 TypeId) const
{
	if (CurrentConfig.EnemyTypeTable.IsValidIndex(TypeId))
		return CurrentConfig.EnemyTypeTable[TypeId].Speed * CurrentConfig.EnemySpeedMult;
	return 50.f * CurrentConfig.EnemySpeedMult;
}

float FSurvivorsGameLogic::GetEnemyTypeMaxHP(int32 TypeId) const
{
	if (CurrentConfig.EnemyTypeTable.IsValidIndex(TypeId))
		return CurrentConfig.EnemyTypeTable[TypeId].BaseHP * CurrentConfig.EnemyHPScale;
	return 1.f;
}

float FSurvivorsGameLogic::XPRequiredForLevel(int32 Level) const
{
	// TODO(issue): Phase 3 で SurvivorsPlayerComponent のロジックを移植する
	// 暫定: ASurvivorsGame::XPRequiredForLevel と同じ計算式をコピー
	if (Level <= 1) return 0.f;

	static const float BaseXP[] = {
		0.f, 5.f, 10.f, 20.f, 35.f, 50.f, 70.f, 90.f, 120.f, 150.f,
		185.f, 220.f, 260.f, 300.f, 345.f, 390.f, 440.f, 490.f, 545.f, 600.f,
	};
	static const int32 BaseCount = 20;

	if (Level - 1 < BaseCount)
		return BaseXP[Level - 1];

	// Level 21+: 前レベル要求値 + 加算テーブル or 固定加算
	static const float WallXP[] = { 600.f, 2400.f };
	static const int32 WallLevels[] = { 21, 41 };

	float Req = BaseXP[BaseCount - 1];
	for (int32 L = BaseCount + 1; L <= Level; ++L)
	{
		float Add = 35.f;
		for (int32 w = 0; w < 2; ++w)
		{
			if (L == WallLevels[w]) { Add = WallXP[w]; break; }
		}
		Req += Add;
	}
	return Req;
}

float FSurvivorsGameLogic::CumulativeXPForLevel(int32 Level) const
{
	float Total = 0.f;
	for (int32 L = 2; L <= Level; ++L)
		Total += XPRequiredForLevel(L);
	return Total;
}

void FSurvivorsGameLogic::ProcessXPGain(float Amount)
{
	// TODO(issue): Phase 3 で実装
}

void FSurvivorsGameLogic::OnLevelUp(int32 NextLevel)
{
	// TODO(issue): Phase 3 で実装
}

void FSurvivorsGameLogic::RecalcPassiveEffects()
{
	// TODO(issue): Phase 3 で実装
}

void FSurvivorsGameLogic::ApplyAction(int32 ActionIdx, float Dt)
{
	// TODO(issue): Phase 3 で実装
}

void FSurvivorsGameLogic::StepSpawn(float Dt)
{
	// TODO(issue): Phase 3 で実装
}

void FSurvivorsGameLogic::CheckFloorPickups()
{
	// TODO(issue): Phase 3 で実装
}

void FSurvivorsGameLogic::TickWeapons(float Dt)
{
	// TODO(issue): Phase 3 で実装
}

void FSurvivorsGameLogic::BuildEnemyGrid()
{
	// TODO(issue): Phase 3 で実装
}

void FSurvivorsGameLogic::BuildPickupGrid()
{
	// TODO(issue): Phase 3 で実装
}

void FSurvivorsGameLogic::QueryEnemyContacts(
	FVector2D Pos, float Radius, TArray<const FSurvivorsTargetProxy*>& Out) const
{
	// TODO(issue): Phase 3 で実装
}

void FSurvivorsGameLogic::QueryPickupContacts(
	FVector2D Pos, float Radius, TArray<const FSurvivorsTargetProxy*>& Out) const
{
	// TODO(issue): Phase 3 で実装
}
