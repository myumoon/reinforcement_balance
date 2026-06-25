#include "Survivors/SurvivorsGameLogic.h"
#include "Survivors/SurvivorsWikiSpec.h"
#include "Survivors/SurvivorsGameConstants.h"
#include "Survivors/Weapons/SurvivorsWeaponLogic.h"
#include "Survivors/Weapons/Projectile/SurvivorsWeaponGarlicLogic.h"
#include "Survivors/Weapons/Projectile/SurvivorsWeaponWhipLogic.h"
#include "Survivors/Weapons/Projectile/SurvivorsWeaponMagicWandLogic.h"
#include "Survivors/Weapons/Projectile/SurvivorsWeaponKnifeLogic.h"
#include "Survivors/Weapons/Projectile/SurvivorsWeaponAxeLogic.h"
#include "Survivors/Weapons/Projectile/SurvivorsWeaponCrossLogic.h"
#include "Survivors/Weapons/Projectile/SurvivorsWeaponKingBibleLogic.h"
#include "Survivors/Weapons/Projectile/SurvivorsWeaponFireWandLogic.h"
#include "Survivors/Weapons/Projectile/SurvivorsWeaponSantaWaterLogic.h"
#include "Survivors/Weapons/Projectile/SurvivorsWeaponRunetracerLogic.h"
#include "Survivors/Weapons/Projectile/SurvivorsWeaponLightningRingLogic.h"
#include "Survivors/Weapons/Projectile/SurvivorsWeaponPentagramLogic.h"
#include "Survivors/Weapons/Projectile/SurvivorsWeaponPeachoneLogic.h"
#include "Survivors/Weapons/Projectile/SurvivorsWeaponEbonyWingsLogic.h"
#include "Survivors/Weapons/Projectile/SurvivorsWeaponVandalierLogic.h"
#include "Survivors/Weapons/Projectile/SurvivorsWeaponLaurelLogic.h"
#include "Misc/SecureHash.h"
#include <algorithm>

// ---- ローカルヘルパー --------------------------------------------------------
namespace
{
	static int32 WSSampleIndex(const TArray<float>& Weights, FRandomStream& RS)
	{
		float Total = 0.f; for (float W : Weights) Total += W;
		if (Total <= 0.f) return 0;
		float R = RS.FRandRange(0.f, Total), Cum = 0.f;
		for (int32 i = 0; i < Weights.Num(); ++i) { Cum += Weights[i]; if (R <= Cum) return i; }
		return Weights.Num() - 1;
	}
	static int32 WeaponWeightSample(const TMap<int32, float>& WMap, FRandomStream& RS)
	{
		float Total = 0.f; for (const auto& P : WMap) Total += P.Value;
		if (Total <= 0.f) return -1;
		float R = RS.FRandRange(0.f, Total), Cum = 0.f;
		for (const auto& P : WMap) { Cum += P.Value; if (R <= Cum) return P.Key; }
		return WMap.CreateConstIterator()->Key;
	}
	static void BuildDirDensity(const TArray<FVector2D>& Pos, const FVector2D& PPos,
		int32 DC, float NDistMax, float NNear, float NMid, float NNorm, float MNorm, TArray<float>& Out)
	{
		TArray<float> Near, ND, MD;
		Near.Init(1.f, DC); ND.Init(0.f, DC); MD.Init(0.f, DC);
		for (const FVector2D& P : Pos)
		{
			const FVector2D Rel = P - PPos;
			const float D = Rel.Size();
			if (D <= KINDA_SMALL_NUMBER) continue;
			const float A01 = (FMath::Atan2(Rel.Y, Rel.X) + PI) / (2.f * PI);
			const int32 Dir = FMath::Clamp(FMath::FloorToInt(A01 * DC), 0, DC - 1);
			Near[Dir] = FMath::Min(Near[Dir], FMath::Clamp(D / NDistMax, 0.f, 1.f));
			if (D <= NNear) ND[Dir] += FMath::Clamp(1.f - D / NNear, 0.f, 1.f);
			else if (D <= NMid) { const float T = (D - NNear) / (NMid - NNear); MD[Dir] += FMath::Clamp(1.f - T, 0.f, 1.f); }
		}
		for (int32 d = 0; d < DC; ++d) Out.Add(Near[d]);
		for (int32 d = 0; d < DC; ++d) Out.Add(FMath::Clamp(ND[d] / NNorm, 0.f, 1.f));
		for (int32 d = 0; d < DC; ++d) Out.Add(FMath::Clamp(MD[d] / MNorm, 0.f, 1.f));
	}
} // namespace

// ============================================================================
// 初期化
// ============================================================================

FSurvivorsGameLogic::FSurvivorsGameLogic() = default;
FSurvivorsGameLogic::~FSurvivorsGameLogic() = default;

void FSurvivorsGameLogic::Initialize(const FSurvivorsGameLogicConfig& Config)
{
	CurrentConfig = Config;
	CachedObsDim  = -1;
	if (CurrentConfig.EnemyTypeTable.IsEmpty()) InitDefaultEnemyTable();
	if (CurrentConfig.SpawnWaves.IsEmpty())     InitDefaultSpawnWaves();
}

void FSurvivorsGameLogic::ApplyConfig(const FSurvivorsGameLogicConfig& Config)
{
	TArray<FBox2D>           Walls = MoveTemp(CurrentConfig.WallBounds);
	TArray<FSpawnWave>       Waves = MoveTemp(CurrentConfig.SpawnWaves);
	TArray<FEnemyTypeParams> Table = MoveTemp(CurrentConfig.EnemyTypeTable);
	CurrentConfig = Config;
	if (CurrentConfig.WallBounds.IsEmpty())     CurrentConfig.WallBounds     = MoveTemp(Walls);
	if (CurrentConfig.SpawnWaves.IsEmpty())     CurrentConfig.SpawnWaves     = MoveTemp(Waves);
	if (CurrentConfig.EnemyTypeTable.IsEmpty()) CurrentConfig.EnemyTypeTable = MoveTemp(Table);
	if (CurrentConfig.EnemyTypeTable.IsEmpty()) InitDefaultEnemyTable();
	if (CurrentConfig.SpawnWaves.IsEmpty())     InitDefaultSpawnWaves();
	CachedObsDim = -1;
}

// ============================================================================
// Reset
// ============================================================================

void FSurvivorsGameLogic::Reset(TOptional<int32> Seed)
{
	if (Seed.IsSet()) RandStream.Initialize(Seed.GetValue());
	else              RandStream.GenerateNewSeed();

	for (int32 i = 0; i < MaxWeaponSlots; ++i)
	{
		WeaponSlots[i].Type    = EWeaponType::None;
		WeaponSlots[i].Level   = FWeaponLevel(0);
		WeaponSlots[i].Cooldown = FCooldownSeconds(0.f);
	}
	for (int32 i = 0; i < MaxPassiveSlots; ++i) { PassiveSlots[i].Type = EPassiveItemType::None; PassiveSlots[i].Level = 0; }
	CachedPassiveEffects = FPassiveEffects();
	CurrentConfig.MaxPlayerHP     = BaseMaxPlayerHPConst;
	CurrentConfig.GemPickupRadius = BaseGemPickupRadiusConst;
	PlayerShieldTimer = 0.f; bShieldActive = false;
	MaxRevivalCount   = 0;   UsedRevivalCount = 0;
	NextEnemyId = 0; NextGemId = 0;
	FloorPickups.Empty(); SpecialPickups.Empty(); Destructibles.Empty();
	PhysicsAccumTime = 0.f;

	// 武器・プロジェクタイルリセット
	Weapons.Reset(); Weapons.SetNum(MaxWeaponSlots);
	Projectiles.Empty(); Projectiles.Reserve(64);
	GroundZones.Empty(); GroundZones.Reserve(16);

	// コンポーネント相当リセット
	PlayerPos = FVector2D::ZeroVector; PlayerVel = FVector2D::ZeroVector;
	PlayerHP  = CurrentConfig.MaxPlayerHP; PlayerXP = 0.f; PlayerLevel = 1;
	Gems.Empty(); Enemies.Empty();
	SpawnAccumulator = 0.f; bBossSpawned = false; LastSpawnDebug = FSurvivorsSpawnDebug();

	// 開始武器選択
	{
		static const TArray<EWeaponType> AllBaseWeapons = {
			EWeaponType::Garlic,   EWeaponType::Whip,          EWeaponType::MagicWand,
			EWeaponType::Knife,    EWeaponType::Axe,           EWeaponType::Cross,
			EWeaponType::KingBible,EWeaponType::FireWand,      EWeaponType::SantaWater,
			EWeaponType::Runetracer,EWeaponType::LightningRing,EWeaponType::Peachone,
			EWeaponType::EbonyWings,
		};
		EWeaponType StartWeapon = EWeaponType::Garlic;
		const bool bUseSubset = (CurrentConfig.WeaponPoolMode == TEXT("fixed_subset") || CurrentConfig.WeaponPoolMode == TEXT("weighted"))
			&& CurrentConfig.AllowedWeaponTypes.Num() > 0;
		const bool bWeighted  = CurrentConfig.WeaponPoolMode == TEXT("weighted") && !CurrentConfig.WeaponWeights.IsEmpty();

		if (CurrentConfig.StartingWeaponMode.Equals(TEXT("random"), ESearchCase::IgnoreCase) ||
		    CurrentConfig.StartingWeaponMode.Equals(TEXT("pool_random"), ESearchCase::IgnoreCase))
		{
			if (CurrentConfig.WeaponPoolMode == TEXT("garlic_only")) StartWeapon = EWeaponType::Garlic;
			else if (bWeighted) { const int32 Id = WeaponWeightSample(CurrentConfig.WeaponWeights, RandStream); StartWeapon = Id > 0 ? static_cast<EWeaponType>(Id) : EWeaponType::Garlic; }
			else if (bUseSubset) { StartWeapon = static_cast<EWeaponType>(CurrentConfig.AllowedWeaponTypes[RandStream.RandRange(0, CurrentConfig.AllowedWeaponTypes.Num()-1)]); }
			else { StartWeapon = AllBaseWeapons[RandStream.RandRange(0, AllBaseWeapons.Num()-1)]; }
		}
		else if (CurrentConfig.StartingWeaponMode.Equals(TEXT("garlic"), ESearchCase::IgnoreCase) || CurrentConfig.WeaponPoolMode == TEXT("garlic_only"))
			StartWeapon = EWeaponType::Garlic;
		else if (CurrentConfig.StartingWeaponMode.Equals(TEXT("whip"), ESearchCase::IgnoreCase))
			StartWeapon = EWeaponType::Whip;
		else if (bWeighted) { const int32 Id = WeaponWeightSample(CurrentConfig.WeaponWeights, RandStream); StartWeapon = Id > 0 ? static_cast<EWeaponType>(Id) : EWeaponType::Garlic; }
		else if (bUseSubset) { StartWeapon = static_cast<EWeaponType>(CurrentConfig.AllowedWeaponTypes[0]); }

		WeaponSlots[0].Type  = StartWeapon;
		WeaponSlots[0].Level = FWeaponLevel(1);
		EquipWeapon(0, StartWeapon, 1);
	}

	ElapsedTime = 0.f; GlobalFreezeUntilTime = -1.f;
	LastReward = 0.f; EpisodeBaseReward = 0.f; EpisodeStepCount = 0;
	bDone = false; bTruncated = false;

	// RSI オーバーライド
	if (CurrentConfig.bHasInitialOverride)
	{
		ElapsedTime = FMath::Clamp(CurrentConfig.InitialElapsedTime, 0.f, 1800.f);
		for (int32 i = 0; i < CurrentConfig.InitialWeaponSlots.Num() && i < MaxWeaponSlots; ++i)
		{
			const int32 WId = CurrentConfig.InitialWeaponSlots[i].WeaponId;
			const int32 WLv = FMath::Clamp(CurrentConfig.InitialWeaponSlots[i].Level, 1, 8);
			EquipWeapon(i, static_cast<EWeaponType>(WId), WLv);
			WeaponSlots[i].Type  = static_cast<EWeaponType>(WId);
			WeaponSlots[i].Level = FWeaponLevel(WLv);
		}
		for (int32 i = 0; i < CurrentConfig.InitialPassiveSlots.Num() && i < MaxPassiveSlots; ++i)
		{
			PassiveSlots[i].Type  = static_cast<EPassiveItemType>(CurrentConfig.InitialPassiveSlots[i].PassiveId);
			PassiveSlots[i].Level = FMath::Clamp(CurrentConfig.InitialPassiveSlots[i].Level, 1, 9);
		}
		if (!CurrentConfig.InitialPassiveSlots.IsEmpty()) RecalcPassiveEffects();
		CurrentConfig.bHasInitialOverride = false;
		CurrentConfig.InitialWeaponSlots.Empty();
		CurrentConfig.InitialPassiveSlots.Empty();
		CurrentConfig.InitialElapsedTime  = 0.f;
	}
}

// ============================================================================
// PhysicsStep
// ============================================================================

void FSurvivorsGameLogic::PhysicsStep(int32 ActionIdx)
{
	if (bDone || bTruncated) return;
	LastReward = 0.f;

	ApplyAction(ActionIdx, PhysicsDt);
	ResolveWallCollisions();

	if (CachedPassiveEffects.RegenPerSec > 0.f)
		PlayerHP = FMath::Min(PlayerHP + CachedPassiveEffects.RegenPerSec * PhysicsDt, CurrentConfig.MaxPlayerHP);

	ElapsedTime += PhysicsDt;
	UpdateEnemies();
	RecycleDistantEnemies();
	TickWeapons(PhysicsDt);

	BuildEnemyGrid();
	{ FSurvivorsHitFrame HF; ComputeAllWeaponHits(HF); ApplyWeaponHits(HF); }
	FinalizePendingEnemies();

	StepSpawn(PhysicsDt);

	BuildEnemyGrid();
	{ FSurvivorsHitFrame HF; ComputeContactHits(HF); ApplyContactHits(HF); }

	BuildPickupGrid();
	{ FSurvivorsHitFrame HF; ComputePickupHits(HF); ApplyPickupHits(HF); }
	FinalizePickupRemovals();

	CheckFloorPickups();
	CheckSpecialPickups();

	if (PlayerHP <= 0.f && UsedRevivalCount < MaxRevivalCount)
	{ ++UsedRevivalCount; PlayerHP = CurrentConfig.MaxPlayerHP * 0.5f; }

	if (PlayerHP <= 0.f)
	{ bDone = true; EpisodeBaseReward += LastReward; EpisodeStepCount++; return; }

	LastReward += CurrentConfig.AliveReward;
	EpisodeBaseReward += LastReward;
	EpisodeStepCount++;

	if (CurrentConfig.MaxEpisodeTime > 0.f && ElapsedTime >= CurrentConfig.MaxEpisodeTime)
	{ bTruncated = true; LastSpawnDebug.bTruncated = true; }
}

// ============================================================================
// ExecStep / ExecReset
// ============================================================================

FSurvivorsStepResult FSurvivorsGameLogic::ExecStep(const TArray<float>& Action, int32 Steps)
{
	FSurvivorsStepResult Result;
	const int32 ActionIdx = Action.Num() > 0 ? FMath::Clamp(static_cast<int32>(Action[0]), 0, 8) : 8;
	float AccReward = 0.f;
	for (int32 i = 0; i < Steps; ++i)
	{
		PhysicsStep(ActionIdx);
		AccReward += LastReward;
		if (bDone)     { Result.bDone      = true; break; }
		if (bTruncated){ Result.bTruncated = true; break; }
	}
	Result.Obs            = GetObservation();
	Result.Reward         = AccReward;
	Result.SpawnDebugJson = GetSpawnDebugJson();
	return Result;
}

FSurvivorsResetResult FSurvivorsGameLogic::ExecReset(TOptional<int32> Seed)
{
	FSurvivorsResetResult Result;
	Reset(Seed);
	Result.Obs           = GetObservation();
	Result.ObsSchemaHash = GetObsSchemaHash();
	return Result;
}

// ============================================================================
// Observation
// ============================================================================

TArray<FSurvivorsObsSegment> FSurvivorsGameLogic::GetObsSchema() const
{
	using namespace SurvivorsGameConstants;
	return {
		{ TEXT("player_pos"),                2 }, { TEXT("player_vel"),                2 },
		{ TEXT("wall_rays"),                 8 }, { TEXT("player_hp"),                 1 },
		{ TEXT("shield_active"),             1 }, { TEXT("shield_timer_norm"),         1 },
		{ TEXT("revival_remaining_norm"),    1 }, { TEXT("armor_flat_norm"),           1 },
		{ TEXT("regen_per_sec_norm"),        1 }, { TEXT("passive_effect_summary"),    5 },
		{ TEXT("weapon_slots"),              MaxWeaponSlots * 3 },
		{ TEXT("passive_slots"),             MaxPassiveSlots * 2 },
		{ TEXT("enemy_count"),               1 }, { TEXT("elapsed_time"),              1 },
		{ TEXT("xp_progress"),              1 }, { TEXT("player_level"),              1 },
		{ TEXT("stage_id_norm"),             1 },
		{ TEXT("red_gem_rel_pos"),           MaxRedGemObs * 2 },
		{ TEXT("green_gem_rel_pos"),         MaxGreenGemObs * 2 },
		{ TEXT("blue_gem_rel_pos"),          MaxBlueGemObs * 2 },
		{ TEXT("gem_pickup_radius"),         1 },
		{ TEXT("enemy_rel_pos"),             MaxEnemyObs * 2 }, { TEXT("enemy_vel"),   MaxEnemyObs * 2 },
		{ TEXT("enemy_type"),                MaxEnemyObs }, { TEXT("enemy_hp"),         MaxEnemyObs },
		{ TEXT("enemy_frozen"),              MaxEnemyObs },
		{ TEXT("enemy_nearest_dist_16dir"),  EnemyDensityDirCount },
		{ TEXT("enemy_density_near_16dir"),  EnemyDensityDirCount },
		{ TEXT("enemy_density_mid_16dir"),   EnemyDensityDirCount },
		{ TEXT("gem_density_all_16dir"),     GemDensityDirCount * 3 },
		{ TEXT("red_green_gem_density_16dir"),GemDensityDirCount * 3 },
		{ TEXT("projectiles"),               MaxProjectileObs * ProjectileObsStride },
		{ TEXT("floor_pickups"),             MaxFloorPickupObs * 3 },
		{ TEXT("special_pickups"),           MaxSpecialPickupObs * 3 },
		{ TEXT("destructibles"),             MaxDestructibleObs * 2 },
		{ TEXT("weapon_attack_range_norm"),  MaxWeaponSlots },
		{ TEXT("weapon_is_directional"),     MaxWeaponSlots },
		{ TEXT("weapon_category_onehot"),    MaxWeaponSlots * 7 },
	};
}

FString FSurvivorsGameLogic::GetObsSchemaHash() const
{
	using namespace SurvivorsGameConstants;
	FString S = FString::Printf(
		TEXT("SurvivorsGame_v795_projectiles_stride9,MaxEnemyObs=%d,MaxWeaponSlots=%d,MaxPassiveSlots=%d"
		     ",MaxProjectileObs=%d,ProjectileObsStride=%d,MaxRedGemObs=%d,MaxGreenGemObs=%d,MaxBlueGemObs=%d"
		     ",MaxFloorPickupObs=%d,MaxSpecialPickupObs=%d,MaxDestructibleObs=%d"
		     ",MaxWeaponTypeCountReserved=%d,MaxPassiveTypeCountReserved=%d"
		     ",EnemyDensityDirCount=%d,GemDensityDirCount=%d"
		     ",EnemyNearestDistanceMax=%.0f,GemNearestDistanceMax=%.0f"),
		MaxEnemyObs,MaxWeaponSlots,MaxPassiveSlots,
		MaxProjectileObs,ProjectileObsStride,MaxRedGemObs,MaxGreenGemObs,MaxBlueGemObs,
		MaxFloorPickupObs,MaxSpecialPickupObs,MaxDestructibleObs,
		MaxWeaponTypeCountReserved,MaxPassiveTypeCountReserved,
		EnemyDensityDirCount,GemDensityDirCount,
		EnemyNearestDistanceMax,GemNearestDistanceMax);
	return FMD5::HashAnsiString(*S);
}

int32 FSurvivorsGameLogic::GetObsDim() const
{
	if (CachedObsDim >= 0) return CachedObsDim;
	int32 T = 0; for (const auto& Seg : GetObsSchema()) T += Seg.Dim;
	CachedObsDim = T; return T;
}

float FSurvivorsGameLogic::GetReward()     const { return LastReward; }
bool  FSurvivorsGameLogic::IsDone()        const { return bDone; }
bool  FSurvivorsGameLogic::IsTruncated()   const { return bTruncated; }

FString FSurvivorsGameLogic::GetSpawnDebugJson() const
{
	return FString::Printf(
		TEXT("{\"elapsed_time\":%.3f,\"max_episode_time\":%.3f,\"enemy_count\":%d,\"current_wave_index\":%d,"
		     "\"min_active_enemies\":%d,\"max_active_enemies\":%d,\"effective_min_enemies\":%d,\"effective_max_enemies\":%d,"
		     "\"max_enemy_type_id\":%d,\"allowed_spawn_type_count\":%d,\"spawn_accumulator\":%.3f,"
		     "\"has_current_wave\":%s,\"used_curriculum_enemy_pool\":%s,\"spawn_blocked\":%s,\"truncated\":%s}"),
		LastSpawnDebug.ElapsedTime,LastSpawnDebug.MaxEpisodeTime,LastSpawnDebug.EnemyCount,
		LastSpawnDebug.CurrentWaveIndex,LastSpawnDebug.MinActiveEnemies,LastSpawnDebug.MaxActiveEnemies,
		LastSpawnDebug.EffectiveMinEnemies,LastSpawnDebug.EffectiveMaxEnemies,LastSpawnDebug.MaxEnemyTypeId,
		LastSpawnDebug.AllowedSpawnTypeCount,LastSpawnDebug.SpawnAccumulator,
		LastSpawnDebug.bHasCurrentWave?TEXT("true"):TEXT("false"),
		LastSpawnDebug.bUsedCurriculumEnemyPool?TEXT("true"):TEXT("false"),
		LastSpawnDebug.bSpawnBlocked?TEXT("true"):TEXT("false"),
		LastSpawnDebug.bTruncated?TEXT("true"):TEXT("false"));
}

TArray<float> FSurvivorsGameLogic::GetObservation() const
{
	TArray<float> Obs;
	Obs.Reserve(GetObsDim());
	using namespace SurvivorsGameConstants;
	const float HN = CurrentConfig.FieldHalfSize;
	const float DN = HN * 2.f;
	const float MaxRayDist = DN;

	// player_pos/vel
	Obs.Add(PlayerPos.X / HN); Obs.Add(PlayerPos.Y / HN);
	const float MS = CurrentConfig.MoveSpeed > 0.f ? CurrentConfig.MoveSpeed : 1.f;
	Obs.Add(PlayerVel.X / MS); Obs.Add(PlayerVel.Y / MS);

	// wall_rays (8)
	for (int32 r = 0; r < 8; ++r)
		Obs.Add(FMath::Clamp(CastRayToObstacles(PlayerPos, RayDirs[r]) / MaxRayDist, 0.f, 1.f));

	// player_hp, shield, revival, armor, regen
	Obs.Add(FMath::Clamp(PlayerHP / CurrentConfig.MaxPlayerHP, 0.f, 1.f));
	Obs.Add(bShieldActive ? 1.f : 0.f);
	Obs.Add(FMath::Clamp(PlayerShieldTimer / MaxShieldDuration, 0.f, 1.f));
	{ const float MR = MaxRevivalCount > 0 ? (float)MaxRevivalCount : 1.f;
	  Obs.Add(FMath::Clamp((float)(MaxRevivalCount - UsedRevivalCount) / MR, 0.f, 1.f)); }
	Obs.Add(FMath::Clamp(CachedPassiveEffects.ArmorFlat / MaxArmorFlat, 0.f, 1.f));
	Obs.Add(FMath::Clamp(CachedPassiveEffects.RegenPerSec / MaxRegenPerSec, 0.f, 1.f));

	// passive_effect_summary (5)
	Obs.Add(FMath::Clamp(CachedPassiveEffects.DamageMult    - 1.f, 0.f, 1.f));
	Obs.Add(FMath::Clamp(1.f - CachedPassiveEffects.CooldownMult,  0.f, 1.f));
	Obs.Add(FMath::Clamp(CachedPassiveEffects.AreaMult      - 1.f, 0.f, 1.f));
	Obs.Add(FMath::Clamp(CachedPassiveEffects.MoveSpeedMult - 1.f, 0.f, 1.f));
	Obs.Add(FMath::Clamp(CachedPassiveEffects.PickupRadiusMult - 1.f, 0.f, 1.f));

	// weapon_slots (MaxWeaponSlots*3)
	for (int32 s = 0; s < MaxWeaponSlots; ++s)
	{
		const FWeaponSlot& Slot = WeaponSlots[s];
		Obs.Add((float)(uint8)Slot.Type / (float)MaxWeaponTypeCountReserved);
		const int32 WML = SurvivorsGameConstants::GetWeaponMaxLevel(Slot.Type);
		Obs.Add(WML > 0 ? (float)Slot.Level.Value / (float)WML : 0.f);
		const FSurvivorsWeaponLogic* WI = Weapons.IsValidIndex(s) ? Weapons[s].Get() : nullptr;
		const float CooldownDen = WI ? FMath::Max(WI->GetCooldownObsDenominator(), KINDA_SMALL_NUMBER) : 1.0f;
		Obs.Add(WI && Slot.Type != EWeaponType::None ? FMath::Clamp(WI->GetCooldownRemaining().Value / CooldownDen, 0.f, 1.f) : 0.f);
	}

	// passive_slots (MaxPassiveSlots*2)
	for (int32 s = 0; s < MaxPassiveSlots; ++s)
	{
		const FPassiveSlot& PS = PassiveSlots[s];
		Obs.Add((float)(uint8)PS.Type / (float)MaxPassiveTypeCountReserved);
		float LN = 0.f;
		if (PS.Type != EPassiveItemType::None)
		{
			const int32 TI = (int32)(uint8)PS.Type;
			if (TI >= 0 && TI < 18 && PassiveMaxLevel[TI] > 0)
				LN = FMath::Clamp((float)PS.Level / (float)PassiveMaxLevel[TI], 0.f, 1.f);
		}
		Obs.Add(LN);
	}

	// enemy_count, elapsed, xp_progress, level, stage
	Obs.Add((float)Enemies.Num() / (float)MaxEnemyObs);
	Obs.Add(FMath::Clamp(ElapsedTime / MaxGameTime, 0.f, 1.f));
	{ const float CC = CumulativeXPForLevel(PlayerLevel), NC = CumulativeXPForLevel(PlayerLevel+1), R = NC - CC;
	  Obs.Add(R > 0.f ? FMath::Clamp((PlayerXP - CC) / R, 0.f, 1.f) : 0.f); }
	Obs.Add(FMath::Clamp((float)PlayerLevel / (float)MaxPlayerLevel, 0.f, 1.f));
	Obs.Add(0.f);

	// gem分類
	TArray<FVector2D> RedPos, GreenPos, BluePos;
	for (const FGemState& G : Gems)
	{ if (G.Type == EGemType::Red) RedPos.Add(G.Pos); else if (G.Type == EGemType::Green) GreenPos.Add(G.Pos); else BluePos.Add(G.Pos); }

	auto AddGemObs = [&](const TArray<FVector2D>& Positions, int32 MaxN)
	{
		TArray<int32> Idx; Idx.Reserve(Positions.Num());
		for (int32 i = 0; i < Positions.Num(); ++i) Idx.Add(i);
		const int32 TN = FMath::Min(MaxN, Idx.Num());
		std::partial_sort(Idx.GetData(), Idx.GetData()+TN, Idx.GetData()+Idx.Num(),
			[&](int32 A, int32 B){ return FVector2D::DistSquared(Positions[A],PlayerPos) < FVector2D::DistSquared(Positions[B],PlayerPos); });
		for (int32 s = 0; s < MaxN; ++s)
		{ if (s < Idx.Num()) { Obs.Add((Positions[Idx[s]].X-PlayerPos.X)/DN); Obs.Add((Positions[Idx[s]].Y-PlayerPos.Y)/DN); }
		  else { Obs.Add(0.f); Obs.Add(0.f); } }
	};
	AddGemObs(RedPos, MaxRedGemObs);
	AddGemObs(GreenPos, MaxGreenGemObs);
	AddGemObs(BluePos, MaxBlueGemObs);
	Obs.Add(FMath::Clamp(CurrentConfig.GemPickupRadius / 150.f, 0.f, 1.f));

	// 敵ソート
	TArray<int32> EIdx; EIdx.Reserve(Enemies.Num());
	for (int32 i = 0; i < Enemies.Num(); ++i) EIdx.Add(i);
	{ const int32 TN = FMath::Min(MaxEnemyObs, EIdx.Num());
	  std::partial_sort(EIdx.GetData(), EIdx.GetData()+TN, EIdx.GetData()+EIdx.Num(),
		[this](int32 A, int32 B){ return FVector2D::DistSquared(Enemies[A].Pos,PlayerPos) < FVector2D::DistSquared(Enemies[B].Pos,PlayerPos); }); }

	for (int32 S = 0; S < MaxEnemyObs; ++S)
	{ if (S<EIdx.Num()){Obs.Add((Enemies[EIdx[S]].Pos.X-PlayerPos.X)/DN);Obs.Add((Enemies[EIdx[S]].Pos.Y-PlayerPos.Y)/DN);}else{Obs.Add(0.f);Obs.Add(0.f);} }
	for (int32 S = 0; S < MaxEnemyObs; ++S)
	{ if (S<EIdx.Num()){Obs.Add(Enemies[EIdx[S]].Vel.X/MS);Obs.Add(Enemies[EIdx[S]].Vel.Y/MS);}else{Obs.Add(0.f);Obs.Add(0.f);} }
	{ const float TN2 = CurrentConfig.EnemyTypeTable.Num()>1?(float)(CurrentConfig.EnemyTypeTable.Num()-1):1.f;
	  for (int32 S = 0; S < MaxEnemyObs; ++S) Obs.Add(S<EIdx.Num()?(float)Enemies[EIdx[S]].TypeId/TN2:0.f); }
	for (int32 S = 0; S < MaxEnemyObs; ++S)
	{ if (S<EIdx.Num()){const FEnemyState& E=Enemies[EIdx[S]];Obs.Add(FMath::Clamp(E.HP/E.MaxHP,0.f,1.f));}else Obs.Add(0.f); }
	for (int32 S = 0; S < MaxEnemyObs; ++S)
	Obs.Add(S<EIdx.Num()?(Enemies[EIdx[S]].bFrozen?1.f:0.f):0.f);

	// enemy density
	{ TArray<FVector2D> EP; EP.Reserve(Enemies.Num()); for (const FEnemyState& E:Enemies) EP.Add(E.Pos);
	  BuildDirDensity(EP,PlayerPos,EnemyDensityDirCount,EnemyNearestDistanceMax,EnemyDensityNearDistanceMax,EnemyDensityMidDistanceMax,EnemyDensityNearNormalizeFactor,EnemyDensityMidNormalizeFactor,Obs); }
	{ TArray<FVector2D> GP; GP.Reserve(Gems.Num()); for (const FGemState& G:Gems) GP.Add(G.Pos);
	  BuildDirDensity(GP,PlayerPos,GemDensityDirCount,GemNearestDistanceMax,GemDensityNearDistanceMax,GemDensityMidDistanceMax,GemDensityNearNormalizeFactor,GemDensityMidNormalizeFactor,Obs); }
	{ TArray<FVector2D> RGP; for (const FGemState& G:Gems) if(G.Type!=EGemType::Blue) RGP.Add(G.Pos);
	  BuildDirDensity(RGP,PlayerPos,GemDensityDirCount,GemNearestDistanceMax,GemDensityNearDistanceMax,GemDensityMidDistanceMax,GemDensityNearNormalizeFactor,GemDensityMidNormalizeFactor,Obs); }

	// projectiles (stride 9: dx,dy,radius_norm,vx_norm,vy_norm,warning,kind_norm,slot_norm,ttl_norm)
	{
		TArray<FProjectileObsState> PV = GetProjectileObsView();
		const int32 TPN = FMath::Min(MaxProjectileObs, PV.Num());
		// ソート: Kind!=None → 距離近い順 → slot 昇順
		std::partial_sort(PV.GetData(), PV.GetData()+TPN, PV.GetData()+PV.Num(),
			[&](const FProjectileObsState& A, const FProjectileObsState& B)
			{
				const bool AValid = A.Kind != EProjectileObsKind::None;
				const bool BValid = B.Kind != EProjectileObsKind::None;
				if (AValid != BValid) return AValid > BValid;
				const float DA = FVector2D::DistSquared(A.Pos, PlayerPos);
				const float DB = FVector2D::DistSquared(B.Pos, PlayerPos);
				if (DA != DB) return DA < DB;
				return A.WeaponSlotIdx < B.WeaponSlotIdx;
			});
		for (int32 p = 0; p < MaxProjectileObs; ++p)
		{
			if (p < PV.Num())
			{
				const FProjectileObsState& P = PV[p];
				Obs.Add((P.Pos.X - PlayerPos.X) / DN);
				Obs.Add((P.Pos.Y - PlayerPos.Y) / DN);
				Obs.Add(FMath::Clamp(P.Radius / MaxProjectileRadius, 0.f, 1.f));
				Obs.Add(FMath::Clamp(P.Vel.X / MS, -1.f, 1.f));
				Obs.Add(FMath::Clamp(P.Vel.Y / MS, -1.f, 1.f));
				Obs.Add(P.bIsWarning ? 1.f : 0.f);
				Obs.Add(GetProjectileObsKindNorm(P.Kind));
				Obs.Add(P.WeaponSlotIdx >= 0 ? (float)P.WeaponSlotIdx / (float)(MaxWeaponSlots - 1) : 0.f);
				Obs.Add(FMath::Clamp(P.Ttl / MaxProjectileObsTtl, 0.f, 1.f));
			}
			else { for (int32 k = 0; k < ProjectileObsStride; ++k) Obs.Add(0.f); }
		}
	}

	// floor_pickups
	{ TArray<int32> FI; for(int32 i=0;i<FloorPickups.Num();++i) if(FloorPickups[i].bActive) FI.Add(i);
	  FI.Sort([&](int32 A,int32 B){return FVector2D::DistSquared(FloorPickups[A].Pos,PlayerPos)<FVector2D::DistSquared(FloorPickups[B].Pos,PlayerPos);});
	  for(int32 s=0;s<MaxFloorPickupObs;++s)
	  { if(s<FI.Num()){const auto& F=FloorPickups[FI[s]];Obs.Add((F.Pos.X-PlayerPos.X)/DN);Obs.Add((F.Pos.Y-PlayerPos.Y)/DN);Obs.Add((float)(uint8)F.Type/2.f);}
	    else{Obs.Add(0.f);Obs.Add(0.f);Obs.Add(0.f);} } }

	// special_pickups
	{ TArray<int32> SI; for(int32 i=0;i<SpecialPickups.Num();++i) if(SpecialPickups[i].bActive) SI.Add(i);
	  SI.Sort([&](int32 A,int32 B){return FVector2D::DistSquared(SpecialPickups[A].Pos,PlayerPos)<FVector2D::DistSquared(SpecialPickups[B].Pos,PlayerPos);});
	  for(int32 s=0;s<MaxSpecialPickupObs;++s)
	  { if(s<SI.Num()){const auto& S=SpecialPickups[SI[s]];Obs.Add((S.Pos.X-PlayerPos.X)/DN);Obs.Add((S.Pos.Y-PlayerPos.Y)/DN);Obs.Add((float)(uint8)S.Type/4.f);}
	    else{Obs.Add(0.f);Obs.Add(0.f);Obs.Add(0.f);} } }

	// destructibles
	{ TArray<int32> DI; for(int32 i=0;i<Destructibles.Num();++i) if(Destructibles[i].bActive) DI.Add(i);
	  DI.Sort([&](int32 A,int32 B){return FVector2D::DistSquared(Destructibles[A].Pos,PlayerPos)<FVector2D::DistSquared(Destructibles[B].Pos,PlayerPos);});
	  for(int32 s=0;s<MaxDestructibleObs;++s)
	  { if(s<DI.Num()){const auto& D=Destructibles[DI[s]];Obs.Add((D.Pos.X-PlayerPos.X)/DN);Obs.Add((D.Pos.Y-PlayerPos.Y)/DN);}
	    else{Obs.Add(0.f);Obs.Add(0.f);} } }

	// weapon_attack_range_norm, weapon_is_directional, weapon_category_onehot
	for(int32 s=0;s<MaxWeaponSlots;++s) { const EWeaponType T=WeaponSlots[s].Type; Obs.Add(T!=EWeaponType::None?GetWeaponEffectiveRange(T):0.f); }
	for(int32 s=0;s<MaxWeaponSlots;++s)
	{ const EWeaponType T=WeaponSlots[s].Type;
	  const bool bD=(T==EWeaponType::Knife||T==EWeaponType::ThousandEdge||T==EWeaponType::Axe||T==EWeaponType::DeathSpiral
	               ||T==EWeaponType::Cross||T==EWeaponType::HeavenSword||T==EWeaponType::Peachone||T==EWeaponType::EbonyWings||T==EWeaponType::Vandalier);
	  Obs.Add(T!=EWeaponType::None?(bD?1.f:0.f):0.f); }
	for(int32 s=0;s<MaxWeaponSlots;++s)
	{ const EWeaponType T=WeaponSlots[s].Type; const int32 Cat=GetWeaponCategory(T);
	  for(int32 c=0;c<7;++c) Obs.Add(T!=EWeaponType::None&&Cat==c?1.f:0.f); }

	return Obs;
}

// ============================================================================
// アクセサ
// ============================================================================

FVector2D FSurvivorsGameLogic::GetItemPos(int32 i) const { return Gems.IsValidIndex(i)?Gems[i].Pos:FVector2D::ZeroVector; }
EGemType  FSurvivorsGameLogic::GetItemGemType(int32 i) const { return Gems.IsValidIndex(i)?Gems[i].Type:EGemType::Blue; }
bool FSurvivorsGameLogic::IsOnScreen(FVector2D WorldPos) const
{ const FVector2D R=WorldPos-PlayerPos; return FMath::Abs(R.X)<=SurvivorsGameConstants::ScreenHalfWidthU&&FMath::Abs(R.Y)<=SurvivorsGameConstants::ScreenHalfHeightU; }
TMap<int32,int32> FSurvivorsGameLogic::GetEnemyCountByType() const
{ TMap<int32,int32> R; for (const FEnemyState& E:Enemies) if(!E.bPendingRemove) R.FindOrAdd(E.TypeId)++; return R; }
float FSurvivorsGameLogic::GetXPRequiredForNextLevel() const { return XPRequiredForLevel(PlayerLevel+1); }
float FSurvivorsGameLogic::GetCurrentLevelXP() const { return PlayerXP - CumulativeXPForLevel(PlayerLevel); }
FString FSurvivorsGameLogic::GetEnemyTypeDebugLabel(int32 TypeId) const
{ if(CurrentConfig.EnemyTypeTable.IsValidIndex(TypeId)&&!CurrentConfig.EnemyTypeTable[TypeId].Name.IsEmpty()) return FString::Printf(TEXT("%s(ID:%d)"),*CurrentConfig.EnemyTypeTable[TypeId].Name,TypeId); return FString::Printf(TEXT("ID:%d"),TypeId); }
int32 FSurvivorsGameLogic::GetPassiveItemMaxLevel(EPassiveItemType Type) const
{ const int32 TI=(int32)(uint8)Type; return (TI>=0&&TI<18)?SurvivorsGameConstants::PassiveMaxLevel[TI]:0; }
float FSurvivorsGameLogic::GetAuraSize() const
{
	for (int32 s = 0; s < MaxWeaponSlots; ++s)
	{
		const FWeaponSlot& Slot = WeaponSlots[s];
		if (Slot.Type == EWeaponType::Garlic || Slot.Type == EWeaponType::SoulEater)
		{
			const int32 Lv = FMath::Clamp(Slot.Level.Value, 1, MaxWeaponLevel);
			return Slot.Type == EWeaponType::SoulEater
				? SurvivorsGameConstants::SoulEaterTable[Lv - 1].AreaRadius.Value
				: SurvivorsGameConstants::GarlicTable[Lv - 1].AreaRadius.Value;
		}
	}
	return 0.f;
}

// Projectile/GroundZone アクセサ
int32     FSurvivorsGameLogic::GetProjectileCount()               const { return Projectiles.Num(); }
FVector2D FSurvivorsGameLogic::GetProjectilePos(int32 i)          const { return Projectiles.IsValidIndex(i)?Projectiles[i].Pos:FVector2D::ZeroVector; }
FSimRadius FSurvivorsGameLogic::GetProjectileRadius(int32 i)      const { return Projectiles.IsValidIndex(i)?Projectiles[i].Radius:FSimRadius(0.f); }
EWeaponType FSurvivorsGameLogic::GetProjectileWeaponType(int32 i) const { return Projectiles.IsValidIndex(i)?Projectiles[i].WeaponType:EWeaponType::None; }
float     FSurvivorsGameLogic::GetProjectileBoxHalfWidth(int32 i) const { return Projectiles.IsValidIndex(i)?Projectiles[i].AngleRad.Value:0.f; }
int32     FSurvivorsGameLogic::GetGroundZoneCount()               const { return GroundZones.Num(); }
FVector2D FSurvivorsGameLogic::GetGroundZonePos(int32 i)          const { return GroundZones.IsValidIndex(i)?GroundZones[i].Pos:FVector2D::ZeroVector; }
float     FSurvivorsGameLogic::GetGroundZoneRadius(int32 i)       const { return GroundZones.IsValidIndex(i)?GroundZones[i].Radius:0.f; }
EWeaponType FSurvivorsGameLogic::GetGroundZoneWeaponType(int32 i) const { return GroundZones.IsValidIndex(i)?GroundZones[i].WeaponType:EWeaponType::None; }
bool      FSurvivorsGameLogic::IsGroundZoneWarning(int32 i)       const { return GroundZones.IsValidIndex(i)?GroundZones[i].bIsWarning:false; }
int32     FSurvivorsGameLogic::GetOrbitOrbCount()                 const { int32 T=0; for(const auto& W:Weapons) if(W) T+=W->GetOrbitOrbCount(); return T; }
FVector2D FSurvivorsGameLogic::GetOrbitOrbPos(int32 GI)           const
{ int32 Off=0; for(const auto& W:Weapons){if(!W)continue;const int32 C=W->GetOrbitOrbCount();if(GI<Off+C)return W->GetOrbitOrbPos(GI-Off);Off+=C;} return FVector2D::ZeroVector; }
EWeaponType FSurvivorsGameLogic::GetOrbitOrbWeaponType(int32 GI)  const
{ int32 Off=0; for(const auto& W:Weapons){if(!W)continue;const int32 C=W->GetOrbitOrbCount();if(GI<Off+C)return W->GetWeaponType();Off+=C;} return EWeaponType::None; }
float     FSurvivorsGameLogic::GetOrbitOrbVisualRadius(int32 GI)  const
{ int32 Off=0; for(const auto& W:Weapons){if(!W)continue;const int32 C=W->GetOrbitOrbCount();if(GI<Off+C)return W->GetOrbitOrbVisualRadius();Off+=C;} return 0.f; }

TArray<FProjectileObsState> FSurvivorsGameLogic::GetProjectileObsView() const
{
	using namespace SurvivorsGameConstants;

	TArray<FProjectileObsState> V;
	V.Reserve(Projectiles.Num() + GroundZones.Num() + GetOrbitOrbCount() + MaxWeaponSlots);

	// 通常 Projectiles
	for (const FProjectileState& P : Projectiles)
	{
		FProjectileObsState OS;
		OS.Pos           = P.Pos;
		OS.Vel           = P.Vel;
		OS.Radius        = P.Radius.Value;
		OS.Ttl           = P.LifeTime.Seconds;
		OS.Kind          = EProjectileObsKind::Projectile;
		OS.WeaponSlotIdx = P.WeaponSlotIdx;
		OS.bIsWarning    = P.bIsWarning;
		V.Add(OS);
	}

	// GroundZones
	for (const FGroundZoneState& Z : GroundZones)
	{
		FProjectileObsState OS;
		OS.Pos           = Z.Pos;
		OS.Vel           = FVector2D::ZeroVector;
		OS.Radius        = Z.Radius;
		OS.Ttl           = Z.LifeTime;
		OS.Kind          = EProjectileObsKind::GroundZone;
		OS.WeaponSlotIdx = Z.WeaponSlotIdx;
		OS.bIsWarning    = Z.bIsWarning;
		V.Add(OS);
	}

	// orbit orbs: Phase 1 = KingBible / UnholyVespers のみ
	// Peachone/EbonyWings/Vandalier は Phase 2 スコープ外のため除外
	int32 OrbOff = 0;
	for (int32 si = 0; si < Weapons.Num(); ++si)
	{
		if (!Weapons[si]) continue;
		const EWeaponType OrbWType = Weapons[si]->GetWeaponType();
		const int32 OrbCount = Weapons[si]->GetOrbitOrbCount();
		if (OrbWType != EWeaponType::KingBible && OrbWType != EWeaponType::UnholyVespers)
		{
			OrbOff += OrbCount;
			continue;
		}
		for (int32 oi = 0; oi < OrbCount; ++oi)
		{
			FProjectileObsState OS;
			OS.Pos           = Weapons[si]->GetOrbitOrbPos(oi);
			OS.Vel           = FVector2D::ZeroVector;
			OS.Radius        = Weapons[si]->GetOrbitOrbVisualRadius();
			OS.Ttl           = Weapons[si]->GetOrbitOrbTtl(oi);
			OS.Kind          = EProjectileObsKind::Orbit;
			OS.WeaponSlotIdx = Weapons[si]->GetOrbitOrbSlotIdx(oi);
			OS.bIsWarning    = false;
			V.Add(OS);
		}
		OrbOff += OrbCount;
	}

	// Garlic / SoulEater aura
	for (int32 s = 0; s < MaxWeaponSlots; ++s)
	{
		const FWeaponSlot& Slot = WeaponSlots[s];
		if (Slot.Type != EWeaponType::Garlic && Slot.Type != EWeaponType::SoulEater) continue;

		const int32 Lv = FMath::Clamp(Slot.Level.Value, 1, MaxWeaponLevel);
		const float BaseRadius = (Slot.Type == EWeaponType::SoulEater)
			? SurvivorsGameConstants::SoulEaterTable[Lv - 1].AreaRadius.Value
			: SurvivorsGameConstants::GarlicTable[Lv - 1].AreaRadius.Value;

		FProjectileObsState OS;
		OS.Pos           = PlayerPos;
		OS.Vel           = FVector2D::ZeroVector;
		OS.Radius        = BaseRadius * CachedPassiveEffects.AreaMult;
		OS.Ttl           = MaxProjectileObsTtl;  // 常時 active
		OS.Kind          = EProjectileObsKind::Aura;
		OS.WeaponSlotIdx = s;
		OS.bIsWarning    = false;
		V.Add(OS);
	}

	return V;
}

int32 FSurvivorsGameLogic::GetOrbitOrbSlotIdx(int32 GI) const
{
	int32 Off = 0;
	for (int32 si = 0; si < Weapons.Num(); ++si)
	{
		if (!Weapons[si]) continue;
		const int32 C = Weapons[si]->GetOrbitOrbCount();
		if (GI < Off + C) return Weapons[si]->GetOrbitOrbSlotIdx(GI - Off);
		Off += C;
	}
	return -1;
}

// ============================================================================
// Player ロジック
// ============================================================================

void FSurvivorsGameLogic::ApplyAction(int32 ActionIdx, float Dt)
{
	FVector2D MoveDir = FVector2D::ZeroVector;
	switch (ActionIdx)
	{
		case 0: MoveDir=FVector2D(0.f,1.f); break;
		case 1: MoveDir=FVector2D(1.f,1.f).GetSafeNormal(); break;
		case 2: MoveDir=FVector2D(1.f,0.f); break;
		case 3: MoveDir=FVector2D(1.f,-1.f).GetSafeNormal(); break;
		case 4: MoveDir=FVector2D(0.f,-1.f); break;
		case 5: MoveDir=FVector2D(-1.f,-1.f).GetSafeNormal(); break;
		case 6: MoveDir=FVector2D(-1.f,0.f); break;
		case 7: MoveDir=FVector2D(-1.f,1.f).GetSafeNormal(); break;
		default: break;
	}
	const float EffMS = CurrentConfig.MoveSpeed * CachedPassiveEffects.MoveSpeedMult;
	PlayerVel = MoveDir.GetSafeNormal() * EffMS;
	PlayerPos += PlayerVel * Dt;
}

float FSurvivorsGameLogic::XPRequiredForLevel(int32 Level) const { return SurvivorsWikiSpec::XPRequiredForLevel(Level); }
float FSurvivorsGameLogic::CumulativeXPForLevel(int32 Level) const { return SurvivorsWikiSpec::CumulativeXPForLevel(Level); }

void FSurvivorsGameLogic::ProcessXPGain(float Amount)
{
	PlayerXP += SurvivorsWikiSpec::EffectiveXPGain(Amount, CachedPassiveEffects.GrowthMult, PlayerLevel);
	while (true)
	{ const float NT = CumulativeXPForLevel(PlayerLevel+1); if (PlayerXP < NT) break; PlayerLevel++; OnLevelUp(PlayerLevel); }
}

void FSurvivorsGameLogic::OnLevelUp(int32 NextLevel)
{
	if (CurrentConfig.WeaponPoolMode.Equals(TEXT("garlic_only"), ESearchCase::IgnoreCase))
	{
		if (WeaponSlots[0].Type != EWeaponType::None)
		{
			const int32 NL = FMath::Min(WeaponSlots[0].Level.Value+1, SurvivorsGameConstants::GetWeaponMaxLevel(WeaponSlots[0].Type));
			WeaponSlots[0].Level = FWeaponLevel(NL);
			if (Weapons.IsValidIndex(0) && Weapons[0]) Weapons[0]->SetLevel(FWeaponLevel(NL));
		}
		return;
	}
	TArray<FLevelUpChoice> Choices = BuildLevelUpChoices();
	if (Choices.IsEmpty()) return;
	int32 ChoiceIdx;
	if (CurrentConfig.WeaponPoolMode.Equals(TEXT("weighted"), ESearchCase::IgnoreCase) && !CurrentConfig.WeaponWeights.IsEmpty())
	{
		TArray<float> CW; CW.Reserve(Choices.Num());
		for (const FLevelUpChoice& C : Choices)
		{ float W=1.f; if(C.WeaponType!=EWeaponType::None){const float* F=CurrentConfig.WeaponWeights.Find((int32)C.WeaponType);W=F&&*F>0.f?*F:0.01f;} CW.Add(W); }
		ChoiceIdx = WSSampleIndex(CW, RandStream);
	}
	else ChoiceIdx = RandStream.RandRange(0, Choices.Num()-1);
	ApplyLevelUpChoice(Choices[ChoiceIdx]);
	RecalcPassiveEffects();
}

FPassiveEffects FSurvivorsGameLogic::ComputePassiveEffects() const
{
	FPassiveEffects PE;
	for (const FPassiveSlot& Slot : PassiveSlots)
	{
		if (Slot.Type == EPassiveItemType::None || Slot.Level <= 0) continue;
		const int32 Lv = Slot.Level;
		switch (Slot.Type)
		{
		case EPassiveItemType::Spinach:       PE.DamageMult += 0.10f*Lv; break;
		case EPassiveItemType::Armor:         PE.ArmorFlat  += 1.f*Lv; break;
		case EPassiveItemType::HollowHeart:   PE.HpMult     += 0.20f*Lv; break;
		case EPassiveItemType::Pummarola:     PE.RegenPerSec += SurvivorsWikiSpec::PummarolaRecoveryForLevel(Lv); break;
		case EPassiveItemType::EmptyTome:     PE.CooldownMult -= 0.08f*Lv; PE.CooldownMult = FMath::Max(PE.CooldownMult, 0.4f); break;
		case EPassiveItemType::Candelabrador: PE.AreaMult    += 0.10f*Lv; break;
		case EPassiveItemType::Bracer:        PE.SpeedMult   += 0.10f*Lv; break;
		case EPassiveItemType::Spellbinder:   PE.DurationMult += SurvivorsWikiSpec::SpellbinderDurationBonusForLevel(Lv); break;
		case EPassiveItemType::Duplicator:    PE.ExtraAmount += 1.f*Lv; break;
		case EPassiveItemType::Wings:         PE.MoveSpeedMult += SurvivorsWikiSpec::WingsMoveSpeedBonusForLevel(Lv); break;
		case EPassiveItemType::Attractorb:    PE.PickupRadiusMult = SurvivorsWikiSpec::AttractorbPickupRadiusMultiplierForLevel(Lv); break;
		case EPassiveItemType::Tirajisu:      PE.MaxRevivalCount += FMath::Min(Lv,2); break;
		case EPassiveItemType::TorronasBox:
			{ const float O=SurvivorsWikiSpec::TorronasOmniBonusForLevel(Lv);
			  PE.DamageMult+=O;PE.AreaMult+=O;PE.SpeedMult+=O;PE.DurationMult+=O;
			  PE.CooldownMult=FMath::Max(PE.CooldownMult-O,0.4f);PE.CurseMult+=SurvivorsWikiSpec::TorronasCurseBonusForLevel(Lv); } break;
		case EPassiveItemType::Crown:         PE.GrowthMult += SurvivorsWikiSpec::CrownGrowthPerLevel*(float)Lv; break;
		case EPassiveItemType::SkullOManiac:  PE.CurseMult  += SurvivorsWikiSpec::SkullCurseBonusForLevel(Lv); break;
		default: break;
		}
	}
	return PE;
}

void FSurvivorsGameLogic::RecalcPassiveEffects()
{
	CachedPassiveEffects = ComputePassiveEffects();
	const float NewMaxHP = BaseMaxPlayerHPConst * CachedPassiveEffects.HpMult;
	if (CurrentConfig.MaxPlayerHP > 0.f) PlayerHP = NewMaxHP * (PlayerHP / CurrentConfig.MaxPlayerHP);
	CurrentConfig.MaxPlayerHP = NewMaxHP;
	MaxRevivalCount = CachedPassiveEffects.MaxRevivalCount;
	CurrentConfig.GemPickupRadius = BaseGemPickupRadiusConst * CachedPassiveEffects.PickupRadiusMult;
}

TArray<int32> FSurvivorsGameLogic::GetEvolvableWeapons() const
{
	TArray<int32> Result;
	for (const SurvivorsGameConstants::FEvolutionRule& Rule : SurvivorsGameConstants::EvolutionTable)
	{
		for (int32 SI = 0; SI < MaxWeaponSlots; ++SI)
		{
			const FWeaponSlot& WS = WeaponSlots[SI];
			if (WS.Type != Rule.BaseWeapon) continue;
			if (WS.Level.Value < SurvivorsGameConstants::GetWeaponMaxLevel(WS.Type)) continue;
			if (Rule.UnionPartner != EWeaponType::None)
			{ bool bH=false; for(int32 k=0;k<MaxWeaponSlots;++k) if(WeaponSlots[k].Type==Rule.UnionPartner&&WeaponSlots[k].Level.Value>=SurvivorsGameConstants::GetWeaponMaxLevel(WeaponSlots[k].Type)){bH=true;break;} if(!bH) continue; }
			if (Rule.RequiredPassive != EPassiveItemType::None)
			{ bool bH=false; for(const FPassiveSlot& PS:PassiveSlots) if(PS.Type==Rule.RequiredPassive&&PS.Level>0){bH=true;break;} if(!bH) continue; }
			Result.Add(SI);
		}
	}
	return Result;
}

void FSurvivorsGameLogic::EvolveWeapon(int32 SlotIdx, EWeaponType EvolvedType)
{
	const EWeaponType BaseType = WeaponSlots[SlotIdx].Type;
	if (WeaponSlots[SlotIdx].Level.Value < SurvivorsGameConstants::GetWeaponMaxLevel(BaseType)) return;
	WeaponSlots[SlotIdx].Type  = EvolvedType;
	WeaponSlots[SlotIdx].Level = FWeaponLevel(1);
	EquipWeapon(SlotIdx, EvolvedType, 1);
	if (EvolvedType == EWeaponType::Vandalier)
	{
		for (int32 k=0;k<MaxWeaponSlots;++k) if(WeaponSlots[k].Type==EWeaponType::EbonyWings)
		{ WeaponSlots[k].Type=EWeaponType::None;WeaponSlots[k].Level=FWeaponLevel(0);WeaponSlots[k].Cooldown=FCooldownSeconds(0.f);UnequipWeapon(k);break; }
	}
}

TArray<FLevelUpChoice> FSurvivorsGameLogic::BuildLevelUpChoices()
{
	static const EPassiveItemType AllPT[] = {
		EPassiveItemType::Spinach,EPassiveItemType::Armor,EPassiveItemType::HollowHeart,
		EPassiveItemType::Pummarola,EPassiveItemType::EmptyTome,EPassiveItemType::Candelabrador,
		EPassiveItemType::Bracer,EPassiveItemType::Spellbinder,EPassiveItemType::Duplicator,
		EPassiveItemType::Wings,EPassiveItemType::Attractorb,EPassiveItemType::Clover,
		EPassiveItemType::Crown,EPassiveItemType::StoneMask,EPassiveItemType::SkullOManiac,
		EPassiveItemType::Tirajisu,EPassiveItemType::TorronasBox,
	};
	TArray<FLevelUpChoice> Evolutions, WeaponUpgrades, NewWeapons, PassiveUpgrades, PassiveNew;

	// 武器アップグレード
	for (int32 i=0;i<MaxWeaponSlots;++i)
	{ if(WeaponSlots[i].Type==EWeaponType::None||WeaponSlots[i].Level.Value>=SurvivorsGameConstants::GetWeaponMaxLevel(WeaponSlots[i].Type)) continue;
	  FLevelUpChoice C; C.ChoiceType=FLevelUpChoice::EChoiceType::WeaponUpgrade; C.WeaponType=WeaponSlots[i].Type; C.SlotIdx=i; C.NewLevel=WeaponSlots[i].Level.Value+1; WeaponUpgrades.Add(C); }

	// 新規武器
	{ int32 Empty=0; for(int32 i=0;i<MaxWeaponSlots;++i) if(WeaponSlots[i].Type==EWeaponType::None) ++Empty;
	  if(Empty>0)
	  { TArray<EWeaponType> Pool;
	    if(CurrentConfig.WeaponPoolMode==TEXT("garlic_only")) Pool={EWeaponType::Garlic};
	    else if((CurrentConfig.WeaponPoolMode==TEXT("fixed_subset")||CurrentConfig.WeaponPoolMode==TEXT("weighted"))&&CurrentConfig.AllowedWeaponTypes.Num()>0)
	      { for(int32 Id:CurrentConfig.AllowedWeaponTypes) Pool.Add((EWeaponType)Id); }
	    else Pool={EWeaponType::Garlic,EWeaponType::Whip,EWeaponType::MagicWand,EWeaponType::Knife,EWeaponType::Axe,EWeaponType::Cross,EWeaponType::KingBible,EWeaponType::FireWand,EWeaponType::SantaWater,EWeaponType::Runetracer,EWeaponType::LightningRing,EWeaponType::Pentagram,EWeaponType::Peachone,EWeaponType::EbonyWings,EWeaponType::Laurel};
	    for(EWeaponType WT:Pool)
	    { bool bO=false; for(int32 i=0;i<MaxWeaponSlots;++i) if(WeaponSlots[i].Type==WT){bO=true;break;} if(bO) continue;
	      FLevelUpChoice C;C.ChoiceType=FLevelUpChoice::EChoiceType::WeaponNew;C.WeaponType=WT;C.NewLevel=1;NewWeapons.Add(C); } } }

	// パッシブ
	if (CurrentConfig.bEnablePassives)
	{
		for(int32 i=0;i<MaxPassiveSlots;++i)
		{ const FPassiveSlot& PS=PassiveSlots[i]; if(PS.Type==EPassiveItemType::None||PS.Level<=0) continue;
		  const int32 ML=SurvivorsGameConstants::PassiveMaxLevel[(int32)(uint8)PS.Type]; if(PS.Level>=ML) continue;
		  FLevelUpChoice C;C.ChoiceType=FLevelUpChoice::EChoiceType::PassiveUpgrade;C.PassiveType=PS.Type;C.SlotIdx=i;C.NewLevel=PS.Level+1;PassiveUpgrades.Add(C); }
		int32 EPSlots=0; for(int32 i=0;i<MaxPassiveSlots;++i) if(PassiveSlots[i].Type==EPassiveItemType::None) ++EPSlots;
		if(EPSlots>0) for(EPassiveItemType PT:AllPT)
		{ const int32 ML=SurvivorsGameConstants::PassiveMaxLevel[(int32)(uint8)PT]; if(ML<=0) continue;
		  bool bO=false; for(int32 i=0;i<MaxPassiveSlots;++i) if(PassiveSlots[i].Type==PT){bO=true;break;} if(bO) continue;
		  FLevelUpChoice C;C.ChoiceType=FLevelUpChoice::EChoiceType::PassiveNew;C.PassiveType=PT;C.NewLevel=1;PassiveNew.Add(C); }
	}

	TArray<FLevelUpChoice> Choices;
	for(const FLevelUpChoice& C:Evolutions){if(Choices.Num()>=3)break;Choices.Add(C);}
	if(CurrentConfig.bEnablePassives&&Choices.Num()<3)
	{ TArray<FLevelUpChoice> AP; AP.Append(PassiveUpgrades); AP.Append(PassiveNew);
	  if(AP.Num()>0) Choices.Add(AP[RandStream.RandRange(0,AP.Num()-1)]); }
	TArray<FLevelUpChoice> WPool; WPool.Append(WeaponUpgrades); WPool.Append(NewWeapons);
	for(int32 j=WPool.Num()-1;j>=0;--j)
	{ for(const FLevelUpChoice& E:Choices) if(E.WeaponType==WPool[j].WeaponType&&E.WeaponType!=EWeaponType::None){WPool.RemoveAt(j);break;} }
	const bool bWL=CurrentConfig.WeaponPoolMode.Equals(TEXT("weighted"),ESearchCase::IgnoreCase)&&!CurrentConfig.WeaponWeights.IsEmpty();
	while(Choices.Num()<3&&WPool.Num()>0)
	{ int32 SI2=0;
	  if(bWL){float T=0.f;for(const FLevelUpChoice& C:WPool){float W=1.f;if(C.ChoiceType==FLevelUpChoice::EChoiceType::WeaponNew){const float* F=CurrentConfig.WeaponWeights.Find((int32)C.WeaponType);W=F&&*F>0.f?*F:1.f;}T+=W;}
	    if(T>0.f){float R=RandStream.FRandRange(0.f,T),Cum2=0.f;for(int32 j=0;j<WPool.Num();++j){float W=1.f;if(WPool[j].ChoiceType==FLevelUpChoice::EChoiceType::WeaponNew){const float* F=CurrentConfig.WeaponWeights.Find((int32)WPool[j].WeaponType);W=F&&*F>0.f?*F:1.f;}Cum2+=W;if(R<=Cum2){SI2=j;break;}}}}
	  else SI2=RandStream.RandRange(0,WPool.Num()-1);
	  Choices.Add(WPool[SI2]);WPool.RemoveAt(SI2); }
	if(CurrentConfig.bEnablePassives&&Choices.Num()<3)
	{ TArray<FLevelUpChoice> AP2; AP2.Append(PassiveUpgrades); AP2.Append(PassiveNew);
	  for(int32 j=AP2.Num()-1;j>=0;--j){for(const FLevelUpChoice& E:Choices) if(E.ChoiceType!=FLevelUpChoice::EChoiceType::WeaponNew&&E.ChoiceType!=FLevelUpChoice::EChoiceType::WeaponUpgrade&&E.ChoiceType!=FLevelUpChoice::EChoiceType::WeaponEvolve&&E.PassiveType==AP2[j].PassiveType){AP2.RemoveAt(j);break;}}
	  while(Choices.Num()<3&&AP2.Num()>0){int32 I=RandStream.RandRange(0,AP2.Num()-1);Choices.Add(AP2[I]);AP2.RemoveAt(I);} }
	return Choices;
}

void FSurvivorsGameLogic::ApplyLevelUpChoice(const FLevelUpChoice& Choice)
{
	switch (Choice.ChoiceType)
	{
	case FLevelUpChoice::EChoiceType::PassiveNew:
		for(int32 i=0;i<MaxPassiveSlots;++i) if(PassiveSlots[i].Type==EPassiveItemType::None){PassiveSlots[i].Type=Choice.PassiveType;PassiveSlots[i].Level=1;break;} return;
	case FLevelUpChoice::EChoiceType::PassiveUpgrade:
		for(int32 i=0;i<MaxPassiveSlots;++i) if(PassiveSlots[i].Type==Choice.PassiveType){const int32 ML=SurvivorsGameConstants::PassiveMaxLevel[(int32)(uint8)Choice.PassiveType];PassiveSlots[i].Level=FMath::Min(PassiveSlots[i].Level+1,ML);break;} return;
	case FLevelUpChoice::EChoiceType::WeaponEvolve:
		for(const SurvivorsGameConstants::FEvolutionRule& Rule:SurvivorsGameConstants::EvolutionTable) if(Rule.EvolvedWeapon==Choice.WeaponType){for(int32 i=0;i<MaxWeaponSlots;++i) if(WeaponSlots[i].Type==Rule.BaseWeapon){EvolveWeapon(i,Choice.WeaponType);return;} break;} return;
	case FLevelUpChoice::EChoiceType::WeaponUpgrade:
		for(int32 i=0;i<MaxWeaponSlots;++i) if(WeaponSlots[i].Type==Choice.WeaponType){const int32 NL=FMath::Min(Choice.NewLevel,SurvivorsGameConstants::GetWeaponMaxLevel(Choice.WeaponType));WeaponSlots[i].Level=FWeaponLevel(NL);if(Weapons.IsValidIndex(i)&&Weapons[i])Weapons[i]->SetLevel(FWeaponLevel(NL));return;} return;
	case FLevelUpChoice::EChoiceType::WeaponNew: default:
		for(int32 i=0;i<MaxWeaponSlots;++i) if(WeaponSlots[i].Type==EWeaponType::None){WeaponSlots[i].Type=Choice.WeaponType;WeaponSlots[i].Level=FWeaponLevel(1);EquipWeapon(i,Choice.WeaponType,1);return;} return;
	}
}

// ============================================================================
// Enemy ロジック
// ============================================================================

void FSurvivorsGameLogic::UpdateEnemies()
{
	const bool bGF = (ElapsedTime < GlobalFreezeUntilTime);
	for (FEnemyState& E : Enemies)
	{
		const bool bRF = CurrentConfig.EnemyTypeTable.IsValidIndex(E.TypeId) && CurrentConfig.EnemyTypeTable[E.TypeId].bResistsFreeze;
		if (E.bFrozen || (bGF && !bRF)) continue;
		E.Vel = (PlayerPos - E.Pos).GetSafeNormal() * GetEnemySpeed(E.TypeId);
		E.Pos += E.Vel * PhysicsDt;
	}
}

void FSurvivorsGameLogic::RecycleDistantEnemies()
{
	const float RecycleSq = FMath::Square(CurrentConfig.EnemyRecycleDistance);
	for (FEnemyState& E : Enemies)
	{
		if (E.bPendingRemove) continue;
		if (CurrentConfig.EnemyTypeTable.IsValidIndex(E.TypeId) && CurrentConfig.EnemyTypeTable[E.TypeId].bIsBoss) continue;
		if ((E.Pos - PlayerPos).SizeSquared() > RecycleSq)
		{
			E.Pos = RandomSpawnPos();
			E.Vel = FVector2D::ZeroVector;
		}
	}
}

float FSurvivorsGameLogic::GetEnemySpeed(int32 TypeId) const
{
	const float CM = FMath::Max(0.f, CachedPassiveEffects.CurseMult);
	if (!CurrentConfig.EnemyTypeTable.IsValidIndex(TypeId)) return 50.f * CurrentConfig.EnemySpeedMult * CM;
	return CurrentConfig.EnemyTypeTable[TypeId].Speed * CurrentConfig.EnemySpeedMult * CM;
}

float FSurvivorsGameLogic::GetEnemyTypeMaxHP(int32 TypeId) const
{
	if (!CurrentConfig.EnemyTypeTable.IsValidIndex(TypeId)) return 1.f;
	return CurrentConfig.EnemyTypeTable[TypeId].BaseHP;
}

void FSurvivorsGameLogic::ComputeContactHits(FSurvivorsHitFrame& HitFrame)
{
	TArray<const FSurvivorsTargetProxy*> Contacts;
	QueryEnemyContacts(PlayerPos, CurrentConfig.PlayerRadius, Contacts);
	for (const FSurvivorsTargetProxy* Proxy : Contacts)
	{
		if ((PlayerPos-Proxy->Pos).SizeSquared() > FMath::Square(CurrentConfig.PlayerRadius+Proxy->Radius)) continue;
		const int32 EI = Proxy->Ref.IndexAtBuildTime;
		if (!Enemies.IsValidIndex(EI)||Enemies[EI].UniqueId!=Proxy->Ref.UniqueId) continue;
		const FEnemyState& E = Enemies[EI];
		if (E.bPendingRemove) continue;
		if (ElapsedTime - E.PlayerLastHitTime >= ContactHitInterval)
		{ FSurvivorsHitEvent Ev; Ev.Type=ESurvivorsHitType::ContactDamage; Ev.Target=Proxy->Ref; Ev.Damage=E.ContactDamage; HitFrame.Events.Add(Ev); }
	}
}

void FSurvivorsGameLogic::ApplyContactHits(FSurvivorsHitFrame& HitFrame)
{
	for (const FSurvivorsHitEvent& Ev : HitFrame.Events)
	{
		if (Ev.Type != ESurvivorsHitType::ContactDamage) continue;
		if (bShieldActive) continue;
		const int32 EI = Ev.Target.IndexAtBuildTime;
		if (!Enemies.IsValidIndex(EI)) continue;
		FEnemyState& E = Enemies[EI];
		if (E.UniqueId!=Ev.Target.UniqueId||E.bPendingRemove) continue;
		PlayerHP -= FMath::Max(0.f, Ev.Damage - CachedPassiveEffects.ArmorFlat);
		E.PlayerLastHitTime = ElapsedTime;
	}
	PlayerHP = FMath::Max(0.f, PlayerHP);
}

void FSurvivorsGameLogic::FinalizePendingEnemies()
{
	for (int32 i = Enemies.Num()-1; i >= 0; --i)
	{
		if (!Enemies[i].bPendingRemove) continue;
		const bool bChest = CurrentConfig.EnemyTypeTable.IsValidIndex(Enemies[i].TypeId) && CurrentConfig.EnemyTypeTable[Enemies[i].TypeId].bIsBoss;
		if (bChest) { FSpecialPickupState C; C.Pos=Enemies[i].Pos; C.Type=ESpecialPickupType::TreasureChest; C.bActive=true; SpecialPickups.Add(C); }
		else DropGem(Enemies[i].TypeId, Enemies[i].Pos);
		Enemies.RemoveAt(i);
	}
}

// ============================================================================
// Gem ロジック
// ============================================================================

void FSurvivorsGameLogic::DropGem(int32 TypeId, FVector2D Pos)
{
	const EGemType CT = (TypeId>=0&&TypeId<UE_ARRAY_COUNT(SurvivorsGameConstants::GemDropTable)) ? SurvivorsGameConstants::GemDropTable[TypeId] : EGemType::Blue;
	const float XPDrop = CurrentConfig.EnemyTypeTable.IsValidIndex(TypeId) ? CurrentConfig.EnemyTypeTable[TypeId].XPDrop : SurvivorsGameConstants::GemXPValues[0];
	const EGemType Type = CT==EGemType::Red ? EGemType::Red : SurvivorsGameConstants::GemTypeForExperience(XPDrop);
	float BXP = SurvivorsGameConstants::GemXPValues[(uint8)Type];
	if (CurrentConfig.EnemyTypeTable.IsValidIndex(TypeId))
	{
		if (Type==EGemType::Red) BXP=FMath::Max(1.f,XPDrop);
		else if (Type==EGemType::Blue) BXP=FMath::Clamp(XPDrop,1.f,SurvivorsGameConstants::BlueGemMaxXP);
		else BXP=FMath::Clamp(XPDrop,SurvivorsGameConstants::BlueGemMaxXP+1.f,SurvivorsGameConstants::GreenGemMaxXP);
	}
	FGemState G; G.Pos=Pos; G.Type=Type; G.BaseExperienceValue=BXP; G.UniqueId=++NextGemId; G.bPendingRemove=false;
	Gems.Add(G);
}

void FSurvivorsGameLogic::ComputePickupHits(FSurvivorsHitFrame& HitFrame)
{
	TArray<const FSurvivorsTargetProxy*> Contacts;
	QueryPickupContacts(PlayerPos, CurrentConfig.GemPickupRadius, Contacts);
	for (const FSurvivorsTargetProxy* Proxy : Contacts)
	{
		if ((PlayerPos-Proxy->Pos).SizeSquared() > FMath::Square(CurrentConfig.GemPickupRadius)) continue;
		const int32 GI = Proxy->Ref.IndexAtBuildTime;
		if (!Gems.IsValidIndex(GI)||Gems[GI].UniqueId!=Proxy->Ref.UniqueId||Gems[GI].bPendingRemove) continue;
		FSurvivorsHitEvent Ev; Ev.Type=ESurvivorsHitType::PickupCollect; Ev.Target=Proxy->Ref; HitFrame.Events.Add(Ev);
	}
}

void FSurvivorsGameLogic::ApplyPickupHits(FSurvivorsHitFrame& HitFrame)
{
	for (const FSurvivorsHitEvent& Ev : HitFrame.Events)
	{
		if (Ev.Type != ESurvivorsHitType::PickupCollect) continue;
		const int32 GI = Ev.Target.IndexAtBuildTime;
		if (!Gems.IsValidIndex(GI)||Gems[GI].UniqueId!=Ev.Target.UniqueId||Gems[GI].bPendingRemove) continue;
		Gems[GI].bPendingRemove = true;
		float XPG = Gems[GI].BaseExperienceValue;
		if (Gems[GI].Type==EGemType::Red)
		{ const int32 Mult=RandStream.RandRange(SurvivorsGameConstants::RedGemMinMultiplier,SurvivorsGameConstants::RedGemMaxMultiplier); XPG=SurvivorsWikiSpec::RedGemExperienceForMultiplier(XPG,Mult); }
		ProcessXPGain(XPG);
		LastReward += CurrentConfig.ItemReward;
	}
}

void FSurvivorsGameLogic::FinalizePickupRemovals()
{ for (int32 i=Gems.Num()-1;i>=0;--i) if(Gems[i].bPendingRemove) Gems.RemoveAt(i); }

// ============================================================================
// Spawn ロジック
// ============================================================================

void FSurvivorsGameLogic::InitDefaultEnemyTable()
{
	struct FRow { const TCHAR* Name; float HP,Spd,Dmg,XP,R,KB; bool Boss; };
	static const FRow Rows[] = {
		{TEXT("Bat"),1.f,35.f,2.f,2.f,10.f,0.f,false},{TEXT("Zombie"),4.f,20.f,3.f,2.f,12.f,0.f,false},
		{TEXT("Skeleton"),6.f,22.5f,4.f,2.f,12.f,0.f,false},{TEXT("Ghost"),3.f,44.f,3.f,2.f,10.f,0.f,false},
		{TEXT("Werewolf"),10.f,40.f,5.f,9.f,14.f,0.f,false},{TEXT("Mummy"),15.f,18.f,6.f,9.f,14.f,0.f,false},
		{TEXT("Plant"),20.f,16.f,7.f,9.f,14.f,0.f,false},{TEXT("BatSwarm"),2.f,52.f,3.f,2.f,8.f,0.f,false},
		{TEXT("FireBeast"),30.f,32.f,10.f,9.f,16.f,0.f,false},{TEXT("MedusaHead"),25.f,48.f,10.f,9.f,12.f,0.f,false},
		{TEXT("GiantBat"),3000.f,24.f,12.f,2.f,32.f,1.f,true},
	};
	CurrentConfig.EnemyTypeTable.Empty();
	for (const FRow& R : Rows)
	{ FEnemyTypeParams P; P.Name=R.Name; P.BaseHP=R.HP; P.Speed=R.Spd; P.ContactDamage=R.Dmg; P.XPDrop=R.XP;
	  P.CollisionRadius=R.R; P.KnockbackResistance=R.KB; P.bIsBoss=R.Boss; P.bResistsFreeze=R.Boss; P.bResistsInstantKill=R.Boss; P.bResistsDebuff=R.Boss;
	  CurrentConfig.EnemyTypeTable.Add(P); }
}

void FSurvivorsGameLogic::InitDefaultSpawnWaves()
{
	CurrentConfig.SpawnWaves.Empty();
	struct FWD { int32 T; float W; };
	auto AddWave = [this](float TS, float TE, float SR, int32 MinE, int32 MaxE, const FWD* Ws, int32 WC)
	{
		FSpawnWave Wave; Wave.TimeStart=TS; Wave.TimeEnd=TE; Wave.SpawnRate=SR; Wave.MinEnemies=MinE; Wave.MaxEnemies=MaxE;
		for (int32 i=0;i<WC;++i){FEnemySpawnWeight EW;EW.TypeId=Ws[i].T;EW.Weight=Ws[i].W;Wave.EnemyWeights.Add(EW);}
		CurrentConfig.SpawnWaves.Add(MoveTemp(Wave));
	};
	{const FWD w[]={{0,1.0f}};                                      AddWave(0.f,60.f,1.0f,15,80,w,UE_ARRAY_COUNT(w));}
	{const FWD w[]={{0,0.6f},{1,0.4f}};                             AddWave(60.f,120.f,1.5f,25,120,w,UE_ARRAY_COUNT(w));}
	{const FWD w[]={{0,0.3f},{1,0.4f},{2,0.3f}};                    AddWave(120.f,180.f,2.0f,35,160,w,UE_ARRAY_COUNT(w));}
	{const FWD w[]={{1,0.3f},{2,0.3f},{3,0.25f},{4,0.15f}};         AddWave(180.f,300.f,2.5f,50,220,w,UE_ARRAY_COUNT(w));}
	{const FWD w[]={{2,0.2f},{3,0.3f},{4,0.3f},{5,0.2f}};           AddWave(300.f,420.f,3.2f,80,300,w,UE_ARRAY_COUNT(w));}
	{const FWD w[]={{4,0.2f},{5,0.3f},{6,0.2f},{7,0.3f}};           AddWave(420.f,600.f,4.0f,120,420,w,UE_ARRAY_COUNT(w));}
	{const FWD w[]={{5,0.2f},{6,0.25f},{7,0.25f},{8,0.3f}};         AddWave(600.f,900.f,5.0f,180,520,w,UE_ARRAY_COUNT(w));}
	{const FWD w[]={{6,0.2f},{7,0.2f},{8,0.3f},{9,0.3f}};           AddWave(900.f,1800.f,6.0f,240,600,w,UE_ARRAY_COUNT(w));}
}

const FSpawnWave* FSurvivorsGameLogic::GetCurrentWave() const
{ const int32 I=GetCurrentWaveIndex(); return I!=INDEX_NONE?&CurrentConfig.SpawnWaves[I]:nullptr; }

int32 FSurvivorsGameLogic::GetCurrentWaveIndex() const
{
	if (CurrentConfig.SpawnWaves.IsEmpty()) return INDEX_NONE;
	for (int32 i=0;i<CurrentConfig.SpawnWaves.Num();++i)
	{ const FSpawnWave& W=CurrentConfig.SpawnWaves[i]; if(ElapsedTime>=W.TimeStart&&ElapsedTime<W.TimeEnd) return i; }
	if (ElapsedTime >= CurrentConfig.SpawnWaves.Last().TimeEnd) return CurrentConfig.SpawnWaves.Num()-1;
	return INDEX_NONE;
}

bool FSurvivorsGameLogic::BuildSpawnWeights(const FSpawnWave& Wave, TArray<FEnemySpawnWeight>& OutW, bool& bOutCP) const
{
	OutW.Reset(); bOutCP = false;
	if (CurrentConfig.EnemyTypeTable.IsEmpty()) return false;
	const int32 MaxNorm = FMath::Min(9, CurrentConfig.EnemyTypeTable.Num()-1);
	const int32 AllowMax = FMath::Clamp(CurrentConfig.MaxEnemyTypeId, 0, MaxNorm);
	if (AllowMax < MaxNorm)
	{
		bOutCP = true;
		for (const FEnemySpawnWeight& WW : Wave.EnemyWeights)
		{ if(WW.TypeId>AllowMax||!CurrentConfig.EnemyTypeTable.IsValidIndex(WW.TypeId)||CurrentConfig.EnemyTypeTable[WW.TypeId].bIsBoss) continue;
		  FEnemySpawnWeight SW; SW.TypeId=WW.TypeId; SW.Weight=FMath::Max(WW.Weight,0.01f); OutW.Add(SW); }
		if (OutW.IsEmpty()&&CurrentConfig.EnemyTypeTable.IsValidIndex(0)&&!CurrentConfig.EnemyTypeTable[0].bIsBoss)
			OutW.Add({0, 1.f});
		return !OutW.IsEmpty();
	}
	for (const FEnemySpawnWeight& WW : Wave.EnemyWeights)
	{ if(CurrentConfig.EnemyTypeTable.IsValidIndex(WW.TypeId)&&!CurrentConfig.EnemyTypeTable[WW.TypeId].bIsBoss) OutW.Add(WW); }
	return !OutW.IsEmpty();
}

int32 FSurvivorsGameLogic::SelectTypeByWeight(const TArray<FEnemySpawnWeight>& Weights)
{
	float T=0.f; for (const FEnemySpawnWeight& W:Weights) T+=W.Weight;
	if (T<=0.f||Weights.IsEmpty()) return 0;
	float R=RandStream.FRandRange(0.f,T);
	for (const FEnemySpawnWeight& W:Weights) { R-=W.Weight; if(R<=0.f) return W.TypeId; }
	return Weights.Last().TypeId;
}

FVector2D FSurvivorsGameLogic::RandomInsideField()
{ const float H=CurrentConfig.FieldHalfSize*0.85f; return FVector2D(RandStream.FRandRange(-H,H),RandStream.FRandRange(-H,H)); }

FVector2D FSurvivorsGameLogic::RandomOnEdge()
{
	const float HS=CurrentConfig.FieldHalfSize;
	const int32 E=RandStream.RandRange(0,3); const float T=RandStream.FRandRange(-HS,HS);
	switch(E){case 0:return FVector2D(HS,T);case 1:return FVector2D(-HS,T);case 2:return FVector2D(T,HS);default:return FVector2D(T,-HS);}
}

FVector2D FSurvivorsGameLogic::RandomSpawnPos()
{
	const float A=RandStream.FRandRange(0.f,2.f*PI), D=RandStream.FRandRange(CurrentConfig.SpawnMinDistance,CurrentConfig.SpawnMaxDistance);
	const float HS=CurrentConfig.FieldHalfSize;
	FVector2D P=PlayerPos+FVector2D(FMath::Cos(A),FMath::Sin(A))*D;
	P.X=FMath::Clamp(P.X,-HS,HS); P.Y=FMath::Clamp(P.Y,-HS,HS); return P;
}

void FSurvivorsGameLogic::SpawnEnemy(const FSpawnWave& Wave)
{
	TArray<FEnemySpawnWeight> Filtered; bool bCP=false;
	BuildSpawnWeights(Wave, Filtered, bCP);
	if (Filtered.IsEmpty()) return;
	const FEnemyTypeId TypeId(SelectTypeByWeight(Filtered));
	const int32 TI = TypeId.ToIndex();
	if (!CurrentConfig.EnemyTypeTable.IsValidIndex(TI)) return;
	const FEnemyTypeParams& P = CurrentConfig.EnemyTypeTable[TI];
	const FSurvivorsElapsedTime CE(ElapsedTime);
	const float THM = CurrentConfig.bTimeScalingEnabled ? 1.f+CurrentConfig.HPScaleRatePerMin*(CE.Seconds/60.f) : 1.f;
	const float TDM = CurrentConfig.bTimeScalingEnabled ? 1.f+CurrentConfig.DamageScaleRatePerMin*(CE.Seconds/60.f) : 1.f;
	const float CM  = FMath::Max(0.f, CachedPassiveEffects.CurseMult);
	FEnemyState E;
	E.Pos=RandomSpawnPos(); E.Vel=FVector2D::ZeroVector; E.TypeId=TI; E.CollisionRadius=P.CollisionRadius;
	E.MaxHP=FHp(P.BaseHP*CurrentConfig.EnemyHPScale*THM*CM).Value; E.HP=E.MaxHP;
	E.ContactDamage=FDamage(P.ContactDamage*CurrentConfig.EnemyDamageScale*TDM*CM).Value;
	E.PlayerLastHitTime=-1000.f; E.UniqueId=NextEnemyId++;
	Enemies.Add(E);
}

void FSurvivorsGameLogic::SpawnBoss()
{
	constexpr int32 BossTypeId = 10;
	if (CurrentConfig.MaxEnemyTypeId < BossTypeId || !CurrentConfig.EnemyTypeTable.IsValidIndex(BossTypeId)) return;
	const FEnemyTypeParams& P = CurrentConfig.EnemyTypeTable[BossTypeId];
	const FSurvivorsElapsedTime CE(ElapsedTime);
	const float THM = CurrentConfig.bTimeScalingEnabled ? 1.f+CurrentConfig.HPScaleRatePerMin*(CE.Seconds/60.f) : 1.f;
	const float TDM = CurrentConfig.bTimeScalingEnabled ? 1.f+CurrentConfig.DamageScaleRatePerMin*(CE.Seconds/60.f) : 1.f;
	const float CM  = FMath::Max(0.f, CachedPassiveEffects.CurseMult);
	FEnemyState Boss;
	Boss.Pos=RandomSpawnPos(); Boss.Vel=FVector2D::ZeroVector; Boss.TypeId=BossTypeId; Boss.CollisionRadius=P.CollisionRadius;
	Boss.MaxHP=FHp(P.BaseHP*CurrentConfig.EnemyHPScale*THM*CM).Value; Boss.HP=Boss.MaxHP;
	Boss.ContactDamage=FDamage(P.ContactDamage*CurrentConfig.EnemyDamageScale*TDM*CM).Value;
	Boss.PlayerLastHitTime=-1000.f; Boss.UniqueId=NextEnemyId++;
	Enemies.Add(Boss);
	UE_LOG(LogTemp, Log, TEXT("[SurvivorsGameLogic] GiantBat spawned at t=%.1f"), ElapsedTime);
}

void FSurvivorsGameLogic::StepSpawn(float Dt)
{
	const int32 CWI = GetCurrentWaveIndex();
	const FSpawnWave* Wave = CWI != INDEX_NONE ? &CurrentConfig.SpawnWaves[CWI] : nullptr;

	LastSpawnDebug = FSurvivorsSpawnDebug();
	LastSpawnDebug.ElapsedTime=ElapsedTime; LastSpawnDebug.MaxEpisodeTime=CurrentConfig.MaxEpisodeTime;
	LastSpawnDebug.EnemyCount=Enemies.Num(); LastSpawnDebug.CurrentWaveIndex=CWI;
	LastSpawnDebug.MinActiveEnemies=CurrentConfig.MinActiveEnemies; LastSpawnDebug.MaxActiveEnemies=CurrentConfig.MaxActiveEnemies;
	LastSpawnDebug.MaxEnemyTypeId=CurrentConfig.MaxEnemyTypeId; LastSpawnDebug.TotalWaveCount=CurrentConfig.SpawnWaves.Num();
	LastSpawnDebug.SpawnAccumulator=SpawnAccumulator; LastSpawnDebug.bHasCurrentWave=Wave!=nullptr; LastSpawnDebug.bTruncated=bTruncated;

	if (Wave)
	{
		const int32 WM = Wave->MinEnemies > 0 ? Wave->MinEnemies : CurrentConfig.MinActiveEnemies;
		const float CM = FMath::Max(0.f, CachedPassiveEffects.CurseMult);
		const int32 EM = FMath::Min(FMath::RoundToInt((float)WM*CM), CurrentConfig.MaxActiveEnemies);
		const int32 EMax = FMath::Min(FMath::RoundToInt((float)Wave->MaxEnemies*CM), CurrentConfig.MaxActiveEnemies);
		TArray<FEnemySpawnWeight> SW; bool bCP=false;
		BuildSpawnWeights(*Wave, SW, bCP);
		LastSpawnDebug.EffectiveMinEnemies=EM; LastSpawnDebug.EffectiveMaxEnemies=EMax;
		LastSpawnDebug.AllowedSpawnTypeCount=SW.Num(); LastSpawnDebug.bUsedCurriculumEnemyPool=bCP; LastSpawnDebug.bSpawnBlocked=SW.IsEmpty();
		while(!SW.IsEmpty()&&Enemies.Num()<EM){const int32 B=Enemies.Num();SpawnEnemy(*Wave);if(Enemies.Num()==B)break;}
		if(Enemies.Num()<EMax)
		{ SpawnAccumulator+=Wave->SpawnRate*CurrentConfig.SpawnRateMult*CM*Dt;
		  while(!SW.IsEmpty()&&SpawnAccumulator>=1.f&&Enemies.Num()<EMax){SpawnEnemy(*Wave);SpawnAccumulator-=1.f;} }
		LastSpawnDebug.EnemyCount=Enemies.Num(); LastSpawnDebug.SpawnAccumulator=SpawnAccumulator;
	}
	if (!bBossSpawned && ElapsedTime >= CurrentConfig.BossSpawnTime) { SpawnBoss(); bBossSpawned=true; }
}

// ============================================================================
// Pickup ロジック
// ============================================================================

void FSurvivorsGameLogic::CheckFloorPickups()
{
	for (FFloorPickupState& FP : FloorPickups)
	{
		if (!FP.bActive) continue;
		if ((PlayerPos-FP.Pos).SizeSquared() > FMath::Square(CurrentConfig.FloorPickupRadius+5.f)) continue;
		FP.bActive = false;
		float Heal = 0.f;
		switch (FP.Type) { case EFloorPickupType::FloorChicken: Heal=30.f; break; case EFloorPickupType::LittleHeart: Heal=1.f; break; default: break; }
		if (Heal > 0.f) PlayerHP = FMath::Min(PlayerHP+Heal, CurrentConfig.MaxPlayerHP);
	}
	for (int32 i=FloorPickups.Num()-1;i>=0;--i) if(!FloorPickups[i].bActive) FloorPickups.RemoveAt(i);
}

void FSurvivorsGameLogic::CheckSpecialPickups()
{
	for (FSpecialPickupState& SP : SpecialPickups)
	{
		if (!SP.bActive) continue;
		if ((PlayerPos-SP.Pos).SizeSquared() > FMath::Square(CurrentConfig.FloorPickupRadius+10.f)) continue;
		SP.bActive = false;
		switch (SP.Type)
		{
		case ESpecialPickupType::Rosary:
			for (FEnemyState& E:Enemies)
			{ const bool bRI=CurrentConfig.EnemyTypeTable.IsValidIndex(E.TypeId)&&CurrentConfig.EnemyTypeTable[E.TypeId].bResistsInstantKill;
			  if(!E.bPendingRemove&&!bRI){E.HP=0.f;E.bPendingRemove=true;LastReward+=CurrentConfig.KillReward;} } break;
		case ESpecialPickupType::Vacuum: for(FGemState& G:Gems) if(!G.bPendingRemove) G.Pos=PlayerPos; break;
		case ESpecialPickupType::Orologion: GlobalFreezeUntilTime=ElapsedTime+10.f; break;
		case ESpecialPickupType::TreasureChest:
			if(CurrentConfig.bEnableEvolutions)
			{ const TArray<int32> ES=GetEvolvableWeapons();
			  if(ES.Num()>0){const int32 SI=ES[RandStream.RandRange(0,ES.Num()-1)];const EWeaponType BT=WeaponSlots[SI].Type;
			    for(const SurvivorsGameConstants::FEvolutionRule& Rule:SurvivorsGameConstants::EvolutionTable) if(Rule.BaseWeapon==BT){EvolveWeapon(SI,Rule.EvolvedWeapon);break;}} } break;
		default: break;
		}
	}
	for (int32 i=SpecialPickups.Num()-1;i>=0;--i) if(!SpecialPickups[i].bActive) SpecialPickups.RemoveAt(i);
}

// ============================================================================
// Collision ロジック
// ============================================================================

void FSurvivorsGameLogic::ResolveWallCollisions()
{
	for (const FBox2D& Box : CurrentConfig.WallBounds)
	{
		const FVector2D Closest(FMath::Clamp(PlayerPos.X,Box.Min.X,Box.Max.X), FMath::Clamp(PlayerPos.Y,Box.Min.Y,Box.Max.Y));
		const FVector2D Delta = PlayerPos - Closest;
		const float DistSq = Delta.SizeSquared();
		const float PR = CurrentConfig.PlayerRadius;
		if (DistSq < PR*PR && DistSq > KINDA_SMALL_NUMBER)
		{ const float D=FMath::Sqrt(DistSq); const FVector2D N=Delta/D; PlayerPos=Closest+N*PR; const float V=FVector2D::DotProduct(PlayerVel,N); if(V<0.f) PlayerVel-=N*V; }
		else if (DistSq <= KINDA_SMALL_NUMBER)
		{ const float px1=PlayerPos.X-Box.Min.X,px2=Box.Max.X-PlayerPos.X,py1=PlayerPos.Y-Box.Min.Y,py2=Box.Max.Y-PlayerPos.Y;
		  const float m=FMath::Min(FMath::Min(px1,px2),FMath::Min(py1,py2));
		  if(m==px1){PlayerPos.X=Box.Min.X-PR;PlayerVel.X=FMath::Min(PlayerVel.X,0.f);}
		  else if(m==px2){PlayerPos.X=Box.Max.X+PR;PlayerVel.X=FMath::Max(PlayerVel.X,0.f);}
		  else if(m==py1){PlayerPos.Y=Box.Min.Y-PR;PlayerVel.Y=FMath::Min(PlayerVel.Y,0.f);}
		  else{PlayerPos.Y=Box.Max.Y+PR;PlayerVel.Y=FMath::Max(PlayerVel.Y,0.f);} }
	}
}

float FSurvivorsGameLogic::CastRayToObstacles(FVector2D Origin, FVector2D Dir) const
{
	const float HS = CurrentConfig.FieldHalfSize;
	float tMin = TNumericLimits<float>::Max();
	if(Dir.X>1e-6f) tMin=FMath::Min(tMin,(HS-Origin.X)/Dir.X);
	if(Dir.X<-1e-6f) tMin=FMath::Min(tMin,(-HS-Origin.X)/Dir.X);
	if(Dir.Y>1e-6f) tMin=FMath::Min(tMin,(HS-Origin.Y)/Dir.Y);
	if(Dir.Y<-1e-6f) tMin=FMath::Min(tMin,(-HS-Origin.Y)/Dir.Y);
	for (const FBox2D& Box : CurrentConfig.WallBounds)
	{
		float tN=0.f, tF=TNumericLimits<float>::Max();
		if(FMath::Abs(Dir.X)>1e-6f){float t1=(Box.Min.X-Origin.X)/Dir.X,t2=(Box.Max.X-Origin.X)/Dir.X;if(t1>t2)Swap(t1,t2);tN=FMath::Max(tN,t1);tF=FMath::Min(tF,t2);}
		else if(Origin.X<Box.Min.X||Origin.X>Box.Max.X) continue;
		if(FMath::Abs(Dir.Y)>1e-6f){float t1=(Box.Min.Y-Origin.Y)/Dir.Y,t2=(Box.Max.Y-Origin.Y)/Dir.Y;if(t1>t2)Swap(t1,t2);tN=FMath::Max(tN,t1);tF=FMath::Min(tF,t2);}
		else if(Origin.Y<Box.Min.Y||Origin.Y>Box.Max.Y) continue;
		if(tN<tF&&tN>0.f) tMin=FMath::Min(tMin,tN);
	}
	return tMin<TNumericLimits<float>::Max()?tMin:0.f;
}

bool FSurvivorsGameLogic::ReflectOffWall(FVector2D& InOutPos, FVector2D& InOutVel, float Radius) const
{
	bool bR = false;
	const float HS = CurrentConfig.FieldHalfSize - Radius;
	if(InOutPos.X>HS){InOutPos.X=HS;InOutVel.X=-FMath::Abs(InOutVel.X);bR=true;}
	else if(InOutPos.X<-HS){InOutPos.X=-HS;InOutVel.X=FMath::Abs(InOutVel.X);bR=true;}
	if(InOutPos.Y>HS){InOutPos.Y=HS;InOutVel.Y=-FMath::Abs(InOutVel.Y);bR=true;}
	else if(InOutPos.Y<-HS){InOutPos.Y=-HS;InOutVel.Y=FMath::Abs(InOutVel.Y);bR=true;}
	return bR;
}

void FSurvivorsGameLogic::RegisterEnemyTargets()
{
	for (int32 i=0;i<Enemies.Num();++i)
	{ const FEnemyState& E=Enemies[i]; FSurvivorsTargetProxy P; P.Ref={ESurvivorsCollisionOwnerKind::Enemy,E.UniqueId,i}; P.Pos=E.Pos; P.Radius=E.CollisionRadius; EnemyGrid.AddTarget(P); }
}

void FSurvivorsGameLogic::RegisterPickupTargets()
{
	for (int32 i=0;i<Gems.Num();++i)
	{ const FGemState& G=Gems[i]; FSurvivorsTargetProxy P; P.Ref={ESurvivorsCollisionOwnerKind::Gem,G.UniqueId,i}; P.Pos=G.Pos; P.Radius=0.f; PickupGrid.AddTarget(P); }
}

void FSurvivorsGameLogic::BuildEnemyGrid()
{
	EnemyGrid.Rebuild(PlayerPos, 1000.f, 128.f);
	RegisterEnemyTargets();
}

void FSurvivorsGameLogic::BuildPickupGrid()
{
	PickupGrid.Rebuild(PlayerPos, 1000.f, 128.f);
	RegisterPickupTargets();
}

void FSurvivorsGameLogic::QueryEnemyContacts(FVector2D Pos, float Radius, TArray<const FSurvivorsTargetProxy*>& Out) const
{
	TArray<int32> Indices; EnemyGrid.QueryContacts(Pos, Radius+EnemyGrid.MaxTargetRadius, Indices);
	for (int32 I:Indices) Out.Add(&EnemyGrid.Targets[I]);
}

void FSurvivorsGameLogic::QueryPickupContacts(FVector2D Pos, float Radius, TArray<const FSurvivorsTargetProxy*>& Out) const
{
	TArray<int32> Indices; PickupGrid.QueryContacts(Pos, Radius+PickupGrid.MaxTargetRadius, Indices);
	for (int32 I:Indices) Out.Add(&PickupGrid.Targets[I]);
}

// ============================================================================
// Weapon ロジック
// ============================================================================

TUniquePtr<FSurvivorsWeaponLogic> FSurvivorsGameLogic::CreateWeaponLogic(EWeaponType Type)
{
	switch (Type)
	{
	case EWeaponType::Garlic:
	case EWeaponType::SoulEater:
		return MakeUnique<FSurvivorsWeaponGarlicLogic>();
	case EWeaponType::Whip:
	case EWeaponType::BloodyTear:
		return MakeUnique<FSurvivorsWeaponWhipLogic>();
	case EWeaponType::MagicWand:
	case EWeaponType::HolyWand:
		return MakeUnique<FSurvivorsWeaponMagicWandLogic>();
	case EWeaponType::Knife:
	case EWeaponType::ThousandEdge:
		return MakeUnique<FSurvivorsWeaponKnifeLogic>();
	case EWeaponType::Axe:
	case EWeaponType::DeathSpiral:
		return MakeUnique<FSurvivorsWeaponAxeLogic>();
	case EWeaponType::Cross:
	case EWeaponType::HeavenSword:
		return MakeUnique<FSurvivorsWeaponCrossLogic>();
	case EWeaponType::KingBible:
	case EWeaponType::UnholyVespers:
		return MakeUnique<FSurvivorsWeaponKingBibleLogic>();
	case EWeaponType::FireWand:
	case EWeaponType::Hellfire:
		return MakeUnique<FSurvivorsWeaponFireWandLogic>();
	case EWeaponType::SantaWater:
	case EWeaponType::LaBorra:
		return MakeUnique<FSurvivorsWeaponSantaWaterLogic>();
	case EWeaponType::Runetracer:
	case EWeaponType::NoFuture:
		return MakeUnique<FSurvivorsWeaponRunetracerLogic>();
	case EWeaponType::LightningRing:
	case EWeaponType::ThunderLoop:
		return MakeUnique<FSurvivorsWeaponLightningRingLogic>();
	case EWeaponType::Pentagram:
	case EWeaponType::GorgeousMoon:
		return MakeUnique<FSurvivorsWeaponPentagramLogic>();
	case EWeaponType::Peachone:
		return MakeUnique<FSurvivorsWeaponPeachoneLogic>();
	case EWeaponType::EbonyWings:
		return MakeUnique<FSurvivorsWeaponEbonyWingsLogic>();
	case EWeaponType::Vandalier:
		return MakeUnique<FSurvivorsWeaponVandalierLogic>();
	case EWeaponType::Laurel:
		return MakeUnique<FSurvivorsWeaponLaurelLogic>();
	default:
		return nullptr;
	}
}

void FSurvivorsGameLogic::EquipWeapon(int32 SlotIdx, EWeaponType Type, int32 Level)
{
	if (SlotIdx < 0 || SlotIdx >= MaxWeaponSlots) return;
	if (!Weapons.IsValidIndex(SlotIdx)) Weapons.SetNum(SlotIdx+1);
	Weapons[SlotIdx] = CreateWeaponLogic(Type);
	if (Weapons[SlotIdx])
	{ Weapons[SlotIdx]->Initialize(this, SlotIdx); Weapons[SlotIdx]->SetWeaponType(Type); Weapons[SlotIdx]->SetLevel(FWeaponLevel(Level)); }
}

void FSurvivorsGameLogic::UnequipWeapon(int32 SlotIdx)
{ if (Weapons.IsValidIndex(SlotIdx)) Weapons[SlotIdx].Reset(); }

void FSurvivorsGameLogic::UpdateProjectilesBySlot(int32 InSlot, float Dt, TFunctionRef<bool(FProjectileState&, float)> Callback)
{
	for (int32 i=Projectiles.Num()-1;i>=0;--i)
	{ if(Projectiles[i].WeaponSlotIdx==InSlot){if(!Callback(Projectiles[i],Dt))Projectiles.RemoveAt(i);} }
}

void FSurvivorsGameLogic::TickWeapons(float Dt)
{
	for (int32 i=0;i<Weapons.Num();++i) if(Weapons[i]) Weapons[i]->Tick(Dt);
	// TickProjectiles
	for (int32 i=Projectiles.Num()-1;i>=0;--i)
	{
		FProjectileState& P = Projectiles[i];
		P.Pos += P.Vel * Dt; P.Age += Dt; P.LifeTime.Tick(Dt);
		if (P.LifeTime.IsExpired())
		{ if((P.WeaponType==EWeaponType::FireWand||P.WeaponType==EWeaponType::Hellfire)&&!P.bPendingExplosion){P.bPendingExplosion=true;continue;}
		  Projectiles.RemoveAt(i); }
	}
	// TickGroundZones
	for (int32 i=GroundZones.Num()-1;i>=0;--i)
	{ FGroundZoneState& Z=GroundZones[i]; Z.LifeTime-=Dt;
	  if(Z.bIsWarning){Z.WarningTime=FMath::Max(0.f,Z.WarningTime-Dt);if(Z.WarningTime<=0.f){Z.bIsWarning=false;Z.EnemyLastHitTime.Empty();}}
	  if(Z.LifeTime<=0.f) GroundZones.RemoveAt(i); }
}

void FSurvivorsGameLogic::ComputeAllWeaponHits(FSurvivorsHitFrame& HitFrame)
{
	for (int32 i=0;i<Weapons.Num();++i) if(Weapons[i]) Weapons[i]->ComputeHits(HitFrame);
	ComputeGroundZoneHits(HitFrame);
	ComputeProjectileHits(HitFrame);
}

void FSurvivorsGameLogic::ComputeGroundZoneHits(FSurvivorsHitFrame& HitFrame)
{
	for (int32 ZI=0;ZI<GroundZones.Num();++ZI)
	{
		const FGroundZoneState& Z = GroundZones[ZI];
		if (Z.bIsWarning) continue;
		TArray<const FSurvivorsTargetProxy*> Contacts; QueryEnemyContacts(Z.Pos, Z.Radius, Contacts);
		for (const FSurvivorsTargetProxy* Proxy : Contacts)
		{
			if((Z.Pos-Proxy->Pos).SizeSquared()>FMath::Square(Z.Radius+Proxy->Radius)) continue;
			const int32 EI=Proxy->Ref.IndexAtBuildTime;
			if(!Enemies.IsValidIndex(EI)||Enemies[EI].UniqueId!=Proxy->Ref.UniqueId||Enemies[EI].bPendingRemove) continue;
			const FEnemyState& E=Enemies[EI];
			const float* LH=Z.EnemyLastHitTime.Find(E.UniqueId);
			if(LH&&(ElapsedTime-*LH)<Z.HitCooldown) continue;
			FSurvivorsHitEvent Ev; Ev.Type=ESurvivorsHitType::GroundZoneDamage; Ev.Target=Proxy->Ref; Ev.Damage=Z.Damage; Ev.WeaponSlot=ZI; HitFrame.Events.Add(Ev);
		}
	}
}

void FSurvivorsGameLogic::ComputeProjectileHits(FSurvivorsHitFrame& HitFrame)
{
	for (int32 ProjIdx=0;ProjIdx<Projectiles.Num();++ProjIdx)
	{
		FProjectileState& P = Projectiles[ProjIdx];
		if(P.WeaponType==EWeaponType::FireWand||P.WeaponType==EWeaponType::Hellfire||P.WeaponType==EWeaponType::Whip||P.WeaponType==EWeaponType::BloodyTear) continue;
		TArray<const FSurvivorsTargetProxy*> Contacts; QueryEnemyContacts(P.Pos, P.Radius.Value, Contacts);
		for (const FSurvivorsTargetProxy* Proxy : Contacts)
		{
			if((P.Pos-Proxy->Pos).SizeSquared()>FMath::Square(P.Radius.Value+Proxy->Radius)) continue;
			const int32 EI=Proxy->Ref.IndexAtBuildTime;
			if(!Enemies.IsValidIndex(EI)||Enemies[EI].UniqueId!=Proxy->Ref.UniqueId||Enemies[EI].bPendingRemove) continue;
			const FEnemyState& E=Enemies[EI];
			const bool bRT=(P.WeaponType==EWeaponType::Runetracer||P.WeaponType==EWeaponType::NoFuture);
			if(bRT){const float* LH=P.EnemyHitDelays.Find(E.UniqueId);if(LH&&(ElapsedTime-*LH)<SurvivorsGameConstants::RunetracerHitboxDelay) continue;}
			else if(P.HitEnemyIds.Contains(E.UniqueId)) continue;
			FSurvivorsHitEvent Ev; Ev.Type=ESurvivorsHitType::ProjectileDamage; Ev.Target=Proxy->Ref; Ev.Damage=P.Damage.Value; Ev.WeaponSlot=ProjIdx;
			if(P.KnockbackStrength>0.f){Ev.KnockbackDir=P.Vel.GetSafeNormal();Ev.KnockbackStrength=P.KnockbackStrength;}
			HitFrame.Events.Add(Ev);
			if(bRT) P.EnemyHitDelays.Add(E.UniqueId,ElapsedTime);
			else P.HitEnemyIds.Add(E.UniqueId);
			if(P.MaxPierceCount>0&&P.HitEnemyIds.Num()>=P.MaxPierceCount) break;
			else if(P.MaxPierceCount==0&&!P.bPiercing) break;
		}
	}
}

void FSurvivorsGameLogic::ApplyWeaponHits(FSurvivorsHitFrame& HitFrame)
{
	TSet<int32> PR;
	for (const FSurvivorsHitEvent& Ev : HitFrame.Events)
	{
		if(Ev.Type!=ESurvivorsHitType::WeaponAreaDamage&&Ev.Type!=ESurvivorsHitType::GroundZoneDamage&&Ev.Type!=ESurvivorsHitType::ProjectileDamage) continue;
		const int32 EI=Ev.Target.IndexAtBuildTime;
		if(!Enemies.IsValidIndex(EI)) continue;
		FEnemyState& E=Enemies[EI];
		if(E.UniqueId!=Ev.Target.UniqueId) continue;
		if(E.bPendingRemove)
		{ if(Ev.Type==ESurvivorsHitType::ProjectileDamage&&Projectiles.IsValidIndex(Ev.WeaponSlot))
		  { const FProjectileState& P=Projectiles[Ev.WeaponSlot]; if((P.MaxPierceCount>0&&P.HitEnemyIds.Num()>=P.MaxPierceCount)||(P.MaxPierceCount==0&&!P.bPiercing)) PR.Add(Ev.WeaponSlot); } continue; }
		E.HP -= Ev.Damage;
		if(Ev.Type==ESurvivorsHitType::WeaponAreaDamage){if(E.WeaponLastHitTime[Ev.WeaponSlot].Seconds<ElapsedTime)E.WeaponLastHitTime[Ev.WeaponSlot]=FSurvivorsElapsedTime(ElapsedTime);if(Ev.OrbIdx>=0){const int32 OK=Ev.WeaponSlot*10+Ev.OrbIdx;E.OrbHitTimes.Add(OK,ElapsedTime);}}
		else if(Ev.Type==ESurvivorsHitType::GroundZoneDamage){if(GroundZones.IsValidIndex(Ev.WeaponSlot))GroundZones[Ev.WeaponSlot].EnemyLastHitTime.Add(E.UniqueId,ElapsedTime);}
		else if(Ev.Type==ESurvivorsHitType::ProjectileDamage){if(Projectiles.IsValidIndex(Ev.WeaponSlot)){const FProjectileState& P=Projectiles[Ev.WeaponSlot];if((P.MaxPierceCount>0&&P.HitEnemyIds.Num()>=P.MaxPierceCount)||(P.MaxPierceCount==0&&!P.bPiercing))PR.Add(Ev.WeaponSlot);}}
		if(Ev.KnockbackStrength>0.f&&Ev.KnockbackResistance<1.f) E.Pos+=Ev.KnockbackDir*Ev.KnockbackStrength*(1.f-Ev.KnockbackResistance);
		if(E.HP<=0.f){E.bPendingRemove=true;LastReward+=CurrentConfig.KillReward;}
	}
	TArray<int32> SR=PR.Array(); SR.Sort([](int32 A,int32 B){return A>B;});
	for (int32 I:SR) if(Projectiles.IsValidIndex(I)) Projectiles.RemoveAt(I);
}
