#include "Survivors/Logic/SurvivorsSpawnComponent.h"

#include "Survivors/Logic/SurvivorsGame.h"

USurvivorsSpawnComponent::USurvivorsSpawnComponent()
{
	PrimaryComponentTick.bCanEverTick = false;
}

void USurvivorsSpawnComponent::Initialize(ASurvivorsGame* InGame)
{
	Game = InGame;
}

void USurvivorsSpawnComponent::Reset()
{
	if (!Game) return;
	Game->SpawnAccumulator = 0.f;
	Game->bBossSpawned = false;
	Game->LastSpawnDebug = FSurvivorsSpawnDebug();
}

void USurvivorsSpawnComponent::InitDefaultSpawnWaves()
{
	if (!Game) return;

	Game->SpawnWaves.Empty();
	struct FWD { int32 T; float W; };
	auto AddWave = [this](float TS, float TE, float SR, int32 MinE, int32 MaxE, const FWD* Ws, int32 WCount)
	{
		FSpawnWave Wave;
		Wave.TimeStart = TS;
		Wave.TimeEnd = TE;
		Wave.SpawnRate = SR;
		Wave.MinEnemies = MinE;
		Wave.MaxEnemies = MaxE;
		for (int32 i = 0; i < WCount; ++i)
		{
			FEnemySpawnWeight EW;
			EW.TypeId = Ws[i].T;
			EW.Weight = Ws[i].W;
			Wave.EnemyWeights.Add(EW);
		}
		Game->SpawnWaves.Add(MoveTemp(Wave));
	};

	{ const FWD w[]={{0,1.0f}};                                                                 AddWave(  0.f,   60.f, 1.0f,  15,  80, w, UE_ARRAY_COUNT(w)); }
	{ const FWD w[]={{0,0.6f},{1,0.4f}};                                                        AddWave( 60.f,  120.f, 1.5f,  25, 120, w, UE_ARRAY_COUNT(w)); }
	{ const FWD w[]={{0,0.3f},{1,0.4f},{2,0.3f}};                                               AddWave(120.f,  180.f, 2.0f,  35, 160, w, UE_ARRAY_COUNT(w)); }
	{ const FWD w[]={{1,0.3f},{2,0.3f},{3,0.25f},{4,0.15f}};                                   AddWave(180.f,  300.f, 2.5f,  50, 220, w, UE_ARRAY_COUNT(w)); }
	{ const FWD w[]={{2,0.2f},{3,0.3f},{4,0.3f},{5,0.2f}};                                     AddWave(300.f,  420.f, 3.2f,  80, 300, w, UE_ARRAY_COUNT(w)); }
	{ const FWD w[]={{4,0.2f},{5,0.3f},{6,0.2f},{7,0.3f}};                                     AddWave(420.f,  600.f, 4.0f, 120, 420, w, UE_ARRAY_COUNT(w)); }
	{ const FWD w[]={{5,0.2f},{6,0.25f},{7,0.25f},{8,0.3f}};                                   AddWave(600.f,  900.f, 5.0f, 180, 520, w, UE_ARRAY_COUNT(w)); }
	{ const FWD w[]={{6,0.2f},{7,0.2f},{8,0.3f},{9,0.3f}};                                     AddWave(900.f, 1800.f, 6.0f, 240, 600, w, UE_ARRAY_COUNT(w)); }
}

void USurvivorsSpawnComponent::InitDefaultEnemyTable()
{
	if (!Game) return;

	struct FRow { const TCHAR* Name; float HP; float Spd; float Dmg; float XP; float R; float KB; bool Boss; };
	static const FRow Rows[] = {
		{ TEXT("Bat"),        1.f,    85.f,  2.f, 2.f, 10.f, 0.f, false },
		{ TEXT("Zombie"),    10.f,    45.f,  5.f, 2.f, 12.f, 0.f, false },
		{ TEXT("Skeleton"),  10.f,    55.f,  5.f, 2.f, 12.f, 0.f, false },
		{ TEXT("Ghost"),      3.f,    95.f,  3.f, 2.f, 10.f, 0.f, false },
		{ TEXT("Werewolf"),  30.f,    80.f, 10.f, 9.f, 14.f, 0.f, false },
		{ TEXT("Mummy"),     40.f,    38.f, 10.f, 9.f, 14.f, 0.f, false },
		{ TEXT("Plant"),     25.f,    35.f, 10.f, 9.f, 14.f, 0.f, false },
		{ TEXT("BatSwarm"),   2.f,   120.f,  3.f, 2.f,  8.f, 0.f, false },
		{ TEXT("FireBeast"), 60.f,    65.f, 15.f, 9.f, 16.f, 0.f, false },
		{ TEXT("MedusaHead"),50.f,    95.f, 15.f, 9.f, 12.f, 0.f, false },
		{ TEXT("GiantBat"), 3000.f,  55.f, 20.f, 2.f, 32.f, 1.f, true  },
	};

	Game->EnemyTypeTable.Empty();
	for (const FRow& R : Rows)
	{
		FEnemyTypeParams P;
		P.Name = R.Name;
		P.BaseHP = R.HP;
		P.Speed = R.Spd;
		P.ContactDamage = R.Dmg;
		P.XPDrop = R.XP;
		P.CollisionRadius = R.R;
		P.KnockbackResistance = R.KB;
		P.bIsBoss = R.Boss;
		P.bResistsFreeze = R.Boss;
		P.bResistsInstantKill = R.Boss;
		P.bResistsDebuff = R.Boss;
		Game->EnemyTypeTable.Add(P);
	}
}

void USurvivorsSpawnComponent::StepSpawn()
{
	if (!Game) return;

	const int32 CurrentWaveIndex = GetCurrentWaveIndex();
	const FSpawnWave* Wave = CurrentWaveIndex != INDEX_NONE ? &Game->SpawnWaves[CurrentWaveIndex] : nullptr;

	Game->LastSpawnDebug = FSurvivorsSpawnDebug();
	Game->LastSpawnDebug.ElapsedTime = Game->ElapsedTime;
	Game->LastSpawnDebug.MaxEpisodeTime = Game->MaxEpisodeTime;
	Game->LastSpawnDebug.EnemyCount = Game->Enemies.Num();
	Game->LastSpawnDebug.CurrentWaveIndex = CurrentWaveIndex;
	Game->LastSpawnDebug.MinActiveEnemies = Game->MinActiveEnemies;
	Game->LastSpawnDebug.MaxActiveEnemies = Game->MaxActiveEnemies;
	Game->LastSpawnDebug.MaxEnemyTypeId = Game->MaxEnemyTypeId;
	Game->LastSpawnDebug.TotalWaveCount = Game->SpawnWaves.Num();
	Game->LastSpawnDebug.SpawnAccumulator = Game->SpawnAccumulator;
	Game->LastSpawnDebug.bHasCurrentWave = Wave != nullptr;
	Game->LastSpawnDebug.bTruncated = Game->bTruncated;

	if (Wave)
	{
		const int32 WaveMin = Wave->MinEnemies > 0 ? Wave->MinEnemies : Game->MinActiveEnemies;
		const float CurseMult = FMath::Max(0.f, Game->CachedPassiveEffects.CurseMult);
		const int32 EffMin = FMath::Min(FMath::RoundToInt(static_cast<float>(WaveMin) * CurseMult), Game->MaxActiveEnemies);
		const int32 EffMax = FMath::Min(FMath::RoundToInt(static_cast<float>(Wave->MaxEnemies) * CurseMult), Game->MaxActiveEnemies);
		TArray<FEnemySpawnWeight> SpawnWeights;
		bool bUsedCurriculumPool = false;
		BuildSpawnWeights(*Wave, SpawnWeights, bUsedCurriculumPool);

		Game->LastSpawnDebug.EffectiveMinEnemies = EffMin;
		Game->LastSpawnDebug.EffectiveMaxEnemies = EffMax;
		Game->LastSpawnDebug.AllowedSpawnTypeCount = SpawnWeights.Num();
		Game->LastSpawnDebug.bUsedCurriculumEnemyPool = bUsedCurriculumPool;
		Game->LastSpawnDebug.bSpawnBlocked = SpawnWeights.IsEmpty();

		while (!SpawnWeights.IsEmpty() && Game->Enemies.Num() < EffMin)
		{
			const int32 Before = Game->Enemies.Num();
			SpawnEnemy(*Wave);
			if (Game->Enemies.Num() == Before) break;
		}

		if (Game->Enemies.Num() < EffMax)
		{
			Game->SpawnAccumulator += Wave->SpawnRate * Game->SpawnRateMult * CurseMult * SurvivorsGameConstants::PhysicsDt;
			while (!SpawnWeights.IsEmpty() && Game->SpawnAccumulator >= 1.f && Game->Enemies.Num() < EffMax)
			{
				SpawnEnemy(*Wave);
				Game->SpawnAccumulator -= 1.f;
			}
		}

		Game->LastSpawnDebug.EnemyCount = Game->Enemies.Num();
		Game->LastSpawnDebug.SpawnAccumulator = Game->SpawnAccumulator;
	}

	if (!Game->bBossSpawned && Game->ElapsedTime >= Game->BossSpawnTime)
	{
		SpawnBoss();
		Game->bBossSpawned = true;
	}
}

FVector2D USurvivorsSpawnComponent::RandomInsideField()
{
	if (!Game) return FVector2D::ZeroVector;
	return FVector2D(
		Game->RandStream.FRandRange(-Game->FieldHalfSize * 0.85f, Game->FieldHalfSize * 0.85f),
		Game->RandStream.FRandRange(-Game->FieldHalfSize * 0.85f, Game->FieldHalfSize * 0.85f));
}

FVector2D USurvivorsSpawnComponent::RandomOnEdge()
{
	if (!Game) return FVector2D::ZeroVector;
	const int32 Edge = Game->RandStream.RandRange(0, 3);
	const float T = Game->RandStream.FRandRange(-Game->FieldHalfSize, Game->FieldHalfSize);
	switch (Edge)
	{
		case 0:  return FVector2D( Game->FieldHalfSize, T);
		case 1:  return FVector2D(-Game->FieldHalfSize, T);
		case 2:  return FVector2D(T,  Game->FieldHalfSize);
		default: return FVector2D(T, -Game->FieldHalfSize);
	}
}

FVector2D USurvivorsSpawnComponent::RandomSpawnPos()
{
	if (!Game) return FVector2D::ZeroVector;
	const float Angle = Game->RandStream.FRandRange(0.f, 2.f * PI);
	const float Dist = Game->RandStream.FRandRange(Game->SpawnMinDistance, Game->SpawnMaxDistance);
	FVector2D Pos = Game->PlayerPos + FVector2D(FMath::Cos(Angle), FMath::Sin(Angle)) * Dist;
	Pos.X = FMath::Clamp(Pos.X, -Game->FieldHalfSize, Game->FieldHalfSize);
	Pos.Y = FMath::Clamp(Pos.Y, -Game->FieldHalfSize, Game->FieldHalfSize);
	return Pos;
}

int32 USurvivorsSpawnComponent::SelectTypeByWeight(const TArray<FEnemySpawnWeight>& Weights)
{
	if (!Game) return 0;

	float Total = 0.f;
	for (const FEnemySpawnWeight& W : Weights) Total += W.Weight;
	if (Total <= 0.f || Weights.Num() == 0) return 0;

	float R = Game->RandStream.FRandRange(0.f, Total);
	for (const FEnemySpawnWeight& W : Weights)
	{
		R -= W.Weight;
		if (R <= 0.f) return W.TypeId;
	}
	return Weights.Last().TypeId;
}

const FSpawnWave* USurvivorsSpawnComponent::GetCurrentWave() const
{
	const int32 WaveIndex = GetCurrentWaveIndex();
	return Game && WaveIndex != INDEX_NONE ? &Game->SpawnWaves[WaveIndex] : nullptr;
}

int32 USurvivorsSpawnComponent::GetCurrentWaveIndex() const
{
	if (!Game || Game->SpawnWaves.IsEmpty())
	{
		return INDEX_NONE;
	}

	for (int32 i = 0; i < Game->SpawnWaves.Num(); ++i)
	{
		const FSpawnWave& Wave = Game->SpawnWaves[i];
		if (Game->ElapsedTime >= Wave.TimeStart && Game->ElapsedTime < Wave.TimeEnd)
		{
			return i;
		}
	}

	if (Game->ElapsedTime >= Game->SpawnWaves.Last().TimeEnd)
	{
		return Game->SpawnWaves.Num() - 1;
	}

	return INDEX_NONE;
}

bool USurvivorsSpawnComponent::BuildSpawnWeights(const FSpawnWave& Wave, TArray<FEnemySpawnWeight>& OutWeights, bool& bOutUsedCurriculumPool) const
{
	OutWeights.Reset();
	bOutUsedCurriculumPool = false;

	if (!Game || Game->EnemyTypeTable.IsEmpty())
	{
		return false;
	}

	const int32 MaxNormalEnemyTypeId = FMath::Min(9, Game->EnemyTypeTable.Num() - 1);
	const int32 AllowedMaxTypeId = FMath::Clamp(Game->MaxEnemyTypeId, 0, MaxNormalEnemyTypeId);
	if (AllowedMaxTypeId < MaxNormalEnemyTypeId)
	{
		bOutUsedCurriculumPool = true;

		// Wave に記載されており AllowedMaxTypeId 以内の TypeId のみ採用する。
		// 未記載 TypeId に weight=1.0 を付与すると意図しない敵種が大量出現するため除外する。
		for (const FEnemySpawnWeight& WaveWeight : Wave.EnemyWeights)
		{
			if (WaveWeight.TypeId > AllowedMaxTypeId) continue;
			if (!Game->EnemyTypeTable.IsValidIndex(WaveWeight.TypeId)) continue;
			if (Game->EnemyTypeTable[WaveWeight.TypeId].bIsBoss) continue;

			FEnemySpawnWeight SpawnWeight;
			SpawnWeight.TypeId = WaveWeight.TypeId;
			SpawnWeight.Weight = FMath::Max(WaveWeight.Weight, 0.01f);
			OutWeights.Add(SpawnWeight);
		}

		// Wave に対象 TypeId が存在しない場合（例: 後半 Wave で全種が上限超え）は
		// TypeId 0 (Bat) にフォールバックして空プールを防ぐ。
		if (OutWeights.IsEmpty() && Game->EnemyTypeTable.IsValidIndex(0)
			&& !Game->EnemyTypeTable[0].bIsBoss)
		{
			OutWeights.Add({ 0, 1.f });
		}

		return !OutWeights.IsEmpty();
	}

	for (const FEnemySpawnWeight& WaveWeight : Wave.EnemyWeights)
	{
		if (Game->EnemyTypeTable.IsValidIndex(WaveWeight.TypeId) && !Game->EnemyTypeTable[WaveWeight.TypeId].bIsBoss)
		{
			OutWeights.Add(WaveWeight);
		}
	}
	return !OutWeights.IsEmpty();
}

void USurvivorsSpawnComponent::SpawnEnemy(const FSpawnWave& Wave)
{
	if (!Game) return;

	TArray<FEnemySpawnWeight> Filtered;
	bool bUsedCurriculumPool = false;
	BuildSpawnWeights(Wave, Filtered, bUsedCurriculumPool);
	if (Filtered.IsEmpty()) return;

	const FEnemyTypeId TypeId(SelectTypeByWeight(Filtered));
	const int32 TypeIdx = TypeId.ToIndex();
	if (!Game->EnemyTypeTable.IsValidIndex(TypeIdx)) return;

	const FEnemyTypeParams& Params = Game->EnemyTypeTable[TypeIdx];
	const FSurvivorsElapsedTime CurrentElapsed(Game->ElapsedTime);
	const float TimeHPMult = Game->bTimeScalingEnabled ? 1.f + Game->HPScaleRatePerMin * (CurrentElapsed.Seconds / 60.f) : 1.f;
	const float TimeDmgMult = Game->bTimeScalingEnabled ? 1.f + Game->DamageScaleRatePerMin * (CurrentElapsed.Seconds / 60.f) : 1.f;
	const float CurseMult = FMath::Max(0.f, Game->CachedPassiveEffects.CurseMult);
	const FHp MaxHP(Params.BaseHP * Game->EnemyHPScale * TimeHPMult * CurseMult);
	const FDamage ContactDamage(Params.ContactDamage * Game->EnemyDamageScale * TimeDmgMult * CurseMult);

	FEnemyState Enemy;
	Enemy.Pos = RandomSpawnPos();
	Enemy.Vel = FVector2D::ZeroVector;
	Enemy.TypeId = TypeIdx;
	Enemy.CollisionRadius = Params.CollisionRadius;
	Enemy.MaxHP = MaxHP.Value;
	Enemy.HP = Enemy.MaxHP;
	Enemy.ContactDamage = ContactDamage.Value;
	Enemy.PlayerLastHitTime = -1000.f;
	// UniqueId の採番（GroundZone の EnemyLastHitTime TMap キー用）
	Enemy.UniqueId = Game->NextEnemyId++;
	// WeaponLastHitTime は構造体デフォルトで -1000f に初期化済み
	Game->Enemies.Add(Enemy);
}

void USurvivorsSpawnComponent::SpawnBoss()
{
	if (!Game) return;

	constexpr int32 BossTypeId = 10;
	if (Game->MaxEnemyTypeId < BossTypeId) return;
	if (!Game->EnemyTypeTable.IsValidIndex(BossTypeId)) return;

	const FEnemyTypeParams& Params = Game->EnemyTypeTable[BossTypeId];
	const FSurvivorsElapsedTime CurrentElapsed(Game->ElapsedTime);
	const float TimeHPMult = Game->bTimeScalingEnabled ? 1.f + Game->HPScaleRatePerMin * (CurrentElapsed.Seconds / 60.f) : 1.f;
	const float TimeDmgMult = Game->bTimeScalingEnabled ? 1.f + Game->DamageScaleRatePerMin * (CurrentElapsed.Seconds / 60.f) : 1.f;
	const float CurseMult = FMath::Max(0.f, Game->CachedPassiveEffects.CurseMult);
	const FHp MaxHP(Params.BaseHP * Game->EnemyHPScale * TimeHPMult * CurseMult);
	const FDamage ContactDamage(Params.ContactDamage * Game->EnemyDamageScale * TimeDmgMult * CurseMult);

	FEnemyState Boss;
	Boss.Pos = RandomSpawnPos();
	Boss.Vel = FVector2D::ZeroVector;
	Boss.TypeId = BossTypeId;
	Boss.CollisionRadius = Params.CollisionRadius;
	Boss.MaxHP = MaxHP.Value;
	Boss.HP = Boss.MaxHP;
	Boss.ContactDamage = ContactDamage.Value;
	Boss.PlayerLastHitTime = -1000.f;
	Boss.UniqueId = Game->NextEnemyId++;
	Game->Enemies.Add(Boss);
	UE_LOG(LogTemp, Log, TEXT("[SurvivorsGame] GiantBat spawned at t=%.1f"), Game->ElapsedTime);
}
