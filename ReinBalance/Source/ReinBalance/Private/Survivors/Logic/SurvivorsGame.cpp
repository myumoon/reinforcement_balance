#include "Survivors/Logic/SurvivorsGame.h"
#include "Survivors/Logic/SurvivorsCollisionComponent.h"
#include "Survivors/Logic/SurvivorsEnemyComponent.h"
#include "Survivors/Logic/SurvivorsGemComponent.h"
#include "Survivors/Logic/SurvivorsObservationComponent.h"
#include "Survivors/Logic/SurvivorsPlayerComponent.h"
#include "Survivors/Logic/SurvivorsSpawnComponent.h"

ASurvivorsGame::ASurvivorsGame()
{
	PrimaryActorTick.bCanEverTick = false;

	PlayerComponent = CreateDefaultSubobject<USurvivorsPlayerComponent>(TEXT("PlayerComponent"));
	GemComponent = CreateDefaultSubobject<USurvivorsGemComponent>(TEXT("GemComponent"));
	EnemyComponent = CreateDefaultSubobject<USurvivorsEnemyComponent>(TEXT("EnemyComponent"));
	SpawnComponent = CreateDefaultSubobject<USurvivorsSpawnComponent>(TEXT("SpawnComponent"));
	CollisionComponent = CreateDefaultSubobject<USurvivorsCollisionComponent>(TEXT("CollisionComponent"));
	ObservationComponent = CreateDefaultSubobject<USurvivorsObservationComponent>(TEXT("ObservationComponent"));

	PlayerComponent->Initialize(this);
	GemComponent->Initialize(this);
	EnemyComponent->Initialize(this);
	SpawnComponent->Initialize(this);
	CollisionComponent->Initialize(this);
	ObservationComponent->Initialize(this);

	WeaponSlots[0].Type  = EWeaponType::Aura;
	WeaponSlots[0].Level = 1;

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
void ASurvivorsGame::ResetState(TOptional<int32> Seed)
{
	if (Seed.IsSet())
		RandStream.Initialize(Seed.GetValue());
	else
		RandStream.GenerateNewSeed();

	PlayerComponent->Reset();
	GemComponent->Reset();
	EnemyComponent->Reset();
	SpawnComponent->Reset();

	ElapsedTime = 0.f;
	LastReward = 0.f;
	bDone = false;
	bTruncated = false;
}
void ASurvivorsGame::PhysicsStep(int32 ActionIdx)
{
	if (bDone || bTruncated) return;

	LastReward = 0.f;

	PlayerComponent->ApplyAction(ActionIdx);
	CollisionComponent->ResolveWallCollisions();

	ElapsedTime += SurvivorsGameConstants::PhysicsDt;
	EnemyComponent->UpdateEnemies();
	EnemyComponent->ApplyAuraDamage();
	SpawnComponent->StepSpawn();
	EnemyComponent->ApplyContactDamage();
	GemComponent->CheckCollections();

	if (PlayerHP <= 0.f)
	{
		bDone = true;
		return;
	}

	LastReward += AliveReward;

	if (MaxEpisodeTime > 0.f && ElapsedTime >= MaxEpisodeTime)
	{
		bTruncated = true;
		LastSpawnDebug.bTruncated = true;
	}
}
TArray<float> ASurvivorsGame::GetObservation() const
{
	return ObservationComponent->GetObservation();
}
float ASurvivorsGame::GetReward() const { return LastReward; }
bool  ASurvivorsGame::IsDone()   const { return bDone; }
bool  ASurvivorsGame::IsTruncated() const { return bTruncated; }

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
	EnemyComponent->ApplyAuraDamage();
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
