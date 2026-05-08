#include "SurvivorsGame.h"
#include "WallActor.h"
#include "Kismet/GameplayStatics.h"
#include "Misc/SecureHash.h"

// Garlic レベル別ステータス（仕様: weapons_garlic.md §2）
// Lv1〜8: damage[HP], hit_interval[s], area_radius[u]
struct FGarlicParams { float Damage; float HitInterval; float AreaRadius; };
static constexpr float  GarlicKnockbackStrength = 20.f; // [u] 全レベル共通（仕様: knockback.md §2）
static constexpr FGarlicParams GarlicTable[8] = {
	{  5.f, 1.30f,  80.f }, // Lv1
	{  5.f, 1.25f,  95.f }, // Lv2
	{ 10.f, 1.20f, 110.f }, // Lv3
	{ 10.f, 1.15f, 125.f }, // Lv4
	{ 15.f, 1.10f, 140.f }, // Lv5
	{ 15.f, 1.05f, 155.f }, // Lv6
	{ 20.f, 1.00f, 170.f }, // Lv7
	{ 20.f, 0.95f, 185.f }, // Lv8
};

// 8方向レイキャスト方向（右回り 0°=右）
const FVector2D ASurvivorsGame::RayDirs[8] = {
	FVector2D( 1.f,       0.f      ), //   0° 右
	FVector2D( 0.7071f,   0.7071f  ), //  45° 右上
	FVector2D( 0.f,       1.f      ), //  90° 上
	FVector2D(-0.7071f,   0.7071f  ), // 135° 左上
	FVector2D(-1.f,       0.f      ), // 180° 左
	FVector2D(-0.7071f,  -0.7071f  ), // 225° 左下
	FVector2D( 0.f,      -1.f      ), // 270° 下
	FVector2D( 0.7071f,  -0.7071f  ), // 315° 右下
};

ASurvivorsGame::ASurvivorsGame()
{
	PrimaryActorTick.bCanEverTick = false;

	WeaponSlots[0].Type  = EWeaponType::Aura;
	WeaponSlots[0].Level = 1;

	InitDefaultEnemyTable();
	InitDefaultSpawnWaves();
}

void ASurvivorsGame::InitDefaultSpawnWaves()
{
	SpawnWaves.Empty();

	// ウェーブ追加ヘルパー（C スタイル配列で初期化リストを安全に渡す）
	struct FWD { int32 T; float W; };
	auto AddWave = [this](float TS, float TE, float SR, int32 ME, const FWD* Ws, int32 WCount)
	{
		FSpawnWave Wave;
		Wave.TimeStart  = TS; Wave.TimeEnd = TE;
		Wave.SpawnRate  = SR; Wave.MaxEnemies = ME;
		for (int32 i = 0; i < WCount; ++i)
		{
			FEnemySpawnWeight EW; EW.TypeId = Ws[i].T; EW.Weight = Ws[i].W;
			Wave.EnemyWeights.Add(EW);
		}
		SpawnWaves.Add(MoveTemp(Wave));
	};

	// type_id: Bat=0, Zombie=1, Skeleton=2, Ghost=3, Werewolf=4,
	//          Mummy=5, Plant=6, BatSwarm=7, FireBeast=8, MedusaHead=9
	{ const FWD w[]={{0,1.0f}};                                                                 AddWave(  0.f,   30.f, 1.0f,  50, w, UE_ARRAY_COUNT(w)); }
	{ const FWD w[]={{0,0.7f},{1,0.3f}};                                                        AddWave( 30.f,   60.f, 1.5f,  80, w, UE_ARRAY_COUNT(w)); }
	{ const FWD w[]={{0,0.4f},{1,0.3f},{2,0.2f},{3,0.1f}};                                     AddWave( 60.f,  120.f, 2.5f, 120, w, UE_ARRAY_COUNT(w)); }
	{ const FWD w[]={{1,0.3f},{2,0.3f},{3,0.2f},{4,0.2f}};                                     AddWave(120.f,  180.f, 3.5f, 160, w, UE_ARRAY_COUNT(w)); }
	{ const FWD w[]={{2,0.25f},{3,0.2f},{4,0.3f},{5,0.25f}};                                   AddWave(180.f,  240.f, 4.0f, 200, w, UE_ARRAY_COUNT(w)); }
	{ const FWD w[]={{4,0.3f},{5,0.3f},{6,0.3f},{3,0.1f}};                                     AddWave(240.f,  300.f, 4.5f, 250, w, UE_ARRAY_COUNT(w)); }
	{ const FWD w[]={{4,0.25f},{5,0.2f},{6,0.2f},{7,0.35f}};                                   AddWave(300.f,  420.f, 5.0f, 300, w, UE_ARRAY_COUNT(w)); }
	{ const FWD w[]={{5,0.2f},{6,0.2f},{7,0.3f},{8,0.3f}};                                     AddWave(420.f,  480.f, 6.0f, 400, w, UE_ARRAY_COUNT(w)); }
	{ const FWD w[]={{6,0.15f},{7,0.25f},{8,0.3f},{9,0.3f}};                                   AddWave(480.f,  600.f, 7.0f, 500, w, UE_ARRAY_COUNT(w)); }
	{ const FWD w[]={{7,0.25f},{8,0.35f},{9,0.4f}};                                             AddWave(600.f, 1800.f, 8.0f, 600, w, UE_ARRAY_COUNT(w)); }
}

void ASurvivorsGame::InitDefaultEnemyTable()
{
	// Mad Forest 敵11種（VS 仕様）。type_id = 配列インデックス。
	// { Name, BaseHP, Speed[u/s], ContactDamage[/hit], CollisionRadius[u], KnockbackResistance, bIsBoss }
	struct FRow { const TCHAR* Name; float HP; float Spd; float Dmg; float R; float KB; bool Boss; };
	static const FRow Rows[] = {
		{ TEXT("Bat"),        1.f,    70.f,  2.f, 10.f, 0.f, false },
		{ TEXT("Zombie"),     4.f,    40.f,  3.f, 12.f, 0.f, false },
		{ TEXT("Skeleton"),   6.f,    45.f,  4.f, 12.f, 0.f, false },
		{ TEXT("Ghost"),      3.f,    88.f,  3.f, 10.f, 0.f, false },
		{ TEXT("Werewolf"),  10.f,    80.f,  5.f, 14.f, 0.f, false },
		{ TEXT("Mummy"),     15.f,    36.f,  6.f, 14.f, 0.f, false },
		{ TEXT("Plant"),     20.f,    32.f,  7.f, 14.f, 0.f, false },
		{ TEXT("BatSwarm"),   2.f,   104.f,  3.f,  8.f, 0.f, false },
		{ TEXT("FireBeast"), 30.f,    64.f, 10.f, 16.f, 0.f, false },
		{ TEXT("MedusaHead"),25.f,    96.f, 10.f, 12.f, 0.f, false },
		{ TEXT("GiantBat"), 3000.f,  48.f, 12.f, 32.f, 1.f, true  },
	};

	EnemyTypeTable.Empty();
	for (const FRow& R : Rows)
	{
		FEnemyTypeParams P;
		P.Name                = R.Name;
		P.BaseHP              = R.HP;
		P.Speed               = R.Spd;
		P.ContactDamage       = R.Dmg;
		P.CollisionRadius     = R.R;
		P.KnockbackResistance = R.KB;
		P.bIsBoss             = R.Boss;
		EnemyTypeTable.Add(P);
	}
}

void ASurvivorsGame::BeginPlay()
{
	Super::BeginPlay();

	TArray<AActor*> Found;
	UGameplayStatics::GetAllActorsOfClass(GetWorld(), AWallActor::StaticClass(), Found);
	for (AActor* A : Found)
		if (AWallActor* W = Cast<AWallActor>(A)) WallActors.Add(W);

	UE_LOG(LogTemp, Log, TEXT("[SurvivorsGame] WallActors found: %d"), WallActors.Num());
	ResetState(TOptional<int32>());
}

// ---- スキーマ ----------------------------------------------------------------

TArray<FSurvivorsObsSegment> ASurvivorsGame::GetObsSchema() const
{
	return {
		{ TEXT("player_pos"),    2              },
		{ TEXT("player_vel"),    2              },
		{ TEXT("wall_rays"),     8              },
		{ TEXT("player_hp"),     1              },
		{ TEXT("weapon_slots"),  MaxWeaponSlots * 2 },
		{ TEXT("enemy_count"),   1              },
		{ TEXT("elapsed_time"),  1              },
		{ TEXT("xp_progress"),   1              },
		{ TEXT("player_level"),  1              },
		{ TEXT("item_rel_pos"),  NumItemObs * 2 },
		{ TEXT("enemy_rel_pos"), MaxEnemyObs * 2 },
		{ TEXT("enemy_vel"),     MaxEnemyObs * 2 },
		{ TEXT("enemy_type"),    MaxEnemyObs    },
		{ TEXT("enemy_hp"),      MaxEnemyObs    },
	};
}

FString ASurvivorsGame::GetObsSchemaHash() const
{
	FString Schema = FString::Printf(
		TEXT("SurvivorsGame,NumItemObs=%d,MaxEnemyObs=%d,MaxWeaponSlots=%d"),
		NumItemObs, MaxEnemyObs, MaxWeaponSlots);
	return FMD5::HashAnsiString(*Schema);
}

// ---- リセット ----------------------------------------------------------------

void ASurvivorsGame::ResetState(TOptional<int32> Seed)
{
	if (Seed.IsSet())
		RandStream.Initialize(Seed.GetValue());
	else
		RandStream.GenerateNewSeed();

	PlayerPos = FVector2D::ZeroVector;
	PlayerVel = FVector2D::ZeroVector;
	PlayerHP    = MaxPlayerHP;
	PlayerXP    = 0.f;
	PlayerLevel = 0;

	// Garlic は Lv1 スタート（プレイヤーレベルアップごとに +1、上限 Lv8）
	WeaponSlots[0].Type  = EWeaponType::Aura;
	WeaponSlots[0].Level = 1;
	AuraRadius = GarlicTable[0].AreaRadius; // 80u

	ItemPositions.SetNum(NumItems);
	for (FVector2D& Item : ItemPositions)
		Item = RandomInsideField();

	Enemies.Empty();
	ElapsedTime      = 0.f;
	SpawnAccumulator = 0.f;
	bBossSpawned     = false;
	LastReward       = 0.f;
	bDone            = false;
}

// ---- ステップ ----------------------------------------------------------------

void ASurvivorsGame::PhysicsStep(int32 ActionIdx)
{
	if (bDone) return;

	LastReward = 0.f;

	// プレイヤー移動（直接速度モデル: VS 仕様に準拠）
	FVector2D MoveDir = FVector2D::ZeroVector;
	switch (ActionIdx)
	{
		case 0: MoveDir.Y =  1.f; break;
		case 1: MoveDir.Y = -1.f; break;
		case 2: MoveDir.X = -1.f; break;
		case 3: MoveDir.X =  1.f; break;
		default: break;
	}
	PlayerVel  = MoveDir * MoveSpeed;
	PlayerPos += PlayerVel * PhysicsDt;
	ResolveWallCollisions();

	ElapsedTime += PhysicsDt;
	UpdateEnemies();
	ApplyAuraDamage();

	// Wave ベーススポーン（accumulator 方式）
	if (const FSpawnWave* Wave = GetCurrentWave())
	{
		const int32 EffMax = FMath::Min(Wave->MaxEnemies, MaxActiveEnemies);
		if (Enemies.Num() < EffMax)
		{
			SpawnAccumulator += Wave->SpawnRate * SpawnRateMult * PhysicsDt;
			while (SpawnAccumulator >= 1.f && Enemies.Num() < EffMax)
			{
				SpawnEnemy(*Wave);
				SpawnAccumulator -= 1.f;
			}
		}
	}

	// ボス 1 回限りスポーン
	if (!bBossSpawned && ElapsedTime >= BossSpawnTime)
	{
		SpawnBoss();
		bBossSpawned = true;
	}

	CheckItemCollections();
	ApplyEnemyContactDamage();

	if (PlayerHP <= 0.f)
	{
		bDone = true;
		return;
	}

	LastReward += AliveReward;
}

// ---- 観測 -------------------------------------------------------------------

TArray<float> ASurvivorsGame::GetObservation() const
{
	TArray<float> Obs;
	Obs.Reserve(GetObsDim());

	const float HN = FieldHalfSize;
	const float DN = FieldHalfSize * 2.f;
	const float MaxRayDist = FieldHalfSize * 2.f; // 正規化基準距離

	// 1. プレイヤー位置 (2)
	Obs.Add(PlayerPos.X / HN);
	Obs.Add(PlayerPos.Y / HN);

	// 2. プレイヤー速度 (2) — MoveSpeed で正規化 (−1〜1)
	Obs.Add(MoveSpeed > 0.f ? PlayerVel.X / MoveSpeed : 0.f);
	Obs.Add(MoveSpeed > 0.f ? PlayerVel.Y / MoveSpeed : 0.f);

	// 3. 8方向壁距離 (8)
	for (int32 r = 0; r < 8; ++r)
	{
		const float Dist = CastRayToObstacles(PlayerPos, RayDirs[r]);
		Obs.Add(FMath::Clamp(Dist / MaxRayDist, 0.f, 1.f));
	}

	// 4. プレイヤー HP (1)
	Obs.Add(FMath::Clamp(PlayerHP / MaxPlayerHP, 0.f, 1.f));

	// 5. 武器スロット × MaxWeaponSlots (6)
	for (int32 s = 0; s < MaxWeaponSlots; ++s)
	{
		const FWeaponSlot& Slot = WeaponSlots[s];
		Obs.Add(static_cast<float>(static_cast<uint8>(Slot.Type)) / static_cast<float>(MaxWeaponTypeSlots));
		Obs.Add(static_cast<float>(Slot.Level) / static_cast<float>(MaxWeaponLevel));
	}

	// 6. 現在の敵数 (1)
	Obs.Add(static_cast<float>(Enemies.Num()) / static_cast<float>(MaxEnemyObs));

	// 7. 経過時間 (1): 0〜1 (MaxGameTime=1800s で正規化)
	Obs.Add(FMath::Clamp(ElapsedTime / MaxGameTime, 0.f, 1.f));

	// 8. xp_progress (1)
	{
		const float Threshold = XPRequiredForLevel(PlayerLevel);
		Obs.Add(Threshold > 0.f ? FMath::Clamp(PlayerXP / Threshold, 0.f, 1.f) : 0.f);
	}

	// 9. player_level (1)
	Obs.Add(static_cast<float>(PlayerLevel) / static_cast<float>(MaxPlayerLevel));

	// 10. アイテム相対位置 dx,dy × NumItemObs (近い順)
	TArray<int32> ItemIdx;
	ItemIdx.Reserve(ItemPositions.Num());
	for (int32 i = 0; i < ItemPositions.Num(); ++i) ItemIdx.Add(i);
	ItemIdx.Sort([&](int32 A, int32 B) {
		return FVector2D::DistSquared(ItemPositions[A], PlayerPos)
			 < FVector2D::DistSquared(ItemPositions[B], PlayerPos);
	});
	for (int32 Slot = 0; Slot < NumItemObs; ++Slot)
	{
		if (Slot < ItemIdx.Num())
		{
			const FVector2D& It = ItemPositions[ItemIdx[Slot]];
			Obs.Add((It.X - PlayerPos.X) / DN);
			Obs.Add((It.Y - PlayerPos.Y) / DN);
		}
		else { Obs.Add(0.f); Obs.Add(0.f); }
	}

	// 11-14. 敵情報 × MaxEnemyObs (近い順)
	TArray<int32> EnemyIdx;
	EnemyIdx.Reserve(Enemies.Num());
	for (int32 i = 0; i < Enemies.Num(); ++i) EnemyIdx.Add(i);
	EnemyIdx.Sort([&](int32 A, int32 B) {
		return FVector2D::DistSquared(Enemies[A].Pos, PlayerPos)
			 < FVector2D::DistSquared(Enemies[B].Pos, PlayerPos);
	});

	// enemy_rel_pos dx,dy (40)
	for (int32 Slot = 0; Slot < MaxEnemyObs; ++Slot)
	{
		if (Slot < EnemyIdx.Num())
		{
			const FEnemyState& E = Enemies[EnemyIdx[Slot]];
			Obs.Add((E.Pos.X - PlayerPos.X) / DN);
			Obs.Add((E.Pos.Y - PlayerPos.Y) / DN);
		}
		else { Obs.Add(0.f); Obs.Add(0.f); }
	}

	// enemy_vel vx,vy (40)
	for (int32 Slot = 0; Slot < MaxEnemyObs; ++Slot)
	{
		if (Slot < EnemyIdx.Num())
		{
			const FEnemyState& E = Enemies[EnemyIdx[Slot]];
			Obs.Add(E.Vel.X); Obs.Add(E.Vel.Y);
		}
		else { Obs.Add(0.f); Obs.Add(0.f); }
	}

	// enemy_type (20)  type_id / (EnemyTypeTable.Num()-1) → [0, 1]
	const float TypeNorm = EnemyTypeTable.Num() > 1
		? static_cast<float>(EnemyTypeTable.Num() - 1) : 1.f;
	for (int32 Slot = 0; Slot < MaxEnemyObs; ++Slot)
	{
		if (Slot < EnemyIdx.Num())
			Obs.Add(static_cast<float>(Enemies[EnemyIdx[Slot]].TypeId) / TypeNorm);
		else
			Obs.Add(0.f);
	}

	// enemy_hp (20)
	for (int32 Slot = 0; Slot < MaxEnemyObs; ++Slot)
	{
		if (Slot < EnemyIdx.Num())
		{
			const FEnemyState& E = Enemies[EnemyIdx[Slot]];
			Obs.Add(FMath::Clamp(E.HP / E.MaxHP, 0.f, 1.f));
		}
		else { Obs.Add(0.f); }
	}

	return Obs;
}

float ASurvivorsGame::GetReward() const { return LastReward; }
bool  ASurvivorsGame::IsDone()   const { return bDone; }

// ---- 内部ユーティリティ -------------------------------------------------------

float ASurvivorsGame::GetEnemySpeed(int32 TypeId) const
{
	if (!EnemyTypeTable.IsValidIndex(TypeId)) return 50.f * EnemySpeedMult;
	return EnemyTypeTable[TypeId].Speed * EnemySpeedMult;
}

float ASurvivorsGame::GetEnemyTypeMaxHP(int32 TypeId) const
{
	if (!EnemyTypeTable.IsValidIndex(TypeId)) return 1.f;
	return EnemyTypeTable[TypeId].BaseHP;
}

float ASurvivorsGame::XPRequiredForLevel(int32 Level) const
{
	return XPBase + XPGrowth * static_cast<float>(Level);
}

void ASurvivorsGame::ProcessXPGain(float Amount)
{
	if (XPBase <= 0.f) return;
	PlayerXP += Amount;
	while (PlayerLevel < MaxPlayerLevel)
	{
		const float Threshold = XPRequiredForLevel(PlayerLevel);
		if (PlayerXP < Threshold) break;
		PlayerXP -= Threshold;
		PlayerLevel++;
		OnLevelUp(PlayerLevel);
	}
}

void ASurvivorsGame::OnLevelUp(int32 NextLevel)
{
	// Garlic はプレイヤーレベルアップごとに +1 Lv（上限 MaxWeaponLevel=8）
	const int32 NewGarlicLv = FMath::Min(WeaponSlots[0].Level + 1, MaxWeaponLevel);
	WeaponSlots[0].Level = NewGarlicLv;
	AuraRadius = GarlicTable[NewGarlicLv - 1].AreaRadius;
}

FVector2D ASurvivorsGame::RandomInsideField()
{
	return FVector2D(
		RandStream.FRandRange(-FieldHalfSize * 0.85f, FieldHalfSize * 0.85f),
		RandStream.FRandRange(-FieldHalfSize * 0.85f, FieldHalfSize * 0.85f));
}

FVector2D ASurvivorsGame::RandomOnEdge()
{
	const int32 Edge = RandStream.RandRange(0, 3);
	const float T    = RandStream.FRandRange(-FieldHalfSize, FieldHalfSize);
	switch (Edge)
	{
		case 0:  return FVector2D( FieldHalfSize, T);
		case 1:  return FVector2D(-FieldHalfSize, T);
		case 2:  return FVector2D(T,  FieldHalfSize);
		default: return FVector2D(T, -FieldHalfSize);
	}
}

FVector2D ASurvivorsGame::RandomSpawnPos()
{
	const float Angle = RandStream.FRandRange(0.f, 2.f * PI);
	const float Dist  = RandStream.FRandRange(SpawnMinDistance, SpawnMaxDistance);
	FVector2D Pos = PlayerPos + FVector2D(FMath::Cos(Angle), FMath::Sin(Angle)) * Dist;
	Pos.X = FMath::Clamp(Pos.X, -FieldHalfSize, FieldHalfSize);
	Pos.Y = FMath::Clamp(Pos.Y, -FieldHalfSize, FieldHalfSize);
	return Pos;
}

int32 ASurvivorsGame::SelectTypeByWeight(const TArray<FEnemySpawnWeight>& Weights)
{
	float Total = 0.f;
	for (const FEnemySpawnWeight& W : Weights) Total += W.Weight;
	if (Total <= 0.f || Weights.Num() == 0) return 0;

	float R = RandStream.FRandRange(0.f, Total);
	for (const FEnemySpawnWeight& W : Weights)
	{
		R -= W.Weight;
		if (R <= 0.f) return W.TypeId;
	}
	return Weights.Last().TypeId;
}

const FSpawnWave* ASurvivorsGame::GetCurrentWave() const
{
	for (const FSpawnWave& Wave : SpawnWaves)
		if (ElapsedTime >= Wave.TimeStart && ElapsedTime < Wave.TimeEnd)
			return &Wave;
	return nullptr;
}

void ASurvivorsGame::SpawnEnemy(const FSpawnWave& Wave)
{
	if (Wave.EnemyWeights.Num() == 0) return;
	const int32 TypeIdx = SelectTypeByWeight(Wave.EnemyWeights);
	if (!EnemyTypeTable.IsValidIndex(TypeIdx)) return;

	const FEnemyTypeParams& Params = EnemyTypeTable[TypeIdx];
	FEnemyState Enemy;
	Enemy.Pos               = RandomSpawnPos();
	Enemy.Vel               = FVector2D::ZeroVector;
	Enemy.TypeId            = TypeIdx;
	Enemy.CollisionRadius   = Params.CollisionRadius;
	Enemy.MaxHP             = Params.BaseHP;
	Enemy.HP                = Params.BaseHP;
	Enemy.GarlicLastHitTime = -1000.f; // 初回ヒットを即時許可
	Enemy.PlayerLastHitTime = -1000.f;
	Enemies.Add(Enemy);
}

void ASurvivorsGame::SpawnBoss()
{
	constexpr int32 BossTypeId = 10; // GiantBat
	if (!EnemyTypeTable.IsValidIndex(BossTypeId)) return;

	const FEnemyTypeParams& Params = EnemyTypeTable[BossTypeId];
	FEnemyState Boss;
	Boss.Pos               = RandomSpawnPos();
	Boss.Vel               = FVector2D::ZeroVector;
	Boss.TypeId            = BossTypeId;
	Boss.CollisionRadius   = Params.CollisionRadius;
	Boss.MaxHP             = Params.BaseHP;
	Boss.HP                = Params.BaseHP;
	Boss.GarlicLastHitTime = -1000.f;
	Boss.PlayerLastHitTime = -1000.f;
	Enemies.Add(Boss); // 上限カウント外（仕様: 別カウント）
	UE_LOG(LogTemp, Log, TEXT("[SurvivorsGame] GiantBat spawned at t=%.1f"), ElapsedTime);
}

void ASurvivorsGame::UpdateEnemies()
{
	// 全敵がプレイヤーに直線追尾（VS 仕様: AI 差異なし）
	for (FEnemyState& E : Enemies)
	{
		E.Vel = (PlayerPos - E.Pos).GetSafeNormal() * GetEnemySpeed(E.TypeId);
		E.Pos += E.Vel * PhysicsDt;
	}
}

void ASurvivorsGame::ApplyAuraDamage()
{
	const int32 GarlicLv   = FMath::Clamp(WeaponSlots[0].Level, 1, MaxWeaponLevel);
	const FGarlicParams& GP = GarlicTable[GarlicLv - 1];

	for (int32 i = Enemies.Num() - 1; i >= 0; --i)
	{
		FEnemyState& E = Enemies[i];
		const float Dist = FVector2D::Distance(PlayerPos, E.Pos);
		if (Dist <= GP.AreaRadius + E.CollisionRadius)
		{
			if (ElapsedTime - E.GarlicLastHitTime >= GP.HitInterval)
			{
				E.HP -= GP.Damage;
				E.GarlicLastHitTime = ElapsedTime;

				// ノックバック: プレイヤー → 敵方向に押し出す（耐性1.0は完全免疫）
				if (EnemyTypeTable.IsValidIndex(E.TypeId))
				{
					const float Resistance = EnemyTypeTable[E.TypeId].KnockbackResistance;
					if (Resistance < 1.f)
					{
						const FVector2D Dir = (E.Pos - PlayerPos).GetSafeNormal();
						E.Pos += Dir * GarlicKnockbackStrength * (1.f - Resistance);
					}
				}

				if (E.HP <= 0.f)
				{
					Enemies.RemoveAt(i);
					LastReward += KillReward;
					ProcessXPGain(ItemXP * KillXPRatio);
				}
			}
		}
	}
}

void ASurvivorsGame::CheckItemCollections()
{
	const float RadSq = ItemCollectRadius * ItemCollectRadius;
	for (FVector2D& Item : ItemPositions)
	{
		if (FVector2D::DistSquared(PlayerPos, Item) < RadSq)
		{
			LastReward += ItemReward;
			ProcessXPGain(ItemXP);
			Item = RandomInsideField();
		}
	}
}

void ASurvivorsGame::ApplyEnemyContactDamage()
{
	// 敵ごとに 0.5s 無敵: 同一敵との連続ヒットを制限
	for (FEnemyState& E : Enemies)
	{
		const float HitR = PlayerRadius + E.CollisionRadius;
		if (FVector2D::DistSquared(PlayerPos, E.Pos) < HitR * HitR)
		{
			if (ElapsedTime - E.PlayerLastHitTime >= ContactHitInterval)
			{
				if (EnemyTypeTable.IsValidIndex(E.TypeId))
					PlayerHP -= EnemyTypeTable[E.TypeId].ContactDamage;
				E.PlayerLastHitTime = ElapsedTime;
			}
		}
	}
	PlayerHP = FMath::Max(PlayerHP, 0.f);
}

void ASurvivorsGame::ResolveWallCollisions()
{
	for (const TObjectPtr<AWallActor>& Wall : WallActors)
	{
		if (!Wall) continue;
		const FBox2D Box = Wall->GetSimBounds(SimToUE);

		const FVector2D Closest(
			FMath::Clamp(PlayerPos.X, Box.Min.X, Box.Max.X),
			FMath::Clamp(PlayerPos.Y, Box.Min.Y, Box.Max.Y));
		const FVector2D Delta = PlayerPos - Closest;
		const float DistSq = Delta.SizeSquared();

		if (DistSq < PlayerRadius * PlayerRadius && DistSq > KINDA_SMALL_NUMBER)
		{
			const float Dist  = FMath::Sqrt(DistSq);
			const FVector2D N = Delta / Dist;
			PlayerPos = Closest + N * PlayerRadius;
			const float VdotN = FVector2D::DotProduct(PlayerVel, N);
			if (VdotN < 0.f) PlayerVel -= N * VdotN;
		}
		else if (DistSq <= KINDA_SMALL_NUMBER)
		{
			// 完全埋まり → 最小ペネトレーション面に押し出す
			const float px1 = PlayerPos.X - Box.Min.X, px2 = Box.Max.X - PlayerPos.X;
			const float py1 = PlayerPos.Y - Box.Min.Y, py2 = Box.Max.Y - PlayerPos.Y;
			const float m = FMath::Min(FMath::Min(px1, px2), FMath::Min(py1, py2));
			if      (m == px1) { PlayerPos.X = Box.Min.X - PlayerRadius; PlayerVel.X = FMath::Min(PlayerVel.X, 0.f); }
			else if (m == px2) { PlayerPos.X = Box.Max.X + PlayerRadius; PlayerVel.X = FMath::Max(PlayerVel.X, 0.f); }
			else if (m == py1) { PlayerPos.Y = Box.Min.Y - PlayerRadius; PlayerVel.Y = FMath::Min(PlayerVel.Y, 0.f); }
			else               { PlayerPos.Y = Box.Max.Y + PlayerRadius; PlayerVel.Y = FMath::Max(PlayerVel.Y, 0.f); }
		}
	}
}

float ASurvivorsGame::CastRayToObstacles(FVector2D Origin, FVector2D Dir) const
{
	// フォールバック: フィールド境界（AWallActor が配置されていない場合の安全網）
	float tMin = TNumericLimits<float>::Max();
	if (Dir.X >  1e-6f) tMin = FMath::Min(tMin, ( FieldHalfSize - Origin.X) / Dir.X);
	if (Dir.X < -1e-6f) tMin = FMath::Min(tMin, (-FieldHalfSize - Origin.X) / Dir.X);
	if (Dir.Y >  1e-6f) tMin = FMath::Min(tMin, ( FieldHalfSize - Origin.Y) / Dir.Y);
	if (Dir.Y < -1e-6f) tMin = FMath::Min(tMin, (-FieldHalfSize - Origin.Y) / Dir.Y);

	// AWallActor AABB スラブ法
	for (const TObjectPtr<AWallActor>& Wall : WallActors)
	{
		if (!Wall) continue;
		const FBox2D Box = Wall->GetSimBounds(SimToUE);

		float tNear = -TNumericLimits<float>::Max();
		float tFar  =  TNumericLimits<float>::Max();

		if (FMath::Abs(Dir.X) > 1e-6f)
		{
			float t1 = (Box.Min.X - Origin.X) / Dir.X;
			float t2 = (Box.Max.X - Origin.X) / Dir.X;
			if (t1 > t2) Swap(t1, t2);
			tNear = FMath::Max(tNear, t1);
			tFar  = FMath::Min(tFar,  t2);
		}
		else if (Origin.X < Box.Min.X || Origin.X > Box.Max.X) continue;

		if (FMath::Abs(Dir.Y) > 1e-6f)
		{
			float t1 = (Box.Min.Y - Origin.Y) / Dir.Y;
			float t2 = (Box.Max.Y - Origin.Y) / Dir.Y;
			if (t1 > t2) Swap(t1, t2);
			tNear = FMath::Max(tNear, t1);
			tFar  = FMath::Min(tFar,  t2);
		}
		else if (Origin.Y < Box.Min.Y || Origin.Y > Box.Max.Y) continue;

		if (tNear < tFar && tNear > 0.f)
			tMin = FMath::Min(tMin, tNear);
	}

	return tMin < TNumericLimits<float>::Max() ? tMin : 0.f;
}
