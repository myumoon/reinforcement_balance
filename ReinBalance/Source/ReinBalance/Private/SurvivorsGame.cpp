#include "SurvivorsGame.h"
#include "Misc/SecureHash.h"

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

	// Phase 1: スロット0にオーラLv1を固定装備
	WeaponSlots[0].Type  = EWeaponType::Aura;
	WeaponSlots[0].Level = 1;
}

void ASurvivorsGame::BeginPlay()
{
	Super::BeginPlay();
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
		{ TEXT("spawn_timer"),   1              },
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
	PlayerHP  = MaxPlayerHP;

	ItemPositions.SetNum(NumItems);
	for (FVector2D& Item : ItemPositions)
		Item = RandomInsideField();

	Enemies.Empty();
	SpawnTimer = EnemySpawnInterval;
	LastReward = 0.f;
	bDone      = false;
}

// ---- ステップ ----------------------------------------------------------------

void ASurvivorsGame::PhysicsStep(int32 ActionIdx)
{
	if (bDone) return;

	LastReward = 0.f;

	// プレイヤー移動（線形ドラッグ）
	float Ax = 0.f, Ay = 0.f;
	switch (ActionIdx)
	{
		case 0: Ay =  PlayerAccel; break;
		case 1: Ay = -PlayerAccel; break;
		case 2: Ax = -PlayerAccel; break;
		case 3: Ax =  PlayerAccel; break;
		default: break;
	}
	PlayerVel.X += (Ax - PlayerDrag * PlayerVel.X) * PhysicsDt;
	PlayerVel.Y += (Ay - PlayerDrag * PlayerVel.Y) * PhysicsDt;
	PlayerPos   += PlayerVel * PhysicsDt;
	ClampPlayerToField();

	UpdateEnemies();
	ApplyAuraDamage();

	SpawnTimer -= PhysicsDt;
	if (SpawnTimer <= 0.f)
	{
		SpawnEnemy();
		SpawnTimer = EnemySpawnInterval;
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

	// 2. プレイヤー速度 (2)
	Obs.Add(PlayerVel.X);
	Obs.Add(PlayerVel.Y);

	// 3. 8方向壁距離 (8)
	for (int32 r = 0; r < 8; ++r)
	{
		const float Dist = CastRayToBoundary(PlayerPos, RayDirs[r]);
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

	// 7. 次スポーンまでの残り時間 (1)
	Obs.Add(FMath::Clamp(SpawnTimer / EnemySpawnInterval, 0.f, 1.f));

	// 8. XP プレースホルダー (1) — Phase1 は 0 固定
	Obs.Add(0.f);

	// 9. プレイヤーレベルプレースホルダー (1) — Phase1 は 0 固定
	Obs.Add(0.f);

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

	// enemy_type (20)  A=0.0, B=0.5, C=1.0
	for (int32 Slot = 0; Slot < MaxEnemyObs; ++Slot)
	{
		if (Slot < EnemyIdx.Num())
		{
			const int32 T = Enemies[EnemyIdx[Slot]].Type;
			Obs.Add(T == 0 ? 0.0f : T == 1 ? 0.5f : 1.0f);
		}
		else { Obs.Add(0.f); }
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

float ASurvivorsGame::GetEnemySpeed(int32 Type) const
{
	switch (Type)
	{
		case 1:  return EnemySpeedB * EnemySpeedMult;
		case 2:  return EnemySpeedC * EnemySpeedMult;
		default: return EnemySpeedA * EnemySpeedMult;
	}
}

float ASurvivorsGame::GetEnemyDamagePerTick(int32 Type) const
{
	float DPS = 0.f;
	switch (Type)
	{
		case 1:  DPS = EnemyDamageB; break;
		case 2:  DPS = EnemyDamageC; break;
		default: DPS = EnemyDamageA; break;
	}
	return DPS * PhysicsDt;
}

float ASurvivorsGame::GetEnemyMaxHP(int32 Type) const
{
	switch (Type)
	{
		case 1:  return EnemyHPB;
		case 2:  return EnemyHPC;
		default: return EnemyHPA;
	}
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

void ASurvivorsGame::SpawnEnemy()
{
	if (Enemies.Num() >= MaxActiveEnemies) return;

	FEnemyState Enemy;
	Enemy.Pos   = RandomOnEdge();
	Enemy.Vel   = FVector2D::ZeroVector;
	Enemy.Type  = RandStream.RandRange(0, 2);
	Enemy.MaxHP = GetEnemyMaxHP(Enemy.Type);
	Enemy.HP    = Enemy.MaxHP;
	Enemies.Add(Enemy);
}

void ASurvivorsGame::UpdateEnemies()
{
	for (FEnemyState& E : Enemies)
	{
		FVector2D Target;
		switch (E.Type)
		{
			case 2:  Target = PlayerPos + PlayerVel * EnemyPredictTime; break;
			default: Target = PlayerPos; break;
		}
		E.Vel = (Target - E.Pos).GetSafeNormal() * GetEnemySpeed(E.Type);
		E.Pos += E.Vel * PhysicsDt;
	}
}

void ASurvivorsGame::ApplyAuraDamage()
{
	const float DmgPerTick = AuraDPS * PhysicsDt;
	const float RadSq      = AuraRadius * AuraRadius;

	for (int32 i = Enemies.Num() - 1; i >= 0; --i)
	{
		if (FVector2D::DistSquared(PlayerPos, Enemies[i].Pos) <= RadSq)
		{
			Enemies[i].HP -= DmgPerTick;
			if (Enemies[i].HP <= 0.f)
			{
				Enemies.RemoveAt(i);
				LastReward += KillReward;
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
			Item = RandomInsideField();
		}
	}
}

void ASurvivorsGame::ApplyEnemyContactDamage()
{
	const float RadSq = EnemyCollisionRadius * EnemyCollisionRadius;
	for (const FEnemyState& E : Enemies)
	{
		if (FVector2D::DistSquared(PlayerPos, E.Pos) < RadSq)
		{
			PlayerHP -= GetEnemyDamagePerTick(E.Type);
		}
	}
	PlayerHP = FMath::Max(PlayerHP, 0.f);
}

void ASurvivorsGame::ClampPlayerToField()
{
	if (PlayerPos.X >  FieldHalfSize) { PlayerPos.X =  FieldHalfSize; PlayerVel.X = 0.f; }
	if (PlayerPos.X < -FieldHalfSize) { PlayerPos.X = -FieldHalfSize; PlayerVel.X = 0.f; }
	if (PlayerPos.Y >  FieldHalfSize) { PlayerPos.Y =  FieldHalfSize; PlayerVel.Y = 0.f; }
	if (PlayerPos.Y < -FieldHalfSize) { PlayerPos.Y = -FieldHalfSize; PlayerVel.Y = 0.f; }
}

float ASurvivorsGame::CastRayToBoundary(FVector2D Origin, FVector2D Dir) const
{
	float t = TNumericLimits<float>::Max();
	if (Dir.X >  1e-6f) t = FMath::Min(t, ( FieldHalfSize - Origin.X) / Dir.X);
	if (Dir.X < -1e-6f) t = FMath::Min(t, (-FieldHalfSize - Origin.X) / Dir.X);
	if (Dir.Y >  1e-6f) t = FMath::Min(t, ( FieldHalfSize - Origin.Y) / Dir.Y);
	if (Dir.Y < -1e-6f) t = FMath::Min(t, (-FieldHalfSize - Origin.Y) / Dir.Y);
	return t < TNumericLimits<float>::Max() ? t : 0.f;
}
