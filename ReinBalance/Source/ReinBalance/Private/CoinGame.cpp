#include "CoinGame.h"

ACoinGame::ACoinGame()
{
	PrimaryActorTick.bCanEverTick = false;
}

void ACoinGame::BeginPlay()
{
	Super::BeginPlay();
	ResetState(TOptional<int32>());
}

void ACoinGame::ResetState(TOptional<int32> Seed)
{
	if (Seed.IsSet())
		RandStream.Initialize(Seed.GetValue());
	else
		RandStream.GenerateNewSeed();

	PlayerPos = FVector2D::ZeroVector;
	PlayerVel = FVector2D::ZeroVector;

	CoinPositions.SetNum(NumCoins);
	for (FVector2D& Coin : CoinPositions)
		Coin = RandomInsideField();

	Enemies.Empty();
	SpawnTimer = EnemySpawnInterval;
	LastReward = 0.f;
	bDone      = false;
}

void ACoinGame::PhysicsStep(int32 ActionIdx)
{
	if (bDone) return;

	LastReward = 0.f;

	// プレイヤー移動（線形ドラッグ: dv/dt = a - drag*v）
	float Ax = 0.f, Ay = 0.f;
	switch (ActionIdx)
	{
		case 0: Ay =  PlayerAccel; break; // 上 (+Y)
		case 1: Ay = -PlayerAccel; break; // 下 (-Y)
		case 2: Ax = -PlayerAccel; break; // 左 (-X)
		case 3: Ax =  PlayerAccel; break; // 右 (+X)
		default: break;                   // 静止
	}
	PlayerVel.X += (Ax - PlayerDrag * PlayerVel.X) * PhysicsDt;
	PlayerVel.Y += (Ay - PlayerDrag * PlayerVel.Y) * PhysicsDt;
	PlayerPos   += PlayerVel * PhysicsDt;
	ClampPlayerToField();

	UpdateEnemies();

	SpawnTimer -= PhysicsDt;
	if (SpawnTimer <= 0.f)
	{
		SpawnEnemy();
		SpawnTimer = EnemySpawnInterval;
	}

	CheckCoinCollections();

	if (CheckEnemyCollisions())
	{
		bDone = true;
		return;
	}

	LastReward += 0.01f;
}

TArray<float> ACoinGame::GetObservation() const
{
	TArray<float> Obs;
	Obs.Reserve(116);

	const float HN = FieldHalfSize;
	const float DN = FieldHalfSize * 2.f;

	// 1. プレイヤー位置 (2)
	Obs.Add(PlayerPos.X / HN);
	Obs.Add(PlayerPos.Y / HN);

	// 2. プレイヤー速度 (2)
	Obs.Add(PlayerVel.X);
	Obs.Add(PlayerVel.Y);

	// 3. 壁への距離 (4) : 上/下/左/右
	Obs.Add((HN - PlayerPos.Y) / HN); // 上 (+Y 壁)
	Obs.Add((HN + PlayerPos.Y) / HN); // 下 (-Y 壁)
	Obs.Add((HN + PlayerPos.X) / HN); // 左 (-X 壁)
	Obs.Add((HN - PlayerPos.X) / HN); // 右 (+X 壁)

	// 4. 現在の敵数 (1)
	Obs.Add(static_cast<float>(Enemies.Num()) / static_cast<float>(MaxEnemyObs));

	// 5. 次スポーンまでの残り時間 (1)
	Obs.Add(FMath::Clamp(SpawnTimer / EnemySpawnInterval, 0.f, 1.f));

	// 6. コイン相対位置 dx,dy × NumCoinObs=3 (6)
	// 距離近い順にソート
	TArray<int32> CoinIdx;
	CoinIdx.Reserve(CoinPositions.Num());
	for (int32 i = 0; i < CoinPositions.Num(); ++i) CoinIdx.Add(i);
	CoinIdx.Sort([&](int32 A, int32 B) {
		return FVector2D::DistSquared(CoinPositions[A], PlayerPos)
			 < FVector2D::DistSquared(CoinPositions[B], PlayerPos);
	});
	for (int32 Slot = 0; Slot < NumCoinObs; ++Slot)
	{
		if (Slot < CoinIdx.Num())
		{
			const FVector2D& C = CoinPositions[CoinIdx[Slot]];
			Obs.Add((C.X - PlayerPos.X) / DN);
			Obs.Add((C.Y - PlayerPos.Y) / DN);
		}
		else { Obs.Add(0.f); Obs.Add(0.f); }
	}

	// 距離近い順に敵をソート
	TArray<int32> EnemyIdx;
	EnemyIdx.Reserve(Enemies.Num());
	for (int32 i = 0; i < Enemies.Num(); ++i) EnemyIdx.Add(i);
	EnemyIdx.Sort([&](int32 A, int32 B) {
		return FVector2D::DistSquared(Enemies[A].Pos, PlayerPos)
			 < FVector2D::DistSquared(Enemies[B].Pos, PlayerPos);
	});

	// 7. 敵相対位置 dx,dy × MaxEnemyObs=20 (40)
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

	// 8. 敵速度 vx,vy × MaxEnemyObs=20 (40)
	for (int32 Slot = 0; Slot < MaxEnemyObs; ++Slot)
	{
		if (Slot < EnemyIdx.Num())
		{
			const FEnemyState& E = Enemies[EnemyIdx[Slot]];
			Obs.Add(E.Vel.X);
			Obs.Add(E.Vel.Y);
		}
		else { Obs.Add(0.f); Obs.Add(0.f); }
	}

	// 9. 敵の種類スカラー × MaxEnemyObs=20 (20)  A=0.0, B=0.5, C=1.0
	for (int32 Slot = 0; Slot < MaxEnemyObs; ++Slot)
	{
		if (Slot < EnemyIdx.Num())
		{
			const int32 T = Enemies[EnemyIdx[Slot]].Type;
			Obs.Add(T == 0 ? 0.0f : T == 1 ? 0.5f : 1.0f);
		}
		else { Obs.Add(0.f); }
	}

	return Obs; // 116 次元
}

float ACoinGame::GetReward() const { return LastReward; }
bool  ACoinGame::IsDone()   const { return bDone; }

FVector2D ACoinGame::RandomInsideField()
{
	return FVector2D(
		RandStream.FRandRange(-FieldHalfSize * 0.9f, FieldHalfSize * 0.9f),
		RandStream.FRandRange(-FieldHalfSize * 0.9f, FieldHalfSize * 0.9f));
}

FVector2D ACoinGame::RandomOnEdge()
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

void ACoinGame::SpawnEnemy()
{
	FEnemyState Enemy;
	Enemy.Pos  = RandomOnEdge();
	Enemy.Vel  = FVector2D::ZeroVector;
	Enemy.Type = RandStream.RandRange(0, 2);
	Enemies.Add(Enemy);
}

void ACoinGame::UpdateEnemies()
{
	for (FEnemyState& E : Enemies)
	{
		FVector2D Target;
		float Speed;
		switch (E.Type)
		{
			case 1:  Target = PlayerPos;                               Speed = EnemySpeedB; break;
			case 2:  Target = PlayerPos + PlayerVel * EnemyPredictTime; Speed = EnemySpeedC; break;
			default: Target = PlayerPos;                               Speed = EnemySpeedA; break;
		}
		E.Vel  = (Target - E.Pos).GetSafeNormal() * Speed;
		E.Pos += E.Vel * PhysicsDt;
	}
}

void ACoinGame::CheckCoinCollections()
{
	const float RadSq = CoinCollectRadius * CoinCollectRadius;
	for (FVector2D& Coin : CoinPositions)
	{
		if (FVector2D::DistSquared(PlayerPos, Coin) < RadSq)
		{
			LastReward += 1.0f;
			Coin = RandomInsideField();
		}
	}
}

bool ACoinGame::CheckEnemyCollisions() const
{
	const float RadSq = EnemyCollisionRadius * EnemyCollisionRadius;
	for (const FEnemyState& E : Enemies)
	{
		if (FVector2D::DistSquared(PlayerPos, E.Pos) < RadSq)
			return true;
	}
	return false;
}

void ACoinGame::ClampPlayerToField()
{
	if (PlayerPos.X >  FieldHalfSize) { PlayerPos.X =  FieldHalfSize; PlayerVel.X = 0.f; }
	if (PlayerPos.X < -FieldHalfSize) { PlayerPos.X = -FieldHalfSize; PlayerVel.X = 0.f; }
	if (PlayerPos.Y >  FieldHalfSize) { PlayerPos.Y =  FieldHalfSize; PlayerVel.Y = 0.f; }
	if (PlayerPos.Y < -FieldHalfSize) { PlayerPos.Y = -FieldHalfSize; PlayerVel.Y = 0.f; }
}
