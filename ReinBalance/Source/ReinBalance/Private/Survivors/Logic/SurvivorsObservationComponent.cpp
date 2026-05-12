#include "Survivors/Logic/SurvivorsObservationComponent.h"

#include "Survivors/Logic/SurvivorsCollisionComponent.h"
#include "Survivors/Logic/SurvivorsGame.h"
#include "Survivors/Logic/SurvivorsPlayerComponent.h"
#include "Misc/SecureHash.h"

USurvivorsObservationComponent::USurvivorsObservationComponent()
{
	PrimaryComponentTick.bCanEverTick = false;
}

void USurvivorsObservationComponent::Initialize(ASurvivorsGame* InGame)
{
	Game = InGame;
}

TArray<FSurvivorsObsSegment> USurvivorsObservationComponent::GetObsSchema() const
{
	return {
		{ TEXT("player_pos"),               2 },
		{ TEXT("player_vel"),               2 },
		{ TEXT("wall_rays"),                8 },
		{ TEXT("player_hp"),                1 },
		{ TEXT("weapon_slots"),             SurvivorsGameConstants::MaxWeaponSlots * 2 },
		{ TEXT("enemy_count"),              1 },
		{ TEXT("elapsed_time"),             1 },
		{ TEXT("xp_progress"),              1 },
		{ TEXT("player_level"),             1 },
		{ TEXT("gem_rel_pos"),              SurvivorsGameConstants::NumGemObs * 2 },
		{ TEXT("enemy_rel_pos"),            SurvivorsGameConstants::MaxEnemyObs * 2 },
		{ TEXT("enemy_vel"),                SurvivorsGameConstants::MaxEnemyObs * 2 },
		{ TEXT("enemy_type"),               SurvivorsGameConstants::MaxEnemyObs },
		{ TEXT("enemy_hp"),                 SurvivorsGameConstants::MaxEnemyObs },
		{ TEXT("enemy_nearest_dist_16dir"), SurvivorsGameConstants::EnemyDensityDirCount },
		{ TEXT("enemy_density_near_16dir"), SurvivorsGameConstants::EnemyDensityDirCount },
		{ TEXT("enemy_density_mid_16dir"),  SurvivorsGameConstants::EnemyDensityDirCount },
		{ TEXT("gem_nearest_dist_16dir"),   SurvivorsGameConstants::GemDensityDirCount },
		{ TEXT("gem_density_near_16dir"),   SurvivorsGameConstants::GemDensityDirCount },
		{ TEXT("gem_density_mid_16dir"),    SurvivorsGameConstants::GemDensityDirCount },
	};
}

FString USurvivorsObservationComponent::GetObsSchemaHash() const
{
	FString Schema = FString::Printf(
		TEXT("SurvivorsGame,NumGemObs=%d,MaxEnemyObs=%d,MaxWeaponSlots=%d,enemy_vel_normed"
		     ",EnemyDensityDirCount=%d,EnemyNearestDistanceMax=%.0f"
		     ",EnemyDensityNearDistanceMax=%.0f,EnemyDensityMidDistanceMax=%.0f"
		     ",GemDensityDirCount=%d,GemNearestDistanceMax=%.0f"
		     ",GemDensityNearDistanceMax=%.0f,GemDensityMidDistanceMax=%.0f"),
		SurvivorsGameConstants::NumGemObs,
		SurvivorsGameConstants::MaxEnemyObs,
		SurvivorsGameConstants::MaxWeaponSlots,
		SurvivorsGameConstants::EnemyDensityDirCount,
		SurvivorsGameConstants::EnemyNearestDistanceMax,
		SurvivorsGameConstants::EnemyDensityNearDistanceMax,
		SurvivorsGameConstants::EnemyDensityMidDistanceMax,
		SurvivorsGameConstants::GemDensityDirCount,
		SurvivorsGameConstants::GemNearestDistanceMax,
		SurvivorsGameConstants::GemDensityNearDistanceMax,
		SurvivorsGameConstants::GemDensityMidDistanceMax);
	return FMD5::HashAnsiString(*Schema);
}

// 対象位置リストからプレイヤー中心の方向別特徴を計算して Obs に追記するヘルパー。
// Nearest[dir]    : dir 方向内の最近傍距離を NearestDistMax で正規化（0=近い危険, 1=遠い安全）
// NearDensity[dir]: 近距離リング（0〜NearDistMax）の距離重み付き密度 / NearNorm
// MidDensity[dir] : 中距離リング（NearDistMax〜MidDistMax）の距離重み付き密度 / MidNorm
static void BuildDirectionalDensityFeatures(
	const TArray<FVector2D>& Positions,
	const FVector2D& PlayerPos,
	int32 DirCount,
	float NearestDistMax,
	float NearDistMax,
	float MidDistMax,
	float NearNorm,
	float MidNorm,
	TArray<float>& OutObs)
{
	TArray<float> Nearest;
	TArray<float> NearDensity;
	TArray<float> MidDensity;
	Nearest.Init(1.0f, DirCount);
	NearDensity.Init(0.0f, DirCount);
	MidDensity.Init(0.0f, DirCount);

	for (const FVector2D& TargetPos : Positions)
	{
		const FVector2D Rel = TargetPos - PlayerPos;
		const float Dist = Rel.Size();
		if (Dist <= KINDA_SMALL_NUMBER)
		{
			continue;
		}

		const float AngleRad = FMath::Atan2(Rel.Y, Rel.X);
		const float Angle01  = (AngleRad + PI) / (2.0f * PI);
		const int32 Dir      = FMath::Clamp(FMath::FloorToInt(Angle01 * DirCount), 0, DirCount - 1);

		const float NearestValue = FMath::Clamp(Dist / NearestDistMax, 0.0f, 1.0f);
		Nearest[Dir] = FMath::Min(Nearest[Dir], NearestValue);

		if (Dist <= NearDistMax)
		{
			const float Weight = FMath::Clamp(1.0f - Dist / NearDistMax, 0.0f, 1.0f);
			NearDensity[Dir] += Weight;
		}
		else if (Dist <= MidDistMax)
		{
			const float T      = (Dist - NearDistMax) / (MidDistMax - NearDistMax);
			const float Weight = FMath::Clamp(1.0f - T, 0.0f, 1.0f);
			MidDensity[Dir] += Weight;
		}
	}

	for (int32 d = 0; d < DirCount; ++d) OutObs.Add(Nearest[d]);
	for (int32 d = 0; d < DirCount; ++d) OutObs.Add(FMath::Clamp(NearDensity[d] / NearNorm, 0.0f, 1.0f));
	for (int32 d = 0; d < DirCount; ++d) OutObs.Add(FMath::Clamp(MidDensity[d] / MidNorm, 0.0f, 1.0f));
}

TArray<float> USurvivorsObservationComponent::GetObservation() const
{
	TArray<float> Obs;
	if (!Game || !Game->CollisionComponent || !Game->PlayerComponent) return Obs;

	Obs.Reserve(Game->GetObsDim());

	const float HN = Game->FieldHalfSize;
	const float DN = Game->FieldHalfSize * 2.f;
	const float MaxRayDist = Game->FieldHalfSize * 2.f;

	Obs.Add(Game->PlayerPos.X / HN);
	Obs.Add(Game->PlayerPos.Y / HN);

	Obs.Add(Game->MoveSpeed > 0.f ? Game->PlayerVel.X / Game->MoveSpeed : 0.f);
	Obs.Add(Game->MoveSpeed > 0.f ? Game->PlayerVel.Y / Game->MoveSpeed : 0.f);

	for (int32 r = 0; r < 8; ++r)
	{
		const float Dist = Game->CollisionComponent->CastRayToObstacles(Game->PlayerPos, SurvivorsGameConstants::RayDirs[r]);
		Obs.Add(FMath::Clamp(Dist / MaxRayDist, 0.f, 1.f));
	}

	Obs.Add(FMath::Clamp(Game->PlayerHP / Game->MaxPlayerHP, 0.f, 1.f));

	for (int32 s = 0; s < SurvivorsGameConstants::MaxWeaponSlots; ++s)
	{
		const FWeaponSlot& Slot = Game->WeaponSlots[s];
		Obs.Add(static_cast<float>(static_cast<uint8>(Slot.Type)) / static_cast<float>(SurvivorsGameConstants::MaxWeaponTypeSlots));
		Obs.Add(static_cast<float>(Slot.Level) / static_cast<float>(SurvivorsGameConstants::MaxWeaponLevel));
	}

	Obs.Add(static_cast<float>(Game->Enemies.Num()) / static_cast<float>(SurvivorsGameConstants::MaxEnemyObs));
	Obs.Add(FMath::Clamp(Game->ElapsedTime / SurvivorsGameConstants::MaxGameTime, 0.f, 1.f));

	const float CurCum = Game->PlayerComponent->CumulativeXPForLevel(Game->PlayerLevel);
	const float NextCum = Game->PlayerComponent->CumulativeXPForLevel(Game->PlayerLevel + 1);
	const float Range = NextCum - CurCum;
	Obs.Add(Range > 0.f ? FMath::Clamp((Game->PlayerXP - CurCum) / Range, 0.f, 1.f) : 0.f);

	Obs.Add(FMath::Clamp(static_cast<float>(Game->PlayerLevel) / static_cast<float>(SurvivorsGameConstants::MaxPlayerLevel), 0.f, 1.f));

	TArray<int32> GemIdx;
	GemIdx.Reserve(Game->Gems.Num());
	for (int32 i = 0; i < Game->Gems.Num(); ++i) GemIdx.Add(i);
	GemIdx.Sort([this](int32 A, int32 B) {
		return FVector2D::DistSquared(Game->Gems[A].Pos, Game->PlayerPos)
			 < FVector2D::DistSquared(Game->Gems[B].Pos, Game->PlayerPos);
	});
	for (int32 Slot = 0; Slot < SurvivorsGameConstants::NumGemObs; ++Slot)
	{
		if (Slot < GemIdx.Num())
		{
			const FVector2D& GPos = Game->Gems[GemIdx[Slot]].Pos;
			Obs.Add((GPos.X - Game->PlayerPos.X) / DN);
			Obs.Add((GPos.Y - Game->PlayerPos.Y) / DN);
		}
		else
		{
			Obs.Add(0.f);
			Obs.Add(0.f);
		}
	}

	TArray<int32> EnemyIdx;
	EnemyIdx.Reserve(Game->Enemies.Num());
	for (int32 i = 0; i < Game->Enemies.Num(); ++i) EnemyIdx.Add(i);
	EnemyIdx.Sort([this](int32 A, int32 B) {
		return FVector2D::DistSquared(Game->Enemies[A].Pos, Game->PlayerPos)
			 < FVector2D::DistSquared(Game->Enemies[B].Pos, Game->PlayerPos);
	});

	for (int32 Slot = 0; Slot < SurvivorsGameConstants::MaxEnemyObs; ++Slot)
	{
		if (Slot < EnemyIdx.Num())
		{
			const FEnemyState& E = Game->Enemies[EnemyIdx[Slot]];
			Obs.Add((E.Pos.X - Game->PlayerPos.X) / DN);
			Obs.Add((E.Pos.Y - Game->PlayerPos.Y) / DN);
		}
		else
		{
			Obs.Add(0.f);
			Obs.Add(0.f);
		}
	}

	const float VN = Game->MoveSpeed > 0.f ? Game->MoveSpeed : 1.f;
	for (int32 Slot = 0; Slot < SurvivorsGameConstants::MaxEnemyObs; ++Slot)
	{
		if (Slot < EnemyIdx.Num())
		{
			const FEnemyState& E = Game->Enemies[EnemyIdx[Slot]];
			Obs.Add(E.Vel.X / VN);
			Obs.Add(E.Vel.Y / VN);
		}
		else
		{
			Obs.Add(0.f);
			Obs.Add(0.f);
		}
	}

	const float TypeNorm = Game->EnemyTypeTable.Num() > 1
		? static_cast<float>(Game->EnemyTypeTable.Num() - 1) : 1.f;
	for (int32 Slot = 0; Slot < SurvivorsGameConstants::MaxEnemyObs; ++Slot)
	{
		if (Slot < EnemyIdx.Num())
			Obs.Add(static_cast<float>(Game->Enemies[EnemyIdx[Slot]].TypeId) / TypeNorm);
		else
			Obs.Add(0.f);
	}

	for (int32 Slot = 0; Slot < SurvivorsGameConstants::MaxEnemyObs; ++Slot)
	{
		if (Slot < EnemyIdx.Num())
		{
			const FEnemyState& E = Game->Enemies[EnemyIdx[Slot]];
			Obs.Add(FMath::Clamp(E.HP / E.MaxHP, 0.f, 1.f));
		}
		else
		{
			Obs.Add(0.f);
		}
	}

	// 敵の方向別最近傍距離・near/mid 密度（16方向×3セグメント）
	{
		TArray<FVector2D> EnemyPositions;
		EnemyPositions.Reserve(Game->Enemies.Num());
		for (const FEnemyState& E : Game->Enemies) EnemyPositions.Add(E.Pos);

		BuildDirectionalDensityFeatures(
			EnemyPositions,
			Game->PlayerPos,
			SurvivorsGameConstants::EnemyDensityDirCount,
			SurvivorsGameConstants::EnemyNearestDistanceMax,
			SurvivorsGameConstants::EnemyDensityNearDistanceMax,
			SurvivorsGameConstants::EnemyDensityMidDistanceMax,
			SurvivorsGameConstants::EnemyDensityNearNormalizeFactor,
			SurvivorsGameConstants::EnemyDensityMidNormalizeFactor,
			Obs);
	}

	// Gem の方向別最近傍距離・near/mid 密度（16方向×3セグメント）
	{
		TArray<FVector2D> GemPositions;
		GemPositions.Reserve(Game->Gems.Num());
		for (const FGemState& G : Game->Gems) GemPositions.Add(G.Pos);

		BuildDirectionalDensityFeatures(
			GemPositions,
			Game->PlayerPos,
			SurvivorsGameConstants::GemDensityDirCount,
			SurvivorsGameConstants::GemNearestDistanceMax,
			SurvivorsGameConstants::GemDensityNearDistanceMax,
			SurvivorsGameConstants::GemDensityMidDistanceMax,
			SurvivorsGameConstants::GemDensityNearNormalizeFactor,
			SurvivorsGameConstants::GemDensityMidNormalizeFactor,
			Obs);
	}

	return Obs;
}
