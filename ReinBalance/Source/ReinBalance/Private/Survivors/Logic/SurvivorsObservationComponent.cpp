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
		{ TEXT("player_pos"),    2 },
		{ TEXT("player_vel"),    2 },
		{ TEXT("wall_rays"),     8 },
		{ TEXT("player_hp"),     1 },
		{ TEXT("weapon_slots"),  SurvivorsGameConstants::MaxWeaponSlots * 2 },
		{ TEXT("enemy_count"),   1 },
		{ TEXT("elapsed_time"),  1 },
		{ TEXT("xp_progress"),   1 },
		{ TEXT("player_level"),  1 },
		{ TEXT("gem_rel_pos"),   SurvivorsGameConstants::NumGemObs * 2 },
		{ TEXT("enemy_rel_pos"), SurvivorsGameConstants::MaxEnemyObs * 2 },
		{ TEXT("enemy_vel"),     SurvivorsGameConstants::MaxEnemyObs * 2 },
		{ TEXT("enemy_type"),    SurvivorsGameConstants::MaxEnemyObs },
		{ TEXT("enemy_hp"),      SurvivorsGameConstants::MaxEnemyObs },
	};
}

FString USurvivorsObservationComponent::GetObsSchemaHash() const
{
	FString Schema = FString::Printf(
		TEXT("SurvivorsGame,NumGemObs=%d,MaxEnemyObs=%d,MaxWeaponSlots=%d,enemy_vel_normed"),
		SurvivorsGameConstants::NumGemObs,
		SurvivorsGameConstants::MaxEnemyObs,
		SurvivorsGameConstants::MaxWeaponSlots);
	return FMD5::HashAnsiString(*Schema);
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

	return Obs;
}
