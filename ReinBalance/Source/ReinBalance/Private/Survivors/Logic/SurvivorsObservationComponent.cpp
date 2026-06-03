#include "Survivors/Logic/SurvivorsObservationComponent.h"

#include "Survivors/Logic/SurvivorsCollisionComponent.h"
#include "Survivors/Logic/SurvivorsGame.h"
#include "Survivors/Logic/SurvivorsPlayerComponent.h"
#include "Survivors/Logic/Weapons/SurvivorsWeaponComponent.h"
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
	using namespace SurvivorsGameConstants;
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
		{ TEXT("weapon_slots"),               MaxWeaponSlots * 3 },   // (type_norm, level_norm, cooldown_norm) × 6
		{ TEXT("passive_slots"),              MaxPassiveSlots * 2 },  // (type_norm, level_norm) × 6
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
		{ TEXT("projectiles"),                MaxProjectileObs * 5 },  // (dx,dy,radius,vx,vy) × 32
		{ TEXT("floor_pickups"),              MaxFloorPickupObs * 3 }, // (dx,dy,type_norm) × 8
		{ TEXT("special_pickups"),            MaxSpecialPickupObs * 3 }, // (dx,dy,type_norm) × 3
		{ TEXT("destructibles"),              MaxDestructibleObs * 2 }, // (dx,dy) × 10
	};
}

FString USurvivorsObservationComponent::GetObsSchemaHash() const
{
	using namespace SurvivorsGameConstants;
	FString Schema = FString::Printf(
		TEXT("SurvivorsGame_v708"
		     ",MaxEnemyObs=%d,MaxWeaponSlots=%d,MaxPassiveSlots=%d"
		     ",MaxProjectileObs=%d,MaxRedGemObs=%d,MaxGreenGemObs=%d,MaxBlueGemObs=%d"
		     ",MaxFloorPickupObs=%d,MaxSpecialPickupObs=%d,MaxDestructibleObs=%d"
		     ",MaxWeaponTypeCountReserved=%d,MaxPassiveTypeCountReserved=%d"
		     ",EnemyDensityDirCount=%d,GemDensityDirCount=%d"
		     ",EnemyNearestDistanceMax=%.0f,GemNearestDistanceMax=%.0f"),
		MaxEnemyObs, MaxWeaponSlots, MaxPassiveSlots,
		MaxProjectileObs, MaxRedGemObs, MaxGreenGemObs, MaxBlueGemObs,
		MaxFloorPickupObs, MaxSpecialPickupObs, MaxDestructibleObs,
		MaxWeaponTypeCountReserved, MaxPassiveTypeCountReserved,
		EnemyDensityDirCount, GemDensityDirCount,
		EnemyNearestDistanceMax, GemNearestDistanceMax);
	return FMD5::HashAnsiString(*Schema);
}

// 対象位置リストからプレイヤー中心の方向別特徴を計算して Obs に追記するヘルパー。
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
		if (Dist <= KINDA_SMALL_NUMBER) continue;

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

	using namespace SurvivorsGameConstants;
	const float HN         = Game->FieldHalfSize;
	const float DN         = Game->FieldHalfSize * 2.f;
	const float MaxRayDist = Game->FieldHalfSize * 2.f;

	// ---- player_pos (2) ----
	Obs.Add(Game->PlayerPos.X / HN);
	Obs.Add(Game->PlayerPos.Y / HN);

	// ---- player_vel (2) ----
	Obs.Add(Game->MoveSpeed > 0.f ? Game->PlayerVel.X / Game->MoveSpeed : 0.f);
	Obs.Add(Game->MoveSpeed > 0.f ? Game->PlayerVel.Y / Game->MoveSpeed : 0.f);

	// ---- wall_rays (8) ----
	for (int32 r = 0; r < 8; ++r)
	{
		const float Dist = Game->CollisionComponent->CastRayToObstacles(Game->PlayerPos, RayDirs[r]);
		Obs.Add(FMath::Clamp(Dist / MaxRayDist, 0.f, 1.f));
	}

	// ---- player_hp (1) ----
	Obs.Add(FMath::Clamp(Game->PlayerHP / Game->MaxPlayerHP, 0.f, 1.f));

	// ---- shield_active (1) ----
	Obs.Add(Game->bShieldActive ? 1.f : 0.f);

	// ---- shield_timer_norm (1) ----
	Obs.Add(FMath::Clamp(Game->PlayerShieldTimer / MaxShieldDuration, 0.f, 1.f));

	// ---- revival_remaining_norm (1) ----
	{
		const float MaxRev = Game->MaxRevivalCount > 0 ? static_cast<float>(Game->MaxRevivalCount) : 1.f;
		const float Remaining = static_cast<float>(Game->MaxRevivalCount - Game->UsedRevivalCount);
		Obs.Add(FMath::Clamp(Remaining / MaxRev, 0.f, 1.f));
	}

	// ---- armor_flat_norm (1) ----
	Obs.Add(FMath::Clamp(Game->CachedPassiveEffects.ArmorFlat / MaxArmorFlat, 0.f, 1.f));

	// ---- regen_per_sec_norm (1) ----
	Obs.Add(FMath::Clamp(Game->CachedPassiveEffects.RegenPerSec / MaxRegenPerSec, 0.f, 1.f));

	// ---- passive_effect_summary (5) ----
	Obs.Add(FMath::Clamp(Game->CachedPassiveEffects.DamageMult    - 1.f, 0.f, 1.f));
	Obs.Add(FMath::Clamp(1.f - Game->CachedPassiveEffects.CooldownMult, 0.f, 1.f));
	Obs.Add(FMath::Clamp(Game->CachedPassiveEffects.AreaMult      - 1.f, 0.f, 1.f));
	Obs.Add(FMath::Clamp(Game->CachedPassiveEffects.MoveSpeedMult - 1.f, 0.f, 1.f));
	Obs.Add(FMath::Clamp(Game->CachedPassiveEffects.PickupRadiusMult - 1.f, 0.f, 1.f));

	// ---- weapon_slots (MaxWeaponSlots * 3 = 18) ----
	for (int32 s = 0; s < MaxWeaponSlots; ++s)
	{
		const FWeaponSlot& Slot = Game->WeaponSlots[s];
		// type_norm: id / MaxWeaponTypeCountReserved(64)
		Obs.Add(static_cast<float>(static_cast<uint8>(Slot.Type)) / static_cast<float>(MaxWeaponTypeCountReserved));
		// level_norm: level / MaxWeaponLevel
		Obs.Add(static_cast<float>(Slot.Level.Value) / static_cast<float>(MaxWeaponLevel));
		// cooldown_norm: 武器インスタンスから取得
		if (Game->WeaponComponent)
		{
			const USurvivorsWeaponBase* WI = Game->WeaponComponent->GetWeaponInstance(s);
			if (WI && Slot.Type != EWeaponType::None)
			{
				// クールダウン残秒 / 武器の最大クールダウン。Garlic の場合は HitInterval
				// 簡易実装: CooldownRemaining を HitInterval(1.3s) で割って 0〜1 に
				const float MaxCD = 2.f;  // 全武器の最大クールダウン上限として暫定2秒
				Obs.Add(FMath::Clamp(WI->GetCooldownRemaining().Value / MaxCD, 0.f, 1.f));
			}
			else
			{
				Obs.Add(0.f);
			}
		}
		else
		{
			Obs.Add(0.f);
		}
	}

	// ---- passive_slots (MaxPassiveSlots * 2 = 12) ----
	for (int32 s = 0; s < MaxPassiveSlots; ++s)
	{
		const FPassiveSlot& Slot = Game->PassiveSlots[s];
		// type_norm: id / MaxPassiveTypeCountReserved(32)
		Obs.Add(static_cast<float>(static_cast<uint8>(Slot.Type)) / static_cast<float>(MaxPassiveTypeCountReserved));
		// level_norm: level / PassiveMaxLevel[type] (アイテム別最大レベル)
		float LevelNorm = 0.f;
		if (Slot.Type != EPassiveItemType::None)
		{
			const int32 TypeIdx = static_cast<int32>(static_cast<uint8>(Slot.Type));
			if (TypeIdx >= 0 && TypeIdx < 18 && PassiveMaxLevel[TypeIdx] > 0)
			{
				LevelNorm = FMath::Clamp(static_cast<float>(Slot.Level) / static_cast<float>(PassiveMaxLevel[TypeIdx]), 0.f, 1.f);
			}
		}
		Obs.Add(LevelNorm);
	}

	// ---- enemy_count (1) ----
	Obs.Add(static_cast<float>(Game->Enemies.Num()) / static_cast<float>(MaxEnemyObs));

	// ---- elapsed_time (1) ----
	Obs.Add(FMath::Clamp(Game->ElapsedTime / MaxGameTime, 0.f, 1.f));

	// ---- xp_progress (1) ----
	{
		const float CurCum  = Game->PlayerComponent->CumulativeXPForLevel(Game->PlayerLevel);
		const float NextCum = Game->PlayerComponent->CumulativeXPForLevel(Game->PlayerLevel + 1);
		const float Range   = NextCum - CurCum;
		Obs.Add(Range > 0.f ? FMath::Clamp((Game->PlayerXP - CurCum) / Range, 0.f, 1.f) : 0.f);
	}

	// ---- player_level (1) ----
	Obs.Add(FMath::Clamp(static_cast<float>(Game->PlayerLevel) / static_cast<float>(MaxPlayerLevel), 0.f, 1.f));

	// ---- stage_id_norm (1) ----
	Obs.Add(0.f);  // Mad Forest = 0、将来ステージ追加時に拡張

	// ---- ジェムをタイプ別に分類して距離昇順でソート ----
	TArray<FVector2D> RedPos, GreenPos, BluePos;
	for (const FGemState& G : Game->Gems)
	{
		switch (G.Type)
		{
			case EGemType::Red:   RedPos.Add(G.Pos);   break;
			case EGemType::Green: GreenPos.Add(G.Pos); break;
			default:              BluePos.Add(G.Pos);  break;
		}
	}

	auto AddGemObs = [&](const TArray<FVector2D>& Positions, int32 MaxCount)
	{
		TArray<int32> Idx;
		Idx.Reserve(Positions.Num());
		for (int32 i = 0; i < Positions.Num(); ++i) Idx.Add(i);
		Idx.Sort([&](int32 A, int32 B)
		{
			return FVector2D::DistSquared(Positions[A], Game->PlayerPos)
				 < FVector2D::DistSquared(Positions[B], Game->PlayerPos);
		});
		for (int32 s = 0; s < MaxCount; ++s)
		{
			if (s < Idx.Num())
			{
				Obs.Add((Positions[Idx[s]].X - Game->PlayerPos.X) / DN);
				Obs.Add((Positions[Idx[s]].Y - Game->PlayerPos.Y) / DN);
			}
			else { Obs.Add(0.f); Obs.Add(0.f); }
		}
	};

	// ---- red_gem_rel_pos (MaxRedGemObs * 2 = 20) ----
	AddGemObs(RedPos, MaxRedGemObs);

	// ---- green_gem_rel_pos (MaxGreenGemObs * 2 = 24) ----
	AddGemObs(GreenPos, MaxGreenGemObs);

	// ---- blue_gem_rel_pos (MaxBlueGemObs * 2 = 24) ----
	AddGemObs(BluePos, MaxBlueGemObs);

	// ---- gem_pickup_radius (1) ----
	// PickupRadius / 最大値（パッシブで増加する場合の上限を想定）
	{
		const float MaxPickupRadius = 150.f;  // 暫定: Attractorb 5Lv で約 2.5x の 75u 想定
		Obs.Add(FMath::Clamp(Game->GemPickupRadius / MaxPickupRadius, 0.f, 1.f));
	}

	// ---- 敵ソート ----
	TArray<int32> EnemyIdx;
	EnemyIdx.Reserve(Game->Enemies.Num());
	for (int32 i = 0; i < Game->Enemies.Num(); ++i) EnemyIdx.Add(i);
	EnemyIdx.Sort([this](int32 A, int32 B)
	{
		return FVector2D::DistSquared(Game->Enemies[A].Pos, Game->PlayerPos)
			 < FVector2D::DistSquared(Game->Enemies[B].Pos, Game->PlayerPos);
	});

	// ---- enemy_rel_pos (MaxEnemyObs * 2 = 64) ----
	for (int32 Slot = 0; Slot < MaxEnemyObs; ++Slot)
	{
		if (Slot < EnemyIdx.Num())
		{
			const FEnemyState& E = Game->Enemies[EnemyIdx[Slot]];
			Obs.Add((E.Pos.X - Game->PlayerPos.X) / DN);
			Obs.Add((E.Pos.Y - Game->PlayerPos.Y) / DN);
		}
		else { Obs.Add(0.f); Obs.Add(0.f); }
	}

	// ---- enemy_vel (MaxEnemyObs * 2 = 64) ----
	const float VN = Game->MoveSpeed > 0.f ? Game->MoveSpeed : 1.f;
	for (int32 Slot = 0; Slot < MaxEnemyObs; ++Slot)
	{
		if (Slot < EnemyIdx.Num())
		{
			const FEnemyState& E = Game->Enemies[EnemyIdx[Slot]];
			Obs.Add(E.Vel.X / VN);
			Obs.Add(E.Vel.Y / VN);
		}
		else { Obs.Add(0.f); Obs.Add(0.f); }
	}

	// ---- enemy_type (MaxEnemyObs = 32) ----
	{
		const float TypeNorm = Game->EnemyTypeTable.Num() > 1
			? static_cast<float>(Game->EnemyTypeTable.Num() - 1) : 1.f;
		for (int32 Slot = 0; Slot < MaxEnemyObs; ++Slot)
		{
			if (Slot < EnemyIdx.Num())
				Obs.Add(static_cast<float>(Game->Enemies[EnemyIdx[Slot]].TypeId) / TypeNorm);
			else
				Obs.Add(0.f);
		}
	}

	// ---- enemy_hp (MaxEnemyObs = 32) ----
	for (int32 Slot = 0; Slot < MaxEnemyObs; ++Slot)
	{
		if (Slot < EnemyIdx.Num())
		{
			const FEnemyState& E = Game->Enemies[EnemyIdx[Slot]];
			Obs.Add(FMath::Clamp(E.HP / E.MaxHP, 0.f, 1.f));
		}
		else { Obs.Add(0.f); }
	}

	// ---- enemy_frozen (MaxEnemyObs = 32) ---- 将来用、現在は常に 0
	for (int32 Slot = 0; Slot < MaxEnemyObs; ++Slot)
	{
		Obs.Add(Slot < EnemyIdx.Num() ? (Game->Enemies[EnemyIdx[Slot]].bFrozen ? 1.f : 0.f) : 0.f);
	}

	// ---- 敵方向特徴（EnemyDensityDirCount × 3 = 48） ----
	{
		TArray<FVector2D> EnemyPositions;
		EnemyPositions.Reserve(Game->Enemies.Num());
		for (const FEnemyState& E : Game->Enemies) EnemyPositions.Add(E.Pos);

		BuildDirectionalDensityFeatures(
			EnemyPositions, Game->PlayerPos,
			EnemyDensityDirCount,
			EnemyNearestDistanceMax, EnemyDensityNearDistanceMax, EnemyDensityMidDistanceMax,
			EnemyDensityNearNormalizeFactor, EnemyDensityMidNormalizeFactor, Obs);
	}

	// ---- gem_density_all_16dir (GemDensityDirCount × 3 = 48) ---- 全ジェム
	{
		TArray<FVector2D> AllGemPositions;
		AllGemPositions.Reserve(Game->Gems.Num());
		for (const FGemState& G : Game->Gems) AllGemPositions.Add(G.Pos);

		BuildDirectionalDensityFeatures(
			AllGemPositions, Game->PlayerPos,
			GemDensityDirCount,
			GemNearestDistanceMax, GemDensityNearDistanceMax, GemDensityMidDistanceMax,
			GemDensityNearNormalizeFactor, GemDensityMidNormalizeFactor, Obs);
	}

	// ---- red_green_gem_density_16dir (GemDensityDirCount × 3 = 48) ---- Red + Green のみ
	{
		TArray<FVector2D> RGPositions;
		for (const FGemState& G : Game->Gems)
			if (G.Type != EGemType::Blue) RGPositions.Add(G.Pos);

		BuildDirectionalDensityFeatures(
			RGPositions, Game->PlayerPos,
			GemDensityDirCount,
			GemNearestDistanceMax, GemDensityNearDistanceMax, GemDensityMidDistanceMax,
			GemDensityNearNormalizeFactor, GemDensityMidNormalizeFactor, Obs);
	}

	// ---- projectiles (MaxProjectileObs * 5 = 160) ----
	if (Game->WeaponComponent)
	{
		TArray<FProjectileState> ProjView = Game->WeaponComponent->GetProjectileObsView();

		// 武器 Level 高い順 → 距離近い順でソート
		ProjView.StableSort([&](const FProjectileState& A, const FProjectileState& B)
		{
			const int32 LvA = (A.WeaponSlotIdx >= 0 && A.WeaponSlotIdx < MaxWeaponSlots)
				? Game->WeaponSlots[A.WeaponSlotIdx].Level.Value : 0;
			const int32 LvB = (B.WeaponSlotIdx >= 0 && B.WeaponSlotIdx < MaxWeaponSlots)
				? Game->WeaponSlots[B.WeaponSlotIdx].Level.Value : 0;
			if (LvA != LvB) return LvA > LvB;
			return FVector2D::DistSquared(A.Pos, Game->PlayerPos)
				 < FVector2D::DistSquared(B.Pos, Game->PlayerPos);
		});

		const float VNorm = Game->MoveSpeed > 0.f ? Game->MoveSpeed : 1.f;
		for (int32 p = 0; p < MaxProjectileObs; ++p)
		{
			if (p < ProjView.Num())
			{
				const FProjectileState& P = ProjView[p];
				Obs.Add((P.Pos.X - Game->PlayerPos.X) / DN);
				Obs.Add((P.Pos.Y - Game->PlayerPos.Y) / DN);
				Obs.Add(FMath::Clamp(P.Radius.Value / MaxProjectileRadius, 0.f, 1.f));
				Obs.Add(P.Vel.X / VNorm);
				Obs.Add(P.Vel.Y / VNorm);
			}
			else { Obs.Add(0.f); Obs.Add(0.f); Obs.Add(0.f); Obs.Add(0.f); Obs.Add(0.f); }
		}
	}
	else
	{
		// WeaponComponent なし時は 0 パディング
		for (int32 p = 0; p < MaxProjectileObs * 5; ++p) Obs.Add(0.f);
	}

	// ---- floor_pickups (MaxFloorPickupObs * 3 = 24) ----
	{
		TArray<int32> FPIdx;
		for (int32 i = 0; i < Game->FloorPickups.Num(); ++i)
			if (Game->FloorPickups[i].bActive) FPIdx.Add(i);
		FPIdx.Sort([&](int32 A, int32 B)
		{
			return FVector2D::DistSquared(Game->FloorPickups[A].Pos, Game->PlayerPos)
				 < FVector2D::DistSquared(Game->FloorPickups[B].Pos, Game->PlayerPos);
		});
		for (int32 s = 0; s < MaxFloorPickupObs; ++s)
		{
			if (s < FPIdx.Num())
			{
				const FFloorPickupState& FP = Game->FloorPickups[FPIdx[s]];
				Obs.Add((FP.Pos.X - Game->PlayerPos.X) / DN);
				Obs.Add((FP.Pos.Y - Game->PlayerPos.Y) / DN);
				Obs.Add(static_cast<float>(static_cast<uint8>(FP.Type)) / 2.f);  // 0=FloorChicken, 1=LittleHeart
			}
			else { Obs.Add(0.f); Obs.Add(0.f); Obs.Add(0.f); }
		}
	}

	// ---- special_pickups (MaxSpecialPickupObs * 3 = 9) ----
	{
		TArray<int32> SPIdx;
		for (int32 i = 0; i < Game->SpecialPickups.Num(); ++i)
			if (Game->SpecialPickups[i].bActive) SPIdx.Add(i);
		SPIdx.Sort([&](int32 A, int32 B)
		{
			return FVector2D::DistSquared(Game->SpecialPickups[A].Pos, Game->PlayerPos)
				 < FVector2D::DistSquared(Game->SpecialPickups[B].Pos, Game->PlayerPos);
		});
		for (int32 s = 0; s < MaxSpecialPickupObs; ++s)
		{
			if (s < SPIdx.Num())
			{
				const FSpecialPickupState& SP = Game->SpecialPickups[SPIdx[s]];
				Obs.Add((SP.Pos.X - Game->PlayerPos.X) / DN);
				Obs.Add((SP.Pos.Y - Game->PlayerPos.Y) / DN);
				Obs.Add(static_cast<float>(static_cast<uint8>(SP.Type)) / 3.f);  // 0=Rosary, 1=Orologion, 2=Vacuum
			}
			else { Obs.Add(0.f); Obs.Add(0.f); Obs.Add(0.f); }
		}
	}

	// ---- destructibles (MaxDestructibleObs * 2 = 20) ----
	{
		TArray<int32> DIdx;
		for (int32 i = 0; i < Game->Destructibles.Num(); ++i)
			if (Game->Destructibles[i].bActive) DIdx.Add(i);
		DIdx.Sort([&](int32 A, int32 B)
		{
			return FVector2D::DistSquared(Game->Destructibles[A].Pos, Game->PlayerPos)
				 < FVector2D::DistSquared(Game->Destructibles[B].Pos, Game->PlayerPos);
		});
		for (int32 s = 0; s < MaxDestructibleObs; ++s)
		{
			if (s < DIdx.Num())
			{
				const FDestructibleState& D = Game->Destructibles[DIdx[s]];
				Obs.Add((D.Pos.X - Game->PlayerPos.X) / DN);
				Obs.Add((D.Pos.Y - Game->PlayerPos.Y) / DN);
			}
			else { Obs.Add(0.f); Obs.Add(0.f); }
		}
	}

	return Obs;
}
