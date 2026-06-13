#include "Survivors/Logic/Weapons/Projectile/SurvivorsSantaWaterWeapon.h"

#include "Survivors/Logic/SurvivorsGame.h"
#include "Survivors/Logic/SurvivorsGameConstants.h"
#include "Survivors/Logic/Weapons/SurvivorsWeaponComponent.h"

void USurvivorsSantaWaterWeapon::OnLevelChanged(FWeaponLevel NewLevel)
{
	CacheParams();
}

void USurvivorsSantaWaterWeapon::CacheParams()
{
	const int32 Lv  = FMath::Clamp(Level.Value, 1, SurvivorsGameConstants::MaxWeaponLevel);
	const int32 Idx = Lv - 1;

	if (WeaponType == EWeaponType::LaBorra)
	{
		const SurvivorsGameConstants::FSantaWaterParams& P = SurvivorsGameConstants::LaBorraTable[Idx];
		CachedDamage   = P.Damage;
		CachedCooldown = P.Cooldown;
		CachedRadius   = P.Radius;
		CachedDuration = P.Duration;
		CachedAmount   = P.Amount;
	}
	else
	{
		const SurvivorsGameConstants::FSantaWaterParams& P = SurvivorsGameConstants::SantaWaterTable[Idx];
		CachedDamage   = P.Damage;
		CachedCooldown = P.Cooldown;
		CachedRadius   = P.Radius;
		CachedDuration = P.Duration;
		CachedAmount   = P.Amount;
	}
}

void USurvivorsSantaWaterWeapon::Tick(float Dt)
{
	if (!Game || !WeaponComp) return;

	// 順次落下: 0.3s 間隔でキューから drop を処理（wiki: projectile interval = 0.3s）
	// while + += で overshoot を保持（大 Dt 時のキャッチアップ対応）
	if (PendingDropPositions.Num() > 0)
	{
		DropTimer -= Dt;
		while (DropTimer <= 0.f && PendingDropPositions.Num() > 0)
		{
			SpawnDrop(PendingDropPositions[0]);
			PendingDropPositions.RemoveAt(0);
			DropTimer += 0.30f;
		}
	}

	CooldownTimer.Value = FMath::Max(0.f, CooldownTimer.Value - Dt);
	if (!CooldownTimer.IsReady() || PendingDropPositions.Num() > 0) return;

	StartDropSequence();
}

void USurvivorsSantaWaterWeapon::StartDropSequence()
{
	const FPassiveEffects& PE = GetPassiveEffects();
	CooldownTimer = FCooldownSeconds(CachedCooldown * PE.CooldownMult);

	BurstDamage   = CachedDamage   * PE.DamageMult;
	BurstRadius   = CachedRadius   * PE.AreaMult;
	BurstDuration = CachedDuration * PE.DurationMult;

	const int32 EffAmount = CachedAmount + static_cast<int32>(PE.ExtraAmount);

	PendingDropPositions.Empty();

	if (EffAmount < 4)
	{
		// Amount < 4: 近い敵順に drop（wiki: "first bottle aims at the closest enemy"）
		TArray<int32> EnemyIdx;
		EnemyIdx.Reserve(Game->Enemies.Num());
		for (int32 EIdx = 0; EIdx < Game->Enemies.Num(); ++EIdx)
		{
			if (!Game->Enemies[EIdx].bPendingRemove) EnemyIdx.Add(EIdx);
		}
		EnemyIdx.Sort([&](int32 A, int32 B) {
			return FVector2D::DistSquared(Game->Enemies[A].Pos, Game->PlayerPos)
				 < FVector2D::DistSquared(Game->Enemies[B].Pos, Game->PlayerPos);
		});

		for (int32 i = 0; i < EffAmount; ++i)
		{
			if (EnemyIdx.IsValidIndex(i))
				PendingDropPositions.Add(Game->Enemies[EnemyIdx[i]].Pos);
			else
				PendingDropPositions.Add(Game->PlayerPos);
		}
	}
	else
	{
		// Amount >= 4: プレイヤー周囲の時計回り円形配置（wiki: "clockwise, roughly circular pattern"）
		for (int32 i = 0; i < EffAmount; ++i)
		{
			const float Angle = 2.0f * UE_PI * i / EffAmount;
			PendingDropPositions.Add(
				Game->PlayerPos + FVector2D(FMath::Cos(Angle), FMath::Sin(Angle)) * 80.f);
		}
	}

	// 最初の drop を即時処理
	if (PendingDropPositions.Num() > 0)
	{
		SpawnDrop(PendingDropPositions[0]);
		PendingDropPositions.RemoveAt(0);
		DropTimer = (PendingDropPositions.Num() > 0) ? 0.30f : 0.f;
	}
}

void USurvivorsSantaWaterWeapon::SpawnDrop(FVector2D DropPos)
{
	FGroundZoneState Z;
	Z.Pos           = DropPos;
	Z.Radius        = BurstRadius;
	Z.Damage        = BurstDamage;
	Z.LifeTime      = SurvivorsGameConstants::SantaWaterWarningTime + BurstDuration;
	Z.WarningTime   = SurvivorsGameConstants::SantaWaterWarningTime;
	Z.HitCooldown   = 0.5f;
	Z.WeaponSlotIdx = SlotIdx;
	Z.WeaponType    = WeaponType;
	Z.bIsWarning    = true;
	WeaponComp->SpawnGroundZone(Z);
}
