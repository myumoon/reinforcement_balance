#include "Survivors/Logic/Weapons/Projectile/SurvivorsWeaponSantaWaterLogic.h"
#include "Survivors/Logic/SurvivorsGameLogic.h"
#include "Survivors/Logic/SurvivorsGameConstants.h"

void FSurvivorsWeaponSantaWaterLogic::OnLevelChanged(FWeaponLevel NewLevel)
{
	CacheParams();
}

void FSurvivorsWeaponSantaWaterLogic::CacheParams()
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

void FSurvivorsWeaponSantaWaterLogic::Tick(float Dt)
{
	if (!Logic) return;

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

void FSurvivorsWeaponSantaWaterLogic::StartDropSequence()
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
		FVector2D NearestEnemyPos = Logic->PlayerPos;
		float MinDistSq = MAX_FLT;
		for (const FEnemyState& E : Logic->Enemies)
		{
			if (E.bPendingRemove) continue;
			const float Dsq = FVector2D::DistSquared(E.Pos, Logic->PlayerPos);
			if (Dsq < MinDistSq) { MinDistSq = Dsq; NearestEnemyPos = E.Pos; }
		}
		PendingDropPositions.Add(NearestEnemyPos);

		for (int32 i = 1; i < EffAmount; ++i)
		{
			const float Angle = Logic->RandStream.FRand() * TWO_PI;
			const float Dist  = Logic->RandStream.FRandRange(0.f, SurvivorsGameConstants::SantaWaterRandomDropRadius);
			PendingDropPositions.Add(Logic->PlayerPos + FVector2D(FMath::Cos(Angle), FMath::Sin(Angle)) * Dist);
		}
	}
	else
	{
		const float StartAngle = Logic->RandStream.FRand() * TWO_PI;
		for (int32 i = 0; i < EffAmount; ++i)
		{
			const float Angle = StartAngle + SurvivorsGameConstants::SantaWaterCircleDropStep * i;
			PendingDropPositions.Add(
				Logic->PlayerPos + FVector2D(FMath::Cos(Angle), FMath::Sin(Angle)) * SurvivorsGameConstants::SantaWaterCircleRadius);
		}
	}

	if (PendingDropPositions.Num() > 0)
	{
		SpawnDrop(PendingDropPositions[0]);
		PendingDropPositions.RemoveAt(0);
		DropTimer = (PendingDropPositions.Num() > 0) ? 0.30f : 0.f;
	}
}

void FSurvivorsWeaponSantaWaterLogic::SpawnDrop(FVector2D DropPos)
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
	Logic->SpawnGroundZone(Z);
}
