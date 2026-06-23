#include "Survivors/Logic/Weapons/Projectile/SurvivorsCrossWeaponF.h"
#include "Survivors/Logic/SurvivorsGameLogic.h"
#include "Survivors/Logic/SurvivorsGameConstants.h"

void FSurvivorsCrossWeapon::OnLevelChanged(FWeaponLevel NewLevel)
{
	CacheParams();
}

void FSurvivorsCrossWeapon::CacheParams()
{
	const int32 Lv  = FMath::Clamp(Level.Value, 1, SurvivorsGameConstants::MaxWeaponLevel);
	const int32 Idx = Lv - 1;

	if (WeaponType == EWeaponType::HeavenSword)
	{
		const SurvivorsGameConstants::FCrossParams& P = SurvivorsGameConstants::HeavenSwordTable[Idx];
		CachedDamage            = P.Damage;
		CachedCooldown          = P.Cooldown;
		CachedSpeed             = P.Speed;
		CachedRadius            = P.Radius;
		CachedAmount            = P.Amount;
		CachedKnockbackStrength = P.KnockbackStrength;
	}
	else
	{
		const SurvivorsGameConstants::FCrossParams& P = SurvivorsGameConstants::CrossTable[Idx];
		CachedDamage            = P.Damage;
		CachedCooldown          = P.Cooldown;
		CachedSpeed             = P.Speed;
		CachedRadius            = P.Radius;
		CachedAmount            = P.Amount;
		CachedKnockbackStrength = P.KnockbackStrength;
	}
}

void FSurvivorsCrossWeapon::Tick(float Dt)
{
	if (!Logic) return;

	Logic->UpdateProjectilesBySlot(SlotIdx, Dt, [](FProjectileState& P, float) -> bool
	{
		const float ReverseTime = FMath::Max(P.AngleRad.Value, SurvivorsGameConstants::PhysicsDt);
		if (!P.bHasReversed && P.Age >= ReverseTime)
		{
			P.Vel        = -P.Vel;
			P.bHasReversed = true;
		}
		return true;
	});

	if (PendingCrossShots > 0)
	{
		CrossBurstTimer -= Dt;
		while (PendingCrossShots > 0 && CrossBurstTimer <= 0.f)
		{
			SpawnCrossShot();
			--PendingCrossShots;
			CrossBurstTimer += SurvivorsGameConstants::CrossProjectileInterval;
		}
	}

	CooldownTimer.Value = FMath::Max(0.f, CooldownTimer.Value - Dt);
	if (!CooldownTimer.IsReady() || PendingCrossShots > 0) return;

	StartBurst();
}

void FSurvivorsCrossWeapon::StartBurst()
{
	const FPassiveEffects& PE = GetPassiveEffects();
	CooldownTimer = FCooldownSeconds(CachedCooldown * PE.CooldownMult);

	BurstDamage      = CachedDamage * PE.DamageMult;
	BurstSpeed       = CachedSpeed  * PE.SpeedMult;
	BurstRadius      = CachedRadius * PE.AreaMult;
	BurstLifeTime    = 6.0f * PE.DurationMult;
	BurstReverseTime = SurvivorsGameConstants::CrossReverseDistance / FMath::Max(BurstSpeed, 1.f);
	BurstKnockback   = CachedKnockbackStrength;

	PendingCrossShots = FMath::Max(1, CachedAmount + static_cast<int32>(PE.ExtraAmount));
	CrossBurstTimer   = 0.f;

	if (PendingCrossShots > 0)
	{
		SpawnCrossShot();
		--PendingCrossShots;
		CrossBurstTimer = (PendingCrossShots > 0) ? SurvivorsGameConstants::CrossProjectileInterval : 0.f;
	}
}

void FSurvivorsCrossWeapon::SpawnCrossShot()
{
	FVector2D Dir = FVector2D::ZeroVector;
	float MinDistSq = MAX_FLT;
	for (const FEnemyState& E : Logic->Enemies)
	{
		if (E.bPendingRemove) continue;
		if (!Logic->IsOnScreen(E.Pos)) continue;
		const float Dsq = (E.Pos - Logic->PlayerPos).SizeSquared();
		if (Dsq < MinDistSq) { MinDistSq = Dsq; Dir = (E.Pos - Logic->PlayerPos).GetSafeNormal(); }
	}
	if (Dir.IsNearlyZero())
	{
		const float Angle = Logic->RandStream.FRand() * TWO_PI;
		Dir = FVector2D(FMath::Cos(Angle), FMath::Sin(Angle));
	}

	FProjectileState P;
	P.Pos               = Logic->PlayerPos;
	P.Vel               = Dir * BurstSpeed;
	P.Radius            = FSimRadius(BurstRadius);
	P.Damage            = FDamage(BurstDamage);
	P.WeaponType        = WeaponType;
	P.WeaponSlotIdx     = SlotIdx;
	P.LifeTime          = FProjectileLifeTime(BurstLifeTime);
	P.Age               = 0.f;
	P.bPiercing         = true;
	P.bHasReversed      = false;
	P.AngleRad          = FOrbitAngleRad(BurstReverseTime);
	P.KnockbackStrength = BurstKnockback;
	Logic->SpawnProjectile(P);
}
