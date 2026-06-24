#include "Survivors/Game/Weapons/Projectile/SurvivorsWeaponMagicWandLogic.h"
#include "Survivors/Game/SurvivorsGameLogic.h"
#include "Survivors/Game/SurvivorsGameConstants.h"

void FSurvivorsWeaponMagicWandLogic::OnLevelChanged(FWeaponLevel NewLevel)
{
	CacheParams();
}

void FSurvivorsWeaponMagicWandLogic::CacheParams()
{
	const int32 Lv  = FMath::Clamp(Level.Value, 1, SurvivorsGameConstants::MaxWeaponLevel);
	const int32 Idx = Lv - 1;

	if (WeaponType == EWeaponType::HolyWand)
	{
		const SurvivorsGameConstants::FMagicWandParams& P = SurvivorsGameConstants::HolyWandTable[Idx];
		CachedDamage   = P.Damage;
		CachedCooldown = P.Cooldown;
		CachedSpeed    = P.Speed;
		CachedAmount   = P.Amount;
		CachedPierce   = P.Pierce;
	}
	else
	{
		const SurvivorsGameConstants::FMagicWandParams& P = SurvivorsGameConstants::MagicWandTable[Idx];
		CachedDamage   = P.Damage;
		CachedCooldown = P.Cooldown;
		CachedSpeed    = P.Speed;
		CachedAmount   = P.Amount;
		CachedPierce   = P.Pierce;
	}
}

void FSurvivorsWeaponMagicWandLogic::Tick(float Dt)
{
	if (!Logic) return;

	if (PendingWandShots > 0)
	{
		WandBurstTimer -= Dt;
		while (PendingWandShots > 0 && WandBurstTimer <= 0.f)
		{
			SpawnWandShot();
			--PendingWandShots;
			if (PendingWandShots > 0) WandBurstTimer += 0.10f;
		}
	}

	CooldownTimer.Value = FMath::Max(0.f, CooldownTimer.Value - Dt);
	if (!CooldownTimer.IsReady() || PendingWandShots > 0) return;

	StartBurst();
}

void FSurvivorsWeaponMagicWandLogic::StartBurst()
{
	const FPassiveEffects& PE = GetPassiveEffects();
	CooldownTimer = FCooldownSeconds(CachedCooldown * PE.CooldownMult);

	BurstDamage   = CachedDamage * PE.DamageMult;
	BurstSpeed    = CachedSpeed  * PE.SpeedMult;
	BurstRadius   = 8.f * PE.AreaMult;
	BurstLifeTime = 6.0f * PE.DurationMult;
	BurstPierce   = CachedPierce;

	bool bHasOnScreenEnemy = false;
	for (const FEnemyState& E : Logic->Enemies)
	{
		if (!E.bPendingRemove && Logic->IsOnScreen(E.Pos)) { bHasOnScreenEnemy = true; break; }
	}
	if (!bHasOnScreenEnemy)
	{
		const float Angle = Logic->RandStream.FRand() * TWO_PI;
		BurstNoTargetDir = FVector2D(FMath::Cos(Angle), FMath::Sin(Angle));
	}

	PendingWandShots = CachedAmount + static_cast<int32>(PE.ExtraAmount);
	WandBurstTimer   = 0.f;

	if (PendingWandShots > 0)
	{
		SpawnWandShot();
		--PendingWandShots;
		WandBurstTimer = (PendingWandShots > 0) ? 0.10f : 0.f;
	}
}

void FSurvivorsWeaponMagicWandLogic::SpawnWandShot()
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
	if (Dir.IsNearlyZero()) Dir = BurstNoTargetDir;

	FProjectileState P;
	P.Pos               = Logic->PlayerPos;
	P.Vel               = Dir * BurstSpeed;
	P.Radius            = FSimRadius(BurstRadius);
	P.Damage            = FDamage(BurstDamage);
	P.WeaponType        = WeaponType;
	P.WeaponSlotIdx     = SlotIdx;
	P.LifeTime          = FProjectileLifeTime(BurstLifeTime);
	P.bPiercing         = false;
	P.MaxPierceCount    = BurstPierce;
	P.KnockbackStrength = SurvivorsGameConstants::KnockbackSim_1;
	Logic->SpawnProjectile(P);
}
