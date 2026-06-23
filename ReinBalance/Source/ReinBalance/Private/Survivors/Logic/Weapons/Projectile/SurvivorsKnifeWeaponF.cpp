#include "Survivors/Logic/Weapons/Projectile/SurvivorsKnifeWeaponF.h"
#include "Survivors/Logic/SurvivorsGameLogic.h"
#include "Survivors/Logic/SurvivorsGameConstants.h"

void FSurvivorsKnifeWeapon::OnLevelChanged(FWeaponLevel NewLevel)
{
	CacheParams();
}

void FSurvivorsKnifeWeapon::CacheParams()
{
	const int32 Lv  = FMath::Clamp(Level.Value, 1, SurvivorsGameConstants::MaxWeaponLevel);
	const int32 Idx = Lv - 1;

	if (WeaponType == EWeaponType::ThousandEdge)
	{
		const SurvivorsGameConstants::FKnifeParams& P = SurvivorsGameConstants::ThousandEdgeTable[Idx];
		CachedDamage   = P.Damage;
		CachedCooldown = P.Cooldown;
		CachedSpeed    = P.Speed;
		CachedAmount   = P.Amount;
		CachedPierce   = P.Pierce;
	}
	else
	{
		const SurvivorsGameConstants::FKnifeParams& P = SurvivorsGameConstants::KnifeTable[Idx];
		CachedDamage   = P.Damage;
		CachedCooldown = P.Cooldown;
		CachedSpeed    = P.Speed;
		CachedAmount   = P.Amount;
		CachedPierce   = P.Pierce;
	}
}

void FSurvivorsKnifeWeapon::Tick(float Dt)
{
	if (!Logic) return;

	if (!Logic->PlayerVel.IsNearlyZero())
	{
		LastFacingDir = Logic->PlayerVel.GetSafeNormal();
	}

	CooldownTimer.Value = FMath::Max(0.f, CooldownTimer.Value - Dt);

	if (PendingKnifeShots > 0)
	{
		KnifeBurstTimer -= Dt;
		while (PendingKnifeShots > 0 && KnifeBurstTimer <= 0.f)
		{
			SpawnKnifeShot();
			--PendingKnifeShots;
			if (PendingKnifeShots > 0) KnifeBurstTimer += 0.10f;
		}
	}

	if (!CooldownTimer.IsReady() || PendingKnifeShots > 0) return;

	StartBurst();
}

void FSurvivorsKnifeWeapon::StartBurst()
{
	const FPassiveEffects& PE = GetPassiveEffects();
	CooldownTimer = FCooldownSeconds(CachedCooldown * PE.CooldownMult);

	BurstDamage   = CachedDamage * PE.DamageMult;
	BurstSpeed    = CachedSpeed  * PE.SpeedMult;
	BurstRadius   = 6.f * PE.AreaMult;
	BurstLifeTime = 1.5f * PE.DurationMult;
	BurstPierce   = CachedPierce;

	PendingKnifeShots = FMath::Max(1, CachedAmount + static_cast<int32>(PE.ExtraAmount));
	KnifeBurstTimer   = 0.f;

	if (PendingKnifeShots > 0)
	{
		SpawnKnifeShot();
		--PendingKnifeShots;
		KnifeBurstTimer = (PendingKnifeShots > 0) ? 0.10f : 0.f;
	}
}

void FSurvivorsKnifeWeapon::SpawnKnifeShot()
{
	const FVector2D Dir    = LastFacingDir.IsNearlyZero() ? FVector2D(1.f, 0.f) : LastFacingDir.GetSafeNormal();
	const FVector2D Perp   = FVector2D(-Dir.Y, Dir.X);
	const float     Offset = Logic->RandStream.FRandRange(-Logic->CurrentConfig.PlayerRadius, Logic->CurrentConfig.PlayerRadius);

	FProjectileState P;
	P.Pos               = Logic->PlayerPos + Perp * Offset;
	P.Vel               = Dir * BurstSpeed;
	P.Radius            = FSimRadius(BurstRadius);
	P.Damage            = FDamage(BurstDamage);
	P.WeaponType        = WeaponType;
	P.WeaponSlotIdx     = SlotIdx;
	P.LifeTime          = FProjectileLifeTime(BurstLifeTime);
	P.bPiercing         = false;
	P.MaxPierceCount    = BurstPierce;
	P.KnockbackStrength = SurvivorsGameConstants::KnockbackSim_Half;
	Logic->SpawnProjectile(P);
}
