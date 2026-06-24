#include "Survivors/Weapons/Projectile/SurvivorsWeaponKnifeLogic.h"
#include "Survivors/SurvivorsGameLogic.h"
#include "Survivors/SurvivorsGameConstants.h"

void FSurvivorsWeaponKnifeLogic::OnLevelChanged(FWeaponLevel NewLevel)
{
	CacheParams();
}

void FSurvivorsWeaponKnifeLogic::CacheParams()
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

void FSurvivorsWeaponKnifeLogic::Tick(float Dt)
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

void FSurvivorsWeaponKnifeLogic::StartBurst()
{
	const FPassiveEffects& PE = GetPassiveEffects();
	CooldownTimer = FCooldownSeconds(CachedCooldown * PE.CooldownMult);

	BurstDamage   = CachedDamage * PE.DamageMult;
	BurstSpeed    = CachedSpeed  * PE.SpeedMult;
	BurstRadius   = 6.f * PE.AreaMult;
	BurstLifeTime = 1.5f * PE.DurationMult;
	BurstPierce   = CachedPierce;

	PendingKnifeShots = FMath::Max(1, CachedAmount + static_cast<int32>(PE.ExtraAmount));
	BurstShotCount    = PendingKnifeShots;
	BurstShotIndex    = 0;
	KnifeBurstTimer   = 0.f;

	if (PendingKnifeShots > 0)
	{
		SpawnKnifeShot();
		--PendingKnifeShots;
		KnifeBurstTimer = (PendingKnifeShots > 0) ? 0.10f : 0.f;
	}
}

void FSurvivorsWeaponKnifeLogic::SpawnKnifeShot()
{
	const FVector2D Dir    = LastFacingDir.IsNearlyZero() ? FVector2D(1.f, 0.f) : LastFacingDir.GetSafeNormal();
	const FVector2D Perp   = FVector2D(-Dir.Y, Dir.X);
	const float     Radius = Logic->CurrentConfig.PlayerRadius;
	float           Offset = Logic->RandStream.FRandRange(-Radius, Radius);
	if (BurstShotCount > 1)
	{
		const float T = static_cast<float>(BurstShotIndex) / static_cast<float>(BurstShotCount - 1);
		const float SpreadOffset = FMath::Lerp(-Radius * 0.5f, Radius * 0.5f, T);
		Offset = FMath::Clamp(SpreadOffset + Offset * 0.25f, -Radius, Radius);
	}
	++BurstShotIndex;

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
