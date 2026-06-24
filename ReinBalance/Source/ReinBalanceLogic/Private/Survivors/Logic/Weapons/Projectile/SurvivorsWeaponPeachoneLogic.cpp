#include "Survivors/Logic/Weapons/Projectile/SurvivorsWeaponPeachoneLogic.h"
#include "Survivors/Logic/SurvivorsGameLogic.h"
#include "Survivors/Logic/SurvivorsGameConstants.h"

void FSurvivorsWeaponPeachoneLogic::OnLevelChanged(FWeaponLevel NewLevel)
{
	CacheParams();
}

void FSurvivorsWeaponPeachoneLogic::CacheParams()
{
	const int32 Lv  = FMath::Clamp(Level.Value, 1, SurvivorsGameConstants::MaxWeaponLevel);
	const int32 Idx = Lv - 1;

	const SurvivorsGameConstants::FPeachoneParams& P = SurvivorsGameConstants::PeachoneTable[Idx];
	CachedDamage           = P.Damage;
	CachedCooldown         = P.Cooldown;
	CachedOrbitRadius      = P.OrbitRadius;
	CachedOrbitRotSpeed    = P.OrbitRotSpeed;
	CachedTargetZoneRadius = P.TargetZoneRadius;
	CachedImpactRadius     = P.ImpactRadius;
	CachedAmount           = P.Amount;
}

void FSurvivorsWeaponPeachoneLogic::UpdateOrbitPos()
{
	if (!Logic) return;
	CurrentOrbitPos = Logic->PlayerPos + FVector2D(
		FMath::Cos(OrbitAngle + PhaseOff) * CachedOrbitRadius,
		FMath::Sin(OrbitAngle + PhaseOff) * CachedOrbitRadius);
}

void FSurvivorsWeaponPeachoneLogic::Tick(float Dt)
{
	if (!Logic) return;

	const FPassiveEffects& PE = GetPassiveEffects();

	const float RotSpeed = CachedOrbitRotSpeed * PE.SpeedMult;
	OrbitAngle += RotDir * RotSpeed * Dt;
	UpdateOrbitPos();

	if (PendingBombShots > 0)
	{
		BombShotTimer -= Dt;
		while (PendingBombShots > 0 && BombShotTimer <= 0.f)
		{
			SpawnBombShot();
			--PendingBombShots;
			BombShotTimer += SurvivorsGameConstants::PeachoneProjectileInterval;
		}
	}

	CooldownTimer.Value = FMath::Max(0.f, CooldownTimer.Value - Dt);
	if (CooldownTimer.IsReady() && PendingBombShots == 0) { StartBombing(); }
}

void FSurvivorsWeaponPeachoneLogic::StartBombing()
{
	const FPassiveEffects& PE = GetPassiveEffects();
	CooldownTimer = FCooldownSeconds(CachedCooldown * PE.CooldownMult);

	BurstDamage           = CachedDamage * PE.DamageMult
		/ static_cast<float>(SurvivorsGameConstants::PeachoneSetsPerActivation);
	BurstImpactRadius     = CachedImpactRadius * PE.AreaMult;
	BurstTargetZoneRadius = CachedTargetZoneRadius;

	const int32 EffAmount = CachedAmount + static_cast<int32>(PE.ExtraAmount);
	PendingBombShots = EffAmount * SurvivorsGameConstants::PeachoneSetsPerActivation;
	BombShotTimer    = 0.f;

	if (PendingBombShots > 0)
	{
		SpawnBombShot();
		--PendingBombShots;
		BombShotTimer = SurvivorsGameConstants::PeachoneProjectileInterval;
	}
}

void FSurvivorsWeaponPeachoneLogic::SpawnBombShot()
{
	if (!Logic) return;

	const float Angle = Logic->RandStream.FRand() * 2.f * UE_PI;
	const float Dist  = FMath::Sqrt(Logic->RandStream.FRand()) * BurstTargetZoneRadius;
	const FVector2D ImpactPos = CurrentOrbitPos + FVector2D(FMath::Cos(Angle), FMath::Sin(Angle)) * Dist;

	FProjectileState P;
	P.Pos               = ImpactPos;
	P.Vel               = FVector2D::ZeroVector;
	P.Radius            = FSimRadius(BurstImpactRadius);
	P.Damage            = FDamage(BurstDamage);
	P.WeaponType        = WeaponType;
	P.WeaponSlotIdx     = SlotIdx;
	P.LifeTime          = FProjectileLifeTime(0.1f);
	P.bPiercing         = true;
	P.MaxPierceCount    = 100;
	P.KnockbackStrength = SurvivorsGameConstants::KnockbackSim_2;
	Logic->SpawnProjectile(P);
}
