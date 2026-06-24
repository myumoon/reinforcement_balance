#include "Survivors/Logic/Weapons/Projectile/SurvivorsVandalierWeaponLogic.h"
#include "Survivors/Logic/SurvivorsGameLogic.h"
#include "Survivors/Logic/SurvivorsGameConstants.h"

void FSurvivorsVandalierWeaponLogic::OnLevelChanged(FWeaponLevel NewLevel)
{
	CacheParams();
}

void FSurvivorsVandalierWeaponLogic::CacheParams()
{
	const int32 Lv  = FMath::Clamp(Level.Value, 1, SurvivorsGameConstants::MaxWeaponLevel);
	const int32 Idx = Lv - 1;

	const SurvivorsGameConstants::FPeachoneParams& P = SurvivorsGameConstants::PeachoneTable[Idx];
	CachedDamage           = P.Damage * 1.5f;
	CachedCooldown         = P.Cooldown * 0.8f;
	CachedOrbitRadius      = P.OrbitRadius + 10.f;
	CachedOrbitRotSpeed    = P.OrbitRotSpeed;
	CachedTargetZoneRadius = P.TargetZoneRadius + 10.f;
	CachedImpactRadius     = P.ImpactRadius;
	CachedAmount           = P.Amount;
}

FVector2D FSurvivorsVandalierWeaponLogic::GetOrbitOrbPos(int32 OrbIdx) const
{
	if (!Logic || OrbIdx < 0 || OrbIdx >= 2) return FVector2D::ZeroVector;
	return Logic->PlayerPos + FVector2D(
		FMath::Cos(OrbitAngle[OrbIdx]) * CachedOrbitRadius,
		FMath::Sin(OrbitAngle[OrbIdx]) * CachedOrbitRadius);
}

void FSurvivorsVandalierWeaponLogic::Tick(float Dt)
{
	if (!Logic) return;

	const FPassiveEffects& PE = GetPassiveEffects();
	const float RotSpeed = CachedOrbitRotSpeed * PE.SpeedMult;
	OrbitAngle[0] += RotSpeed * Dt;
	OrbitAngle[1] -= RotSpeed * Dt;

	for (int32 OrbIdx = 0; OrbIdx < 2; ++OrbIdx)
	{
		if (PendingBombShots[OrbIdx] > 0)
		{
			BombShotTimer[OrbIdx] -= Dt;
			while (PendingBombShots[OrbIdx] > 0 && BombShotTimer[OrbIdx] <= 0.f)
			{
				SpawnBombShot(OrbIdx);
				--PendingBombShots[OrbIdx];
				BombShotTimer[OrbIdx] += SurvivorsGameConstants::PeachoneProjectileInterval;
			}
		}
	}

	const bool bBurstDone = (PendingBombShots[0] == 0 && PendingBombShots[1] == 0);
	CooldownTimer.Value = FMath::Max(0.f, CooldownTimer.Value - Dt);
	if (CooldownTimer.IsReady() && bBurstDone) { StartBombing(); }
}

void FSurvivorsVandalierWeaponLogic::StartBombing()
{
	const FPassiveEffects& PE = GetPassiveEffects();
	CooldownTimer = FCooldownSeconds(CachedCooldown * PE.CooldownMult);

	BurstDamage           = CachedDamage * PE.DamageMult
		/ static_cast<float>(SurvivorsGameConstants::PeachoneSetsPerActivation) / 2.f;
	BurstImpactRadius     = CachedImpactRadius * PE.AreaMult;
	BurstTargetZoneRadius = CachedTargetZoneRadius;

	const int32 EffAmount  = CachedAmount + static_cast<int32>(PE.ExtraAmount);
	const int32 TotalShots = EffAmount * SurvivorsGameConstants::PeachoneSetsPerActivation;

	for (int32 OrbIdx = 0; OrbIdx < 2; ++OrbIdx)
	{
		PendingBombShots[OrbIdx] = TotalShots;
		BombShotTimer[OrbIdx]    = 0.f;

		if (PendingBombShots[OrbIdx] > 0)
		{
			SpawnBombShot(OrbIdx);
			--PendingBombShots[OrbIdx];
			BombShotTimer[OrbIdx] = SurvivorsGameConstants::PeachoneProjectileInterval;
		}
	}
}

void FSurvivorsVandalierWeaponLogic::SpawnBombShot(int32 OrbIdx)
{
	if (!Logic) return;

	const FVector2D ZoneCenter = GetOrbitOrbPos(OrbIdx);
	const float Angle = Logic->RandStream.FRand() * 2.f * UE_PI;
	const float Dist  = FMath::Sqrt(Logic->RandStream.FRand()) * BurstTargetZoneRadius;
	const FVector2D ImpactPos = ZoneCenter + FVector2D(FMath::Cos(Angle), FMath::Sin(Angle)) * Dist;

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
