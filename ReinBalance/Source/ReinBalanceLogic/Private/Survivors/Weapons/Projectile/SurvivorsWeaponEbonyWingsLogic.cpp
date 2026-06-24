#include "Survivors/Weapons/Projectile/SurvivorsWeaponEbonyWingsLogic.h"
#include "Survivors/SurvivorsGameConstants.h"

void FSurvivorsWeaponEbonyWingsLogic::OnLevelChanged(FWeaponLevel NewLevel)
{
	CacheParams();
}

void FSurvivorsWeaponEbonyWingsLogic::CacheParams()
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

	RotDir   = -1.f;
	PhaseOff = UE_PI;
}
