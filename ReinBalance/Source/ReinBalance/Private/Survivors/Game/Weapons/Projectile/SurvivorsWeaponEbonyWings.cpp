#include "Survivors/Game/Weapons/Projectile/SurvivorsWeaponEbonyWings.h"
#include "Survivors/Game/SurvivorsGameConstants.h"

void USurvivorsWeaponEbonyWings::OnLevelChanged(FWeaponLevel NewLevel)
{
	CacheParams();
}

void USurvivorsWeaponEbonyWings::CacheParams()
{
	const int32 Lv  = FMath::Clamp(Level.Value, 1, SurvivorsGameConstants::MaxWeaponLevel);
	const int32 Idx = Lv - 1;

	// EbonyWings は Peachone と同じパラメータ・逆回転・π オフセット
	const SurvivorsGameConstants::FPeachoneParams& P = SurvivorsGameConstants::PeachoneTable[Idx];
	CachedDamage           = P.Damage;
	CachedCooldown         = P.Cooldown;
	CachedOrbitRadius      = P.OrbitRadius;
	CachedOrbitRotSpeed    = P.OrbitRotSpeed;
	CachedTargetZoneRadius = P.TargetZoneRadius;
	CachedImpactRadius     = P.ImpactRadius;
	CachedAmount           = P.Amount;

	RotDir   = -1.f;   // 反時計回り
	PhaseOff = UE_PI;  // π オフセット
}
