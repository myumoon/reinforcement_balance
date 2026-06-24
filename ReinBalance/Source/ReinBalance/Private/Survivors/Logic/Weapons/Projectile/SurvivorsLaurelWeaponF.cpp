#include "Survivors/Logic/Weapons/Projectile/SurvivorsLaurelWeaponF.h"
#include "Survivors/Logic/SurvivorsGameLogic.h"
#include "Survivors/Logic/SurvivorsGameConstants.h"

void FSurvivorsLaurelWeapon::OnLevelChanged(FWeaponLevel NewLevel)
{
	CacheParams();
}

void FSurvivorsLaurelWeapon::CacheParams()
{
	const int32 Lv  = FMath::Clamp(Level.Value, 1, SurvivorsGameConstants::MaxWeaponLevel);
	const int32 Idx = Lv - 1;

	const SurvivorsGameConstants::FLaurelParams& P = SurvivorsGameConstants::LaurelTable[Idx];
	CachedShieldDuration = P.ShieldDuration;
	CachedCooldown       = P.Cooldown;
}

void FSurvivorsLaurelWeapon::Tick(float Dt)
{
	if (!Logic) return;

	const FPassiveEffects& PE = GetPassiveEffects();

	if (Logic->bShieldActive)
	{
		Logic->PlayerShieldTimer -= Dt;
		if (Logic->PlayerShieldTimer <= 0.f)
		{
			Logic->bShieldActive     = false;
			Logic->PlayerShieldTimer = 0.f;
		}
	}

	if (!Logic->bShieldActive)
	{
		CooldownTimer.Value = FMath::Max(0.f, CooldownTimer.Value - Dt);
		if (CooldownTimer.IsReady())
		{
			const float EffShieldDuration = CachedShieldDuration * PE.DurationMult;
			Logic->bShieldActive          = true;
			Logic->PlayerShieldTimer      = EffShieldDuration;
			CooldownTimer = FCooldownSeconds(CachedCooldown * PE.CooldownMult);
		}
	}
}
