#include "Survivors/Weapons/Projectile/SurvivorsWeaponLaurelLogic.h"
#include "Survivors/SurvivorsGameLogic.h"
#include "Survivors/SurvivorsGameConstants.h"

void FSurvivorsWeaponLaurelLogic::OnLevelChanged(FWeaponLevel NewLevel)
{
	CacheParams();
}

void FSurvivorsWeaponLaurelLogic::CacheParams()
{
	const int32 Lv  = FMath::Clamp(Level.Value, 1, SurvivorsGameConstants::MaxWeaponLevel);
	const int32 Idx = Lv - 1;

	const SurvivorsGameConstants::FLaurelParams& P = SurvivorsGameConstants::LaurelTable[Idx];
	CachedShieldDuration = P.ShieldDuration;
	CachedCooldown       = P.Cooldown;
}

float FSurvivorsWeaponLaurelLogic::GetCooldownObsDenominator() const
{
	// Laurel: 8〜10s の CD を正規化するため、cached cooldown を使う
	const FPassiveEffects& PE = GetPassiveEffects();
	return FMath::Max(CachedCooldown * PE.CooldownMult, KINDA_SMALL_NUMBER);
}

void FSurvivorsWeaponLaurelLogic::Tick(float Dt)
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
