#include "Survivors/Game/Weapons/Projectile/SurvivorsLaurelWeapon.h"

#include "Survivors/Game/SurvivorsGame.h"
#include "Survivors/Game/SurvivorsGameConstants.h"

void USurvivorsLaurelWeapon::OnLevelChanged(FWeaponLevel NewLevel)
{
	CacheParams();
}

void USurvivorsLaurelWeapon::CacheParams()
{
	const int32 Lv  = FMath::Clamp(Level.Value, 1, SurvivorsGameConstants::MaxWeaponLevel);
	const int32 Idx = Lv - 1;

	const SurvivorsGameConstants::FLaurelParams& P = SurvivorsGameConstants::LaurelTable[Idx];
	CachedShieldDuration = P.ShieldDuration;
	CachedCooldown       = P.Cooldown;
}

void USurvivorsLaurelWeapon::Tick(float Dt)
{
	if (!Game) return;

	const FPassiveEffects& PE = GetPassiveEffects();

	// シールドタイマー更新
	if (Game->bShieldActive)
	{
		Game->PlayerShieldTimer -= Dt;
		if (Game->PlayerShieldTimer <= 0.f)
		{
			Game->bShieldActive     = false;
			Game->PlayerShieldTimer = 0.f;
		}
	}

	// クールダウン（シールド非アクティブ時のみカウント）
	if (!Game->bShieldActive)
	{
		CooldownTimer.Value = FMath::Max(0.f, CooldownTimer.Value - Dt);

		if (CooldownTimer.IsReady())
		{
			// シールド発動
			const float EffShieldDuration = CachedShieldDuration * PE.DurationMult;
			Game->bShieldActive     = true;
			Game->PlayerShieldTimer = EffShieldDuration;
			CooldownTimer = FCooldownSeconds(CachedCooldown * PE.CooldownMult);
		}
	}
}
