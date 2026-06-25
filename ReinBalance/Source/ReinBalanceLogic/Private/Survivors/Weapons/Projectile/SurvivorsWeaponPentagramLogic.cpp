#include "Survivors/Weapons/Projectile/SurvivorsWeaponPentagramLogic.h"
#include "Survivors/SurvivorsGameLogic.h"
#include "Survivors/SurvivorsGameConstants.h"

float FSurvivorsWeaponPentagramLogic::GetCooldownObsDenominator() const
{
	// Pentagram/GorgeousMoon: 60〜90s の長 CD を正規化するため、cached cooldown を使う
	const FPassiveEffects& PE = GetPassiveEffects();
	return FMath::Max(CachedCooldown * PE.CooldownMult, KINDA_SMALL_NUMBER);
}

void FSurvivorsWeaponPentagramLogic::OnLevelChanged(FWeaponLevel NewLevel)
{
	CacheParams();
}

void FSurvivorsWeaponPentagramLogic::CacheParams()
{
	const int32 Lv  = FMath::Clamp(Level.Value, 1, SurvivorsGameConstants::MaxWeaponLevel);
	const int32 Idx = Lv - 1;

	if (WeaponType == EWeaponType::GorgeousMoon)
	{
		const SurvivorsGameConstants::FPentagramParams& P = SurvivorsGameConstants::GorgeousMoonTable[Idx];
		CachedDamage   = P.Damage;
		CachedCooldown = P.Cooldown;
		CachedRadius   = P.Radius;
	}
	else
	{
		const SurvivorsGameConstants::FPentagramParams& P = SurvivorsGameConstants::PentagramTable[Idx];
		CachedDamage   = P.Damage;
		CachedCooldown = P.Cooldown;
		CachedRadius   = P.Radius;
	}
}

void FSurvivorsWeaponPentagramLogic::Tick(float Dt)
{
	if (!Logic) return;

	CooldownTimer.Value = FMath::Max(0.f, CooldownTimer.Value - Dt);
	if (CooldownTimer.IsReady() && !bPendingFire) { bPendingFire = true; }
}

void FSurvivorsWeaponPentagramLogic::ComputeHits(FSurvivorsHitFrame& HitFrame)
{
	if (!bPendingFire || !Logic) return;
	bPendingFire = false;

	const FPassiveEffects& PE = GetPassiveEffects();
	CooldownTimer = FCooldownSeconds(CachedCooldown * PE.CooldownMult);

	const float EffDamage = CachedDamage * PE.DamageMult;
	const float EffRadius = CachedRadius * PE.AreaMult;

	TArray<const FSurvivorsTargetProxy*> Contacts;
	Logic->QueryEnemyContacts(Logic->PlayerPos, EffRadius, Contacts);

	for (const FSurvivorsTargetProxy* Proxy : Contacts)
	{
		const int32 EIdx = Proxy->Ref.IndexAtBuildTime;
		if (!Logic->Enemies.IsValidIndex(EIdx) || Logic->Enemies[EIdx].UniqueId != Proxy->Ref.UniqueId) continue;
		if (Logic->Enemies[EIdx].bPendingRemove) continue;

		FSurvivorsHitEvent Ev;
		Ev.Type       = ESurvivorsHitType::WeaponAreaDamage;
		Ev.Target     = Proxy->Ref;
		Ev.Damage     = EffDamage;
		Ev.WeaponSlot = SlotIdx;
		HitFrame.Events.Add(Ev);
	}
}
