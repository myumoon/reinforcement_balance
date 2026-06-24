#include "Survivors/Logic/Weapons/Projectile/SurvivorsPentagramWeaponF.h"
#include "Survivors/Logic/SurvivorsGameLogic.h"
#include "Survivors/Logic/SurvivorsGameConstants.h"

void FSurvivorsPentagramWeapon::OnLevelChanged(FWeaponLevel NewLevel)
{
	CacheParams();
}

void FSurvivorsPentagramWeapon::CacheParams()
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

void FSurvivorsPentagramWeapon::Tick(float Dt)
{
	if (!Logic) return;

	CooldownTimer.Value = FMath::Max(0.f, CooldownTimer.Value - Dt);
	if (CooldownTimer.IsReady() && !bPendingFire) { bPendingFire = true; }
}

void FSurvivorsPentagramWeapon::ComputeHits(FSurvivorsHitFrame& HitFrame)
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
