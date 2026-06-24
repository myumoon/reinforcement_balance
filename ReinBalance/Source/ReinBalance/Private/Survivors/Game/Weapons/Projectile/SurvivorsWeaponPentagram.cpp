#include "Survivors/Game/Weapons/Projectile/SurvivorsWeaponPentagram.h"

#include "Survivors/Game/SurvivorsCollisionComponent.h"
#include "Survivors/Game/SurvivorsGame.h"
#include "Survivors/Logic/SurvivorsGameConstants.h"

void USurvivorsWeaponPentagram::OnLevelChanged(FWeaponLevel NewLevel)
{
	CacheParams();
}

void USurvivorsWeaponPentagram::CacheParams()
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

void USurvivorsWeaponPentagram::Tick(float Dt)
{
	if (!Game) return;

	CooldownTimer.Value = FMath::Max(0.f, CooldownTimer.Value - Dt);
	if (CooldownTimer.IsReady() && !bPendingFire)
	{
		bPendingFire = true;
	}
}

void USurvivorsWeaponPentagram::ComputeHits(USurvivorsCollisionComponent* CollComp, FSurvivorsHitFrame& HitFrame)
{
	if (!bPendingFire || !Game || !CollComp) return;
	bPendingFire = false;

	const FPassiveEffects& PE = GetPassiveEffects();
	CooldownTimer = FCooldownSeconds(CachedCooldown * PE.CooldownMult);

	const float EffDamage = CachedDamage * PE.DamageMult;
	const float EffRadius = CachedRadius * PE.AreaMult;

	// 全敵に即ダメージ（Radius は 9999 なので実質全員）
	TArray<const FSurvivorsTargetProxy*> Contacts;
	CollComp->QueryEnemyContacts(Game->PlayerPos, EffRadius, Contacts);

	for (const FSurvivorsTargetProxy* Proxy : Contacts)
	{
		const int32 EIdx = Proxy->Ref.IndexAtBuildTime;
		if (!Game->Enemies.IsValidIndex(EIdx) || Game->Enemies[EIdx].UniqueId != Proxy->Ref.UniqueId) continue;
		if (Game->Enemies[EIdx].bPendingRemove) continue;

		FSurvivorsHitEvent Ev;
		Ev.Type      = ESurvivorsHitType::WeaponAreaDamage;
		Ev.Target    = Proxy->Ref;
		Ev.Damage    = EffDamage;
		Ev.WeaponSlot = SlotIdx;
		HitFrame.Events.Add(Ev);
	}
}
