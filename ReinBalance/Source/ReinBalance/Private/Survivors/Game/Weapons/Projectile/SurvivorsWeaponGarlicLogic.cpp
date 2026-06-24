#include "Survivors/Game/Weapons/Projectile/SurvivorsWeaponGarlicLogic.h"
#include "Survivors/Game/SurvivorsGameLogic.h"
#include "Survivors/Game/SurvivorsGameConstants.h"

void FSurvivorsWeaponGarlicLogic::OnLevelChanged(FWeaponLevel NewLevel)
{
	CacheParams();
}

void FSurvivorsWeaponGarlicLogic::CacheParams()
{
	const int32 Lv  = FMath::Clamp(Level.Value, 1, SurvivorsGameConstants::MaxWeaponLevel);
	const int32 Idx = Lv - 1;

	if (WeaponType == EWeaponType::SoulEater)
	{
		const FGarlicParams& P = SurvivorsGameConstants::SoulEaterTable[Idx];
		CachedDamage      = P.Damage.Value;
		CachedHitInterval = P.HitInterval;
		CachedAreaRadius  = P.AreaRadius.Value;
	}
	else
	{
		const FGarlicParams& P = SurvivorsGameConstants::GarlicTable[Idx];
		CachedDamage      = P.Damage.Value;
		CachedHitInterval = P.HitInterval;
		CachedAreaRadius  = P.AreaRadius.Value;
	}
}

void FSurvivorsWeaponGarlicLogic::Tick(float Dt)
{
	if (!Logic) return;

	CooldownTimer.Value = FMath::Max(0.f, CooldownTimer.Value - Dt);
}

void FSurvivorsWeaponGarlicLogic::ComputeHits(FSurvivorsHitFrame& HitFrame)
{
	if (!Logic) return;

	const FPassiveEffects& PE = GetPassiveEffects();
	const float EffRadius = CachedAreaRadius * PE.AreaMult;
	const float EffDamage = CachedDamage * PE.DamageMult;

	TArray<const FSurvivorsTargetProxy*> Contacts;
	Logic->QueryEnemyContacts(Logic->PlayerPos, EffRadius, Contacts);

	for (const FSurvivorsTargetProxy* Proxy : Contacts)
	{
		if ((Logic->PlayerPos - Proxy->Pos).SizeSquared() > FMath::Square(EffRadius + Proxy->Radius)) continue;

		const int32 EIdx = Proxy->Ref.IndexAtBuildTime;
		if (!Logic->Enemies.IsValidIndex(EIdx) || Logic->Enemies[EIdx].UniqueId != Proxy->Ref.UniqueId) continue;
		const FEnemyState& E = Logic->Enemies[EIdx];
		if (E.bPendingRemove) continue;

		if (Logic->ElapsedTime - E.WeaponLastHitTime[SlotIdx].Seconds < CachedHitInterval) continue;

		const FVector2D KDir = (E.Pos - Logic->PlayerPos).GetSafeNormal();
		float KRes = 0.f;
		if (Logic->CurrentConfig.EnemyTypeTable.IsValidIndex(E.TypeId))
			KRes = Logic->CurrentConfig.EnemyTypeTable[E.TypeId].KnockbackResistance;

		FSurvivorsHitEvent Ev;
		Ev.Type                = ESurvivorsHitType::WeaponAreaDamage;
		Ev.Target              = Proxy->Ref;
		Ev.Damage              = EffDamage;
		Ev.WeaponSlot          = SlotIdx;
		Ev.KnockbackDir        = KDir;
		Ev.KnockbackStrength   = SurvivorsGameConstants::GarlicKnockbackStrength;
		Ev.KnockbackResistance = KRes;
		HitFrame.Events.Add(Ev);
	}
}
