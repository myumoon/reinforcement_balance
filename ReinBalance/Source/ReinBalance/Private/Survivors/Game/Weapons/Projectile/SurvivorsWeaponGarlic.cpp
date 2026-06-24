#include "Survivors/Game/Weapons/Projectile/SurvivorsWeaponGarlic.h"

#include "Survivors/Game/SurvivorsCollisionComponent.h"
#include "Survivors/Game/SurvivorsGame.h"
#include "Survivors/Logic/SurvivorsGameConstants.h"
#include "Survivors/Game/SurvivorsGemComponent.h"
#include "Survivors/Game/Weapons/SurvivorsWeaponComponent.h"

void USurvivorsWeaponGarlic::OnLevelChanged(FWeaponLevel NewLevel)
{
	CacheParams();
}

void USurvivorsWeaponGarlic::CacheParams()
{
	const int32 Lv = FMath::Clamp(Level.Value, 1, SurvivorsGameConstants::MaxWeaponLevel);
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
		// Garlic（デフォルト）
		const FGarlicParams& P = SurvivorsGameConstants::GarlicTable[Idx];
		CachedDamage      = P.Damage.Value;
		CachedHitInterval = P.HitInterval;
		CachedAreaRadius  = P.AreaRadius.Value;
	}
}

void USurvivorsWeaponGarlic::Tick(float Dt)
{
	if (!Game) return;

	// クールダウン更新のみ（ダメージ処理は ComputeHits / ApplyWeaponHits に移管）
	CooldownTimer.Value = FMath::Max(0.f, CooldownTimer.Value - Dt);
}

void USurvivorsWeaponGarlic::ComputeHits(USurvivorsCollisionComponent* CollComp, FSurvivorsHitFrame& HitFrame)
{
	if (!Game || !CollComp) return;

	const FPassiveEffects& PE = GetPassiveEffects();
	const float EffRadius = CachedAreaRadius * PE.AreaMult;
	const float EffDamage = CachedDamage * PE.DamageMult;

	TArray<const FSurvivorsTargetProxy*> Contacts;
	CollComp->QueryEnemyContacts(Game->PlayerPos, EffRadius, Contacts);

	for (const FSurvivorsTargetProxy* Proxy : Contacts)
	{
		// narrowphase
		if ((Game->PlayerPos - Proxy->Pos).SizeSquared() > FMath::Square(EffRadius + Proxy->Radius)) continue;

		// UniqueId 確認
		const int32 EIdx = Proxy->Ref.IndexAtBuildTime;
		if (!Game->Enemies.IsValidIndex(EIdx) || Game->Enemies[EIdx].UniqueId != Proxy->Ref.UniqueId) continue;
		const FEnemyState& E = Game->Enemies[EIdx];
		if (E.bPendingRemove) continue;

		// hit interval
		if (Game->ElapsedTime - E.WeaponLastHitTime[SlotIdx].Seconds < CachedHitInterval) continue;

		// knockback
		const FVector2D KDir = (E.Pos - Game->PlayerPos).GetSafeNormal();
		float KRes = 0.f;
		if (Game->EnemyTypeTable.IsValidIndex(E.TypeId))
			KRes = Game->EnemyTypeTable[E.TypeId].KnockbackResistance;

		FSurvivorsHitEvent Ev;
		Ev.Type = ESurvivorsHitType::WeaponAreaDamage;
		Ev.Target = Proxy->Ref;
		Ev.Damage = EffDamage;
		Ev.WeaponSlot = SlotIdx;
		Ev.KnockbackDir = KDir;
		Ev.KnockbackStrength = SurvivorsGameConstants::GarlicKnockbackStrength;
		Ev.KnockbackResistance = KRes;
		HitFrame.Events.Add(Ev);
	}
}
