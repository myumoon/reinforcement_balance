#include "Survivors/Logic/Weapons/Projectile/SurvivorsLightningRingWeaponF.h"
#include "Survivors/Logic/SurvivorsGameLogic.h"
#include "Survivors/Logic/SurvivorsGameConstants.h"

void FSurvivorsLightningRingWeapon::OnLevelChanged(FWeaponLevel NewLevel)
{
	CacheParams();
}

void FSurvivorsLightningRingWeapon::CacheParams()
{
	const int32 Lv  = FMath::Clamp(Level.Value, 1, SurvivorsGameConstants::MaxWeaponLevel);
	const int32 Idx = Lv - 1;

	if (WeaponType == EWeaponType::ThunderLoop)
	{
		const SurvivorsGameConstants::FLightningRingParams& P = SurvivorsGameConstants::ThunderLoopTable[Idx];
		CachedDamage   = P.Damage;
		CachedCooldown = P.Cooldown;
		CachedRadius   = P.Radius;
		CachedAmount   = P.Amount;
	}
	else
	{
		const SurvivorsGameConstants::FLightningRingParams& P = SurvivorsGameConstants::LightningRingTable[Idx];
		CachedDamage   = P.Damage;
		CachedCooldown = P.Cooldown;
		CachedRadius   = P.Radius;
		CachedAmount   = P.Amount;
	}
}

void FSurvivorsLightningRingWeapon::Tick(float Dt)
{
	if (!Logic) return;

	CooldownTimer.Value = FMath::Max(0.f, CooldownTimer.Value - Dt);
	if (CooldownTimer.IsReady() && !bPendingFire) { bPendingFire = true; }
}

void FSurvivorsLightningRingWeapon::ComputeHits(FSurvivorsHitFrame& HitFrame)
{
	if (!bPendingFire || !Logic) return;
	bPendingFire = false;

	const FPassiveEffects& PE = GetPassiveEffects();
	CooldownTimer = FCooldownSeconds(CachedCooldown * PE.CooldownMult);

	const float EffDamage = CachedDamage * PE.DamageMult;
	const float EffRadius = CachedRadius * PE.AreaMult;
	const int32 EffAmount = CachedAmount + static_cast<int32>(PE.ExtraAmount);

	TArray<int32> EnemyIndices;
	EnemyIndices.Reserve(Logic->Enemies.Num());
	for (int32 i = 0; i < Logic->Enemies.Num(); ++i)
	{
		if (!Logic->Enemies[i].bPendingRemove) EnemyIndices.Add(i);
	}

	const int32 SelectCount = FMath::Min(EffAmount, EnemyIndices.Num());
	for (int32 i = 0; i < SelectCount; ++i)
	{
		const int32 SwapIdx = i + Logic->RandStream.RandRange(0, EnemyIndices.Num() - 1 - i);
		EnemyIndices.Swap(i, SwapIdx);
	}

	TSet<int32> HitEnemyIds;
	for (int32 i = 0; i < SelectCount; ++i)
	{
		const int32 StrikeIdx    = EnemyIndices[i];
		const FEnemyState& Strike = Logic->Enemies[StrikeIdx];

		FGroundZoneState Marker;
		Marker.Pos           = Strike.Pos;
		Marker.Radius        = EffRadius;
		Marker.Damage        = 0.f;
		Marker.LifeTime      = SurvivorsGameConstants::LightningRingStrikeLifeTime;
		Marker.WarningTime   = 0.f;
		Marker.HitCooldown   = 9999.f;
		Marker.WeaponSlotIdx = SlotIdx;
		Marker.WeaponType    = WeaponType;
		Marker.bIsWarning    = false;
		Logic->SpawnGroundZone(Marker);

		TArray<const FSurvivorsTargetProxy*> Contacts;
		Logic->QueryEnemyContacts(Strike.Pos, EffRadius, Contacts);

		for (const FSurvivorsTargetProxy* Proxy : Contacts)
		{
			if ((Strike.Pos - Proxy->Pos).SizeSquared() > FMath::Square(EffRadius + Proxy->Radius)) continue;
			const int32 EIdx = Proxy->Ref.IndexAtBuildTime;
			if (!Logic->Enemies.IsValidIndex(EIdx) || Logic->Enemies[EIdx].UniqueId != Proxy->Ref.UniqueId) continue;
			const FEnemyState& E = Logic->Enemies[EIdx];
			if (E.bPendingRemove || HitEnemyIds.Contains(E.UniqueId)) continue;

			FSurvivorsHitEvent Ev;
			Ev.Type              = ESurvivorsHitType::WeaponAreaDamage;
			Ev.Target            = Proxy->Ref;
			Ev.Damage            = EffDamage;
			Ev.WeaponSlot        = SlotIdx;
			Ev.KnockbackDir      = (E.Pos - Logic->PlayerPos).GetSafeNormal();
			Ev.KnockbackStrength = SurvivorsGameConstants::KnockbackSim_1;
			HitFrame.Events.Add(Ev);
			HitEnemyIds.Add(E.UniqueId);
		}
	}
}
