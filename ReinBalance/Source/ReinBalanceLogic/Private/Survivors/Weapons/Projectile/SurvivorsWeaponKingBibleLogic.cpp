#include "Survivors/Weapons/Projectile/SurvivorsWeaponKingBibleLogic.h"
#include "Survivors/SurvivorsGameLogic.h"
#include "Survivors/SurvivorsGameConstants.h"

void FSurvivorsWeaponKingBibleLogic::OnLevelChanged(FWeaponLevel NewLevel)
{
	CacheParams();
}

void FSurvivorsWeaponKingBibleLogic::CacheParams()
{
	const int32 Lv  = FMath::Clamp(Level.Value, 1, SurvivorsGameConstants::MaxWeaponLevel);
	const int32 Idx = Lv - 1;

	if (WeaponType == EWeaponType::UnholyVespers)
	{
		const SurvivorsGameConstants::FKingBibleParams& P = SurvivorsGameConstants::UnholyVespersTable[Idx];
		CachedDamage            = P.Damage;
		CachedCooldown          = P.Cooldown;
		CachedDuration          = P.Duration;
		CachedOrbitRadius       = P.OrbitRadius;
		CachedAmount            = P.Amount;
		CachedRotSpeed          = P.RotSpeed;
		CachedKnockbackStrength = P.KnockbackStrength;
	}
	else
	{
		const SurvivorsGameConstants::FKingBibleParams& P = SurvivorsGameConstants::KingBibleTable[Idx];
		CachedDamage            = P.Damage;
		CachedCooldown          = P.Cooldown;
		CachedDuration          = P.Duration;
		CachedOrbitRadius       = P.OrbitRadius;
		CachedAmount            = P.Amount;
		CachedRotSpeed          = P.RotSpeed;
		CachedKnockbackStrength = P.KnockbackStrength;
	}

	RebuildOrbPositions();
}

void FSurvivorsWeaponKingBibleLogic::ActivateOrbs(const FPassiveEffects& PE)
{
	bOrbsActive = true;
	ActiveTimer = CachedDuration * PE.DurationMult;

	const float CooldownInterval = (WeaponType == EWeaponType::UnholyVespers)
		? CachedCooldown * PE.CooldownMult
		: CachedCooldown * PE.CooldownMult + CachedDuration;
	CooldownTimer = FCooldownSeconds(FMath::Max(SurvivorsGameConstants::PhysicsDt, CooldownInterval));
	RebuildOrbPositions();
}

void FSurvivorsWeaponKingBibleLogic::RebuildOrbPositions()
{
	if (!Logic) return;
	if (!bOrbsActive) { OrbPositions.Reset(); return; }

	const FPassiveEffects& PE = GetPassiveEffects();
	const float EffRadius = CachedOrbitRadius * PE.AreaMult;
	const int32 EffAmount = FMath::Max(1, CachedAmount + static_cast<int32>(PE.ExtraAmount));

	OrbPositions.SetNum(EffAmount);
	for (int32 i = 0; i < EffAmount; ++i)
	{
		const float Angle = MasterAngle + (TWO_PI * i / EffAmount);
		OrbPositions[i]   = Logic->PlayerPos + FVector2D(FMath::Cos(Angle) * EffRadius, FMath::Sin(Angle) * EffRadius);
	}
}

void FSurvivorsWeaponKingBibleLogic::Tick(float Dt)
{
	if (!Logic) return;

	const FPassiveEffects& PE = GetPassiveEffects();
	CooldownTimer.Value = FMath::Max(0.f, CooldownTimer.Value - Dt);

	if (bOrbsActive)
	{
		ActiveTimer = FMath::Max(0.f, ActiveTimer - Dt);
		if (ActiveTimer <= 0.f) { bOrbsActive = false; OrbPositions.Reset(); }
	}

	if (CooldownTimer.IsReady()) { ActivateOrbs(PE); }
	if (!bOrbsActive) return;

	const float EffRadius = CachedOrbitRadius * PE.AreaMult;
	const int32 EffAmount = FMath::Max(1, CachedAmount + static_cast<int32>(PE.ExtraAmount));

	MasterAngle += CachedRotSpeed * PE.SpeedMult * Dt;
	OrbPositions.SetNum(EffAmount);
	for (int32 i = 0; i < EffAmount; ++i)
	{
		const float Angle = MasterAngle + (TWO_PI * i / EffAmount);
		OrbPositions[i]   = Logic->PlayerPos + FVector2D(FMath::Cos(Angle) * EffRadius, FMath::Sin(Angle) * EffRadius);
	}
}

void FSurvivorsWeaponKingBibleLogic::ComputeHits(FSurvivorsHitFrame& HitFrame)
{
	if (!Logic || !bOrbsActive) return;

	const FPassiveEffects& PE = GetPassiveEffects();
	const float EffDamage  = CachedDamage * PE.DamageMult;

	for (int32 OrbIdx = 0; OrbIdx < OrbPositions.Num(); ++OrbIdx)
	{
		const FVector2D& OrbPos = OrbPositions[OrbIdx];

		TArray<const FSurvivorsTargetProxy*> Contacts;
		Logic->QueryEnemyContacts(OrbPos, OrbVisualRadius, Contacts);

		for (const FSurvivorsTargetProxy* Proxy : Contacts)
		{
			if ((OrbPos - Proxy->Pos).SizeSquared() > FMath::Square(OrbVisualRadius + Proxy->Radius)) continue;

			const int32 EIdx = Proxy->Ref.IndexAtBuildTime;
			if (!Logic->Enemies.IsValidIndex(EIdx) || Logic->Enemies[EIdx].UniqueId != Proxy->Ref.UniqueId) continue;
			const FEnemyState& E = Logic->Enemies[EIdx];
			if (E.bPendingRemove) continue;

			const int32 OrbKey     = SlotIdx * 10 + OrbIdx;
			const float* LastOrbHit = E.OrbHitTimes.Find(OrbKey);
			if (LastOrbHit && (Logic->ElapsedTime - *LastOrbHit) < SurvivorsGameConstants::KingBibleOrbHitInterval)
				continue;

			FSurvivorsHitEvent Ev;
			Ev.Type              = ESurvivorsHitType::WeaponAreaDamage;
			Ev.Target            = Proxy->Ref;
			Ev.Damage            = EffDamage;
			Ev.WeaponSlot        = SlotIdx;
			Ev.OrbIdx            = OrbIdx;
			Ev.KnockbackDir      = (Proxy->Pos - Logic->PlayerPos).GetSafeNormal();
			Ev.KnockbackStrength = CachedKnockbackStrength;
			HitFrame.Events.Add(Ev);
			break;
		}
	}
}
