#include "Survivors/Logic/Weapons/Projectile/SurvivorsFireWandWeaponLogic.h"
#include "Survivors/Logic/SurvivorsGameLogic.h"
#include "Survivors/Logic/SurvivorsGameConstants.h"

void FSurvivorsFireWandWeaponLogic::OnLevelChanged(FWeaponLevel NewLevel)
{
	CacheParams();
}

void FSurvivorsFireWandWeaponLogic::CacheParams()
{
	const int32 Lv  = FMath::Clamp(Level.Value, 1, SurvivorsGameConstants::MaxWeaponLevel);
	const int32 Idx = Lv - 1;

	if (WeaponType == EWeaponType::Hellfire)
	{
		const SurvivorsGameConstants::FFireWandParams& P = SurvivorsGameConstants::HellfireTable[Idx];
		CachedDamage          = P.Damage;
		CachedCooldown        = P.Cooldown;
		CachedSpeed           = P.Speed;
		CachedExplosionRadius = P.ExplosionRadius;
		CachedAmount          = P.Amount;
	}
	else
	{
		const SurvivorsGameConstants::FFireWandParams& P = SurvivorsGameConstants::FireWandTable[Idx];
		CachedDamage          = P.Damage;
		CachedCooldown        = P.Cooldown;
		CachedSpeed           = P.Speed;
		CachedExplosionRadius = P.ExplosionRadius;
		CachedAmount          = P.Amount;
	}
}

void FSurvivorsFireWandWeaponLogic::Tick(float Dt)
{
	if (!Logic) return;

	const FPassiveEffects& PE = GetPassiveEffects();
	const float EffExplosionRadius = CachedExplosionRadius * PE.AreaMult;
	const float EffDamage          = CachedDamage * PE.DamageMult;
	const float EffDuration        = 0.2f * PE.DurationMult;

	Logic->UpdateProjectilesBySlot(SlotIdx, 0.f, [&](FProjectileState& P, float) -> bool
	{
		if (P.bPendingExplosion || P.LifeTime.IsExpired())
		{
			FGroundZoneState Z;
			Z.Pos           = P.Pos;
			Z.Radius        = EffExplosionRadius;
			Z.Damage        = EffDamage;
			Z.LifeTime      = EffDuration;
			Z.HitCooldown   = EffDuration;
			Z.WeaponSlotIdx = SlotIdx;
			Z.WeaponType    = WeaponType;
			Logic->SpawnGroundZone(Z);
			return false;
		}
		return true;
	});

	CooldownTimer.Value = FMath::Max(0.f, CooldownTimer.Value - Dt);
	if (!CooldownTimer.IsReady()) return;

	CooldownTimer = FCooldownSeconds(CachedCooldown * PE.CooldownMult);

	const float EffSpeed  = CachedSpeed * PE.SpeedMult;
	const float LifeTime  = 9.0f * PE.DurationMult;
	const int32 EffAmount = FMath::Max(1, CachedAmount + static_cast<int32>(PE.ExtraAmount));

	FVector2D Dir = FVector2D::ZeroVector;
	TArray<int32> Candidates;
	for (int32 EIdx = 0; EIdx < Logic->Enemies.Num(); ++EIdx)
	{
		const FEnemyState& E = Logic->Enemies[EIdx];
		if (E.bPendingRemove) continue;
		if (!Logic->IsOnScreen(E.Pos)) continue;
		Candidates.Add(EIdx);
	}
	if (Candidates.Num() > 0)
	{
		const int32 ChoiceIdx = Logic->RandStream.RandRange(0, Candidates.Num() - 1);
		Dir = (Logic->Enemies[Candidates[ChoiceIdx]].Pos - Logic->PlayerPos).GetSafeNormal();
	}
	if (Dir.IsNearlyZero())
	{
		const float RandomAngle = Logic->RandStream.FRand() * TWO_PI;
		Dir = FVector2D(FMath::Cos(RandomAngle), FMath::Sin(RandomAngle));
	}

	const float BaseAngle = FMath::Atan2(Dir.Y, Dir.X);
	const float AngleStep = FMath::DegreesToRadians(SurvivorsGameConstants::FireWandAngleStepDeg);
	for (int32 i = 0; i < EffAmount; ++i)
	{
		const float Offset   = (static_cast<float>(i) - 0.5f * static_cast<float>(EffAmount - 1)) * AngleStep;
		const float Angle    = BaseAngle + Offset;
		const FVector2D ShotDir(FMath::Cos(Angle), FMath::Sin(Angle));

		FProjectileState P;
		P.Pos           = Logic->PlayerPos;
		P.Vel           = ShotDir * EffSpeed;
		P.Radius        = FSimRadius(8.f);
		P.Damage        = FDamage(0.f);
		P.WeaponType    = WeaponType;
		P.WeaponSlotIdx = SlotIdx;
		P.LifeTime      = FProjectileLifeTime(LifeTime);
		P.bPiercing     = true;
		Logic->SpawnProjectile(P);
	}
}

void FSurvivorsFireWandWeaponLogic::ComputeHits(FSurvivorsHitFrame& HitFrame)
{
	if (!Logic) return;

	const FPassiveEffects& PE = GetPassiveEffects();
	const float EffExplosionRadius = CachedExplosionRadius * PE.AreaMult;
	const float EffDamage          = CachedDamage * PE.DamageMult;
	const float EffDuration        = 0.2f * PE.DurationMult;

	TArray<FProjectileState>& Projectiles = Logic->GetProjectiles();
	for (int32 i = Projectiles.Num() - 1; i >= 0; --i)
	{
		FProjectileState& P = Projectiles[i];
		if (P.WeaponSlotIdx != SlotIdx) continue;
		if (P.WeaponType != EWeaponType::FireWand && P.WeaponType != EWeaponType::Hellfire) continue;

		TArray<const FSurvivorsTargetProxy*> Contacts;
		Logic->QueryEnemyContacts(P.Pos, P.Radius.Value, Contacts);

		bool bHitEnemy = false;
		for (const FSurvivorsTargetProxy* Proxy : Contacts)
		{
			if ((P.Pos - Proxy->Pos).SizeSquared() > FMath::Square(P.Radius.Value + Proxy->Radius)) continue;
			const int32 EIdx = Proxy->Ref.IndexAtBuildTime;
			if (!Logic->Enemies.IsValidIndex(EIdx) || Logic->Enemies[EIdx].UniqueId != Proxy->Ref.UniqueId) continue;
			if (Logic->Enemies[EIdx].bPendingRemove) continue;
			bHitEnemy = true;
			break;
		}

		if (bHitEnemy || P.bPendingExplosion)
		{
			FGroundZoneState Z;
			Z.Pos           = P.Pos;
			Z.Radius        = EffExplosionRadius;
			Z.Damage        = EffDamage;
			Z.LifeTime      = EffDuration;
			Z.HitCooldown   = EffDuration;
			Z.WeaponSlotIdx = SlotIdx;
			Z.WeaponType    = WeaponType;
			Logic->SpawnGroundZone(Z);
			Projectiles.RemoveAt(i);
		}
	}
}
