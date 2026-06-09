#include "Survivors/Logic/Weapons/Projectile/SurvivorsVandalierWeapon.h"

#include "Survivors/Logic/SurvivorsCollisionComponent.h"
#include "Survivors/Logic/SurvivorsGame.h"
#include "Survivors/Logic/SurvivorsGameConstants.h"

void USurvivorsVandalierWeapon::OnLevelChanged(FWeaponLevel NewLevel)
{
	CacheParams();
}

void USurvivorsVandalierWeapon::CacheParams()
{
	const int32 Lv  = FMath::Clamp(Level.Value, 1, SurvivorsGameConstants::MaxWeaponLevel);
	const int32 Idx = Lv - 1;

	// Vandalier は Peachone Lv8 をベースにした固定パラメータ（Lv1〜8 でスケール）
	const SurvivorsGameConstants::FPeachoneParams& P = SurvivorsGameConstants::PeachoneTable[Idx];
	// Vandalier は Peachone + EbonyWings 統合のため約 1.5 倍強化
	CachedDamage      = P.Damage * 1.5f;
	CachedCooldown    = P.Cooldown * 0.8f;
	CachedOrbitRadius = P.OrbitRadius + 10.f;
	CachedBombRadius  = P.BombRadius + 10.f;
}

FVector2D USurvivorsVandalierWeapon::GetOrbitOrbPos(int32 OrbIdx) const
{
	if (!Game || OrbIdx < 0 || OrbIdx >= 2) return FVector2D::ZeroVector;
	const FPassiveEffects& PE = GetPassiveEffects();
	const float EffOrbitRadius = CachedOrbitRadius * PE.AreaMult;
	return Game->PlayerPos + FVector2D(
		FMath::Cos(OrbitAngle[OrbIdx]) * EffOrbitRadius,
		FMath::Sin(OrbitAngle[OrbIdx]) * EffOrbitRadius);
}

void USurvivorsVandalierWeapon::Tick(float Dt)
{
	if (!Game) return;

	const float RotSpeed = 1.5f;  // rad/sec
	OrbitAngle[0] += RotSpeed * Dt;
	OrbitAngle[1] -= RotSpeed * Dt;  // 逆回転

	// クールダウン
	CooldownTimer.Value = FMath::Max(0.f, CooldownTimer.Value - Dt);
	if (CooldownTimer.IsReady() && !bPendingFire)
	{
		bPendingFire = true;
	}
}

void USurvivorsVandalierWeapon::ComputeHits(USurvivorsCollisionComponent* CollComp, FSurvivorsHitFrame& HitFrame)
{
	if (!bPendingFire || !Game || !CollComp) return;
	bPendingFire = false;

	const FPassiveEffects& PE = GetPassiveEffects();
	CooldownTimer = FCooldownSeconds(CachedCooldown * PE.CooldownMult);

	const float EffDamage     = CachedDamage    * PE.DamageMult;
	const float EffOrbitRadius = CachedOrbitRadius * PE.AreaMult;
	const float EffBombRadius  = CachedBombRadius  * PE.AreaMult;

	// 2 軌道それぞれで爆発
	for (int32 OrbIdx = 0; OrbIdx < 2; ++OrbIdx)
	{
		const FVector2D OrbPos = Game->PlayerPos + FVector2D(
			FMath::Cos(OrbitAngle[OrbIdx]) * EffOrbitRadius,
			FMath::Sin(OrbitAngle[OrbIdx]) * EffOrbitRadius);

		TArray<const FSurvivorsTargetProxy*> Contacts;
		CollComp->QueryEnemyContacts(OrbPos, EffBombRadius, Contacts);

		for (const FSurvivorsTargetProxy* Proxy : Contacts)
		{
			if ((OrbPos - Proxy->Pos).SizeSquared() > FMath::Square(EffBombRadius + Proxy->Radius)) continue;

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
}
