#include "Survivors/Logic/Weapons/Projectile/SurvivorsPeachoneWeapon.h"

#include "Survivors/Logic/SurvivorsCollisionComponent.h"
#include "Survivors/Logic/SurvivorsGame.h"
#include "Survivors/Logic/SurvivorsGameConstants.h"

void USurvivorsPeachoneWeapon::OnLevelChanged(FWeaponLevel NewLevel)
{
	CacheParams();
}

void USurvivorsPeachoneWeapon::CacheParams()
{
	const int32 Lv  = FMath::Clamp(Level.Value, 1, SurvivorsGameConstants::MaxWeaponLevel);
	const int32 Idx = Lv - 1;

	// WeaponType は Peachone 固定（EbonyWings は派生クラスでオーバーライド）
	const SurvivorsGameConstants::FPeachoneParams& P = SurvivorsGameConstants::PeachoneTable[Idx];
	CachedDamage      = P.Damage;
	CachedCooldown    = P.Cooldown;
	CachedOrbitRadius = P.OrbitRadius;
	CachedBombRadius  = P.BombRadius;
}

void USurvivorsPeachoneWeapon::UpdateOrbitPos()
{
	if (!Game) return;
	const FPassiveEffects& PE = GetPassiveEffects();
	const float EffRadius = CachedOrbitRadius * PE.AreaMult;
	CurrentOrbitPos = Game->PlayerPos + FVector2D(
		FMath::Cos(OrbitAngle + PhaseOff) * EffRadius,
		FMath::Sin(OrbitAngle + PhaseOff) * EffRadius);
}

void USurvivorsPeachoneWeapon::Tick(float Dt)
{
	if (!Game) return;

	const FPassiveEffects& PE = GetPassiveEffects();
	const float EffCooldown = CachedCooldown * PE.CooldownMult;

	// 軌道角度を更新
	const float RotSpeed = 3.0f * PE.SpeedMult;  // rad/sec
	OrbitAngle += RotDir * RotSpeed * Dt;
	UpdateOrbitPos();

	// クールダウン
	CooldownTimer.Value = FMath::Max(0.f, CooldownTimer.Value - Dt);
	if (CooldownTimer.IsReady() && !bPendingFire)
	{
		bPendingFire = true;
	}
}

void USurvivorsPeachoneWeapon::ComputeHits(USurvivorsCollisionComponent* CollComp, FSurvivorsHitFrame& HitFrame)
{
	if (!bPendingFire || !Game || !CollComp) return;
	bPendingFire = false;

	const FPassiveEffects& PE = GetPassiveEffects();
	CooldownTimer = FCooldownSeconds(CachedCooldown * PE.CooldownMult);

	const float EffDamage      = CachedDamage    * PE.DamageMult;
	const float EffBombRadius  = CachedBombRadius * PE.AreaMult;

	// 現在の軌道位置で爆発
	TArray<const FSurvivorsTargetProxy*> Contacts;
	CollComp->QueryEnemyContacts(CurrentOrbitPos, EffBombRadius, Contacts);

	for (const FSurvivorsTargetProxy* Proxy : Contacts)
	{
		if ((CurrentOrbitPos - Proxy->Pos).SizeSquared() > FMath::Square(EffBombRadius + Proxy->Radius)) continue;

		const int32 EIdx = Proxy->Ref.IndexAtBuildTime;
		if (!Game->Enemies.IsValidIndex(EIdx) || Game->Enemies[EIdx].UniqueId != Proxy->Ref.UniqueId) continue;
		if (Game->Enemies[EIdx].bPendingRemove) continue;

		FSurvivorsHitEvent Ev;
		Ev.Type              = ESurvivorsHitType::WeaponAreaDamage;
		Ev.Target            = Proxy->Ref;
		Ev.Damage            = EffDamage;
		Ev.WeaponSlot        = SlotIdx;
		Ev.KnockbackDir      = (Proxy->Pos - Game->PlayerPos).GetSafeNormal();
		Ev.KnockbackStrength = SurvivorsGameConstants::KnockbackSim_2;  // Knockback=2
		HitFrame.Events.Add(Ev);
	}
}
