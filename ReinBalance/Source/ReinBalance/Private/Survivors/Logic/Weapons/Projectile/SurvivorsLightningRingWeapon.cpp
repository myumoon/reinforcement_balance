#include "Survivors/Logic/Weapons/Projectile/SurvivorsLightningRingWeapon.h"

#include "Survivors/Logic/SurvivorsCollisionComponent.h"
#include "Survivors/Logic/SurvivorsGame.h"
#include "Survivors/Logic/SurvivorsGameConstants.h"

void USurvivorsLightningRingWeapon::OnLevelChanged(FWeaponLevel NewLevel)
{
	CacheParams();
}

void USurvivorsLightningRingWeapon::CacheParams()
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

// TODO(lightning ring marker): 現状は落雷後にランダム敵位置で直接ダメージを与えるのみ。
// weapon_lightning_ring.md: 「the lightning bolt visual and the ground hitbox separately」
// 改善計画:
//   1. 各 strike の impact 位置を短寿命 FGroundZoneState (bIsWarning=false, HitCooldown 大) で生成
//      → OBS に strike 位置が見えるようになる。
//   2. View 側: player center の ring 表示を strike marker に差し替える。
//   3. Obs 側: 既存 GroundZone obs stream に乗せるか、Projectile obs に bIsWarning=false の
//      marker として追加する。
// 現状の実装は OBS/view での認識には不十分であるため、次フェーズで対応する。

void USurvivorsLightningRingWeapon::Tick(float Dt)
{
	if (!Game) return;

	CooldownTimer.Value = FMath::Max(0.f, CooldownTimer.Value - Dt);

	if (CooldownTimer.IsReady() && !bPendingFire)
	{
		bPendingFire = true;
	}
}

void USurvivorsLightningRingWeapon::ComputeHits(USurvivorsCollisionComponent* CollComp, FSurvivorsHitFrame& HitFrame)
{
	if (!bPendingFire || !Game || !CollComp) return;
	bPendingFire = false;

	const FPassiveEffects& PE = GetPassiveEffects();
	CooldownTimer = FCooldownSeconds(CachedCooldown * PE.CooldownMult);

	const float EffDamage = CachedDamage * PE.DamageMult;
	const float EffRadius = CachedRadius * PE.AreaMult;
	const int32 EffAmount = CachedAmount + static_cast<int32>(PE.ExtraAmount);

	// ランダムに EffAmount 体の敵を選んでダメージ
	TArray<int32> EnemyIndices;
	EnemyIndices.Reserve(Game->Enemies.Num());
	for (int32 i = 0; i < Game->Enemies.Num(); ++i)
	{
		if (!Game->Enemies[i].bPendingRemove)
			EnemyIndices.Add(i);
	}

	// Fisher-Yates シャッフル（最大 EffAmount 個選出）
	const int32 SelectCount = FMath::Min(EffAmount, EnemyIndices.Num());
	for (int32 i = 0; i < SelectCount; ++i)
	{
		const int32 SwapIdx = i + Game->RandStream.RandRange(0, EnemyIndices.Num() - 1 - i);
		EnemyIndices.Swap(i, SwapIdx);
	}

	TSet<int32> HitEnemyIds;
	for (int32 i = 0; i < SelectCount; ++i)
	{
		const int32 StrikeIdx = EnemyIndices[i];
		const FEnemyState& StrikeTarget = Game->Enemies[StrikeIdx];

		TArray<const FSurvivorsTargetProxy*> Contacts;
		CollComp->QueryEnemyContacts(StrikeTarget.Pos, EffRadius, Contacts);

		for (const FSurvivorsTargetProxy* Proxy : Contacts)
		{
			if ((StrikeTarget.Pos - Proxy->Pos).SizeSquared() > FMath::Square(EffRadius + Proxy->Radius)) continue;

			const int32 EIdx = Proxy->Ref.IndexAtBuildTime;
			if (!Game->Enemies.IsValidIndex(EIdx) || Game->Enemies[EIdx].UniqueId != Proxy->Ref.UniqueId) continue;
			const FEnemyState& E = Game->Enemies[EIdx];
			if (E.bPendingRemove || HitEnemyIds.Contains(E.UniqueId)) continue;

			FSurvivorsHitEvent Ev;
			Ev.Type              = ESurvivorsHitType::WeaponAreaDamage;
			Ev.Target            = Proxy->Ref;
			Ev.Damage            = EffDamage;
			Ev.WeaponSlot        = SlotIdx;
			Ev.KnockbackDir      = (E.Pos - Game->PlayerPos).GetSafeNormal();
			Ev.KnockbackStrength = SurvivorsGameConstants::KnockbackSim_1;  // Knockback=1
			HitFrame.Events.Add(Ev);
			HitEnemyIds.Add(E.UniqueId);
		}
	}
}
