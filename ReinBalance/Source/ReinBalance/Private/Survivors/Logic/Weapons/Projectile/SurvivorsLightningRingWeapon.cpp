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
		CachedAmount   = P.Amount;
	}
	else
	{
		const SurvivorsGameConstants::FLightningRingParams& P = SurvivorsGameConstants::LightningRingTable[Idx];
		CachedDamage   = P.Damage;
		CachedCooldown = P.Cooldown;
		CachedAmount   = P.Amount;
	}
}

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

	for (int32 i = 0; i < SelectCount; ++i)
	{
		const int32 EIdx = EnemyIndices[i];
		const FEnemyState& E = Game->Enemies[EIdx];

		FSurvivorsCollisionRef Ref;
		Ref.Kind             = ESurvivorsCollisionOwnerKind::Enemy;
		Ref.UniqueId         = E.UniqueId;
		Ref.IndexAtBuildTime = EIdx;

		FSurvivorsHitEvent Ev;
		Ev.Type      = ESurvivorsHitType::WeaponAreaDamage;
		Ev.Target    = Ref;
		Ev.Damage    = EffDamage;
		Ev.WeaponSlot = SlotIdx;
		HitFrame.Events.Add(Ev);
	}
}
