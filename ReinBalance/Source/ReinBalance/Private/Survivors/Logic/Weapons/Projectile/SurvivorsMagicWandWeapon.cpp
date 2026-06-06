#include "Survivors/Logic/Weapons/Projectile/SurvivorsMagicWandWeapon.h"

#include "Survivors/Logic/SurvivorsCollisionComponent.h"
#include "Survivors/Logic/SurvivorsGame.h"
#include "Survivors/Logic/SurvivorsGameConstants.h"
#include "Survivors/Logic/Weapons/SurvivorsWeaponComponent.h"

void USurvivorsMagicWandWeapon::OnLevelChanged(FWeaponLevel NewLevel)
{
	CacheParams();
}

void USurvivorsMagicWandWeapon::CacheParams()
{
	const int32 Lv  = FMath::Clamp(Level.Value, 1, SurvivorsGameConstants::MaxWeaponLevel);
	const int32 Idx = Lv - 1;

	if (WeaponType == EWeaponType::HolyWand)
	{
		const SurvivorsGameConstants::FMagicWandParams& P = SurvivorsGameConstants::HolyWandTable[Idx];
		CachedDamage   = P.Damage;
		CachedCooldown = P.Cooldown;
		CachedSpeed    = P.Speed;
		CachedAmount   = P.Amount;
		CachedPierce   = P.Pierce;
	}
	else
	{
		const SurvivorsGameConstants::FMagicWandParams& P = SurvivorsGameConstants::MagicWandTable[Idx];
		CachedDamage   = P.Damage;
		CachedCooldown = P.Cooldown;
		CachedSpeed    = P.Speed;
		CachedAmount   = P.Amount;
		CachedPierce   = P.Pierce;
	}
}

void USurvivorsMagicWandWeapon::Tick(float Dt)
{
	if (!Game || !WeaponComp) return;

	CooldownTimer.Value = FMath::Max(0.f, CooldownTimer.Value - Dt);
	if (!CooldownTimer.IsReady()) return;

	const FPassiveEffects& PE = GetPassiveEffects();
	const float EffCooldown = CachedCooldown * PE.CooldownMult;
	CooldownTimer = FCooldownSeconds(EffCooldown);

	const float EffDamage = CachedDamage * PE.DamageMult;
	const float EffSpeed  = CachedSpeed  * PE.SpeedMult;
	const int32 EffAmount = CachedAmount + static_cast<int32>(PE.ExtraAmount);
	const float LifeTime  = 1.5f * PE.DurationMult;

	// 最近傍敵を探す（Enemies 配列を直接走査しない → Enemy 座標のみ最短距離で選ぶ）
	// ただし当たり判定は ComputeProjectileHits が行うためここは発射のみ
	// 最近傍候補がいない場合は上方向に発射
	FVector2D BaseDir = FVector2D(0.f, 1.f);
	float MinDistSq = MAX_FLT;
	for (const FEnemyState& E : Game->Enemies)
	{
		if (E.bPendingRemove) continue;
		const float Dsq = (E.Pos - Game->PlayerPos).SizeSquared();
		if (Dsq < MinDistSq)
		{
			MinDistSq = Dsq;
			BaseDir   = (E.Pos - Game->PlayerPos).GetSafeNormal();
		}
	}

	for (int32 i = 0; i < EffAmount; ++i)
	{
		// Amount 本を扇状にわずかにばらけて発射（1本ならそのまま）
		FVector2D Dir = BaseDir;
		if (EffAmount > 1)
		{
			const float SpreadAngle = (static_cast<float>(i) / (EffAmount - 1) - 0.5f) * 0.35f;  // ±10度程度
			const float C = FMath::Cos(SpreadAngle);
			const float S = FMath::Sin(SpreadAngle);
			Dir = FVector2D(BaseDir.X * C - BaseDir.Y * S, BaseDir.X * S + BaseDir.Y * C);
		}

		FProjectileState P;
		P.Pos               = Game->PlayerPos;
		P.Vel               = Dir * EffSpeed;
		P.Radius            = FSimRadius(8.f);
		P.Damage            = FDamage(EffDamage);
		P.WeaponType        = WeaponType;
		P.WeaponSlotIdx     = SlotIdx;
		P.LifeTime          = FProjectileLifeTime(LifeTime);
		P.bPiercing         = false;
		P.MaxPierceCount    = CachedPierce;  // Pierce=1 から最大2まで増加
		P.KnockbackStrength = SurvivorsGameConstants::KnockbackSim_1;  // Knockback=1
		WeaponComp->SpawnProjectile(P);
	}
}
