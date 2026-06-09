#include "Survivors/Logic/Weapons/Projectile/SurvivorsKnifeWeapon.h"

#include "Survivors/Logic/SurvivorsGame.h"
#include "Survivors/Logic/SurvivorsGameConstants.h"
#include "Survivors/Logic/Weapons/SurvivorsWeaponComponent.h"

void USurvivorsKnifeWeapon::OnLevelChanged(FWeaponLevel NewLevel)
{
	CacheParams();
}

void USurvivorsKnifeWeapon::CacheParams()
{
	const int32 Lv  = FMath::Clamp(Level.Value, 1, SurvivorsGameConstants::MaxWeaponLevel);
	const int32 Idx = Lv - 1;

	if (WeaponType == EWeaponType::ThousandEdge)
	{
		const SurvivorsGameConstants::FKnifeParams& P = SurvivorsGameConstants::ThousandEdgeTable[Idx];
		CachedDamage   = P.Damage;
		CachedCooldown = P.Cooldown;
		CachedSpeed    = P.Speed;
		CachedAmount   = P.Amount;
		CachedPierce   = P.Pierce;
	}
	else
	{
		const SurvivorsGameConstants::FKnifeParams& P = SurvivorsGameConstants::KnifeTable[Idx];
		CachedDamage   = P.Damage;
		CachedCooldown = P.Cooldown;
		CachedSpeed    = P.Speed;
		CachedAmount   = P.Amount;
		CachedPierce   = P.Pierce;
	}
}

void USurvivorsKnifeWeapon::Tick(float Dt)
{
	if (!Game || !WeaponComp) return;

	CooldownTimer.Value = FMath::Max(0.f, CooldownTimer.Value - Dt);
	if (!CooldownTimer.IsReady()) return;

	const FPassiveEffects& PE = GetPassiveEffects();
	CooldownTimer = FCooldownSeconds(CachedCooldown * PE.CooldownMult);

	const float EffDamage = CachedDamage * PE.DamageMult;
	const float EffSpeed  = CachedSpeed  * PE.SpeedMult;
	const int32 EffAmount = CachedAmount + static_cast<int32>(PE.ExtraAmount);
	const float LifeTime  = 1.5f * PE.DurationMult;

	// 最近傍敵の方向を前方として扇状に発射
	FVector2D BaseDir = FVector2D(1.f, 0.f);
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
		FVector2D Dir = BaseDir;
		if (EffAmount > 1)
		{
			// 均等に扇状配置（例: 3本なら -15, 0, +15 度）
			const float HalfSpread = 0.26f;  // ~15度
			const float Angle = (static_cast<float>(i) / (EffAmount - 1) - 0.5f) * 2.f * HalfSpread;
			const float C = FMath::Cos(Angle);
			const float S = FMath::Sin(Angle);
			Dir = FVector2D(BaseDir.X * C - BaseDir.Y * S, BaseDir.X * S + BaseDir.Y * C);
		}

		FProjectileState P;
		P.Pos               = Game->PlayerPos;
		P.Vel               = Dir * EffSpeed;
		P.Radius            = FSimRadius(6.f * PE.AreaMult);
		P.Damage            = FDamage(EffDamage);
		P.WeaponType        = WeaponType;
		P.WeaponSlotIdx     = SlotIdx;
		P.LifeTime          = FProjectileLifeTime(LifeTime);
		P.bPiercing         = false;
		P.MaxPierceCount    = CachedPierce;  // Pierce=1 から最大3まで増加
		P.KnockbackStrength = SurvivorsGameConstants::KnockbackSim_Half;  // Knockback=0.5
		WeaponComp->SpawnProjectile(P);
	}
}
