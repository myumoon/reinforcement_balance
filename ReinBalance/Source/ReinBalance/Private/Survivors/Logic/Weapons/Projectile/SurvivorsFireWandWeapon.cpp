#include "Survivors/Logic/Weapons/Projectile/SurvivorsFireWandWeapon.h"

#include "Survivors/Logic/SurvivorsGame.h"
#include "Survivors/Logic/SurvivorsGameConstants.h"
#include "Survivors/Logic/Weapons/SurvivorsWeaponComponent.h"

void USurvivorsFireWandWeapon::OnLevelChanged(FWeaponLevel NewLevel)
{
	CacheParams();
}

void USurvivorsFireWandWeapon::CacheParams()
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
	}
	else
	{
		const SurvivorsGameConstants::FFireWandParams& P = SurvivorsGameConstants::FireWandTable[Idx];
		CachedDamage          = P.Damage;
		CachedCooldown        = P.Cooldown;
		CachedSpeed           = P.Speed;
		CachedExplosionRadius = P.ExplosionRadius;
	}
}

void USurvivorsFireWandWeapon::Tick(float Dt)
{
	if (!Game || !WeaponComp) return;

	const FPassiveEffects& PE = GetPassiveEffects();

	// 寿命切れプロジェクタイルを GroundZone 爆発に変換（UpdateProjectilesBySlot で処理）
	const float EffExplosionRadius = CachedExplosionRadius * PE.AreaMult;
	const float EffDamage          = CachedDamage * PE.DamageMult;
	const float EffDuration        = 0.2f * PE.DurationMult;  // 爆発は短時間

	// 寿命切れ弾を検出して爆発 GroundZone 生成
	// （WeaponComponent::TickProjectiles が寿命切れを削除する前に処理する必要があるが、
	//  Tick の呼び出しは TickWeapons 内で TickProjectiles より先）
	WeaponComp->UpdateProjectilesBySlot(SlotIdx, 0.f, [&](FProjectileState& P, float) -> bool
	{
		if (P.LifeTime.IsExpired())
		{
			// 爆発 GroundZone 生成
			FGroundZoneState Z;
			Z.Pos          = P.Pos;
			Z.Radius       = EffExplosionRadius;
			Z.Damage       = EffDamage;
			Z.LifeTime     = EffDuration;
			Z.HitCooldown  = EffDuration;  // 1 回だけヒット
			Z.WeaponSlotIdx = SlotIdx;
			Z.WeaponType   = WeaponType;
			WeaponComp->SpawnGroundZone(Z);
			return false;  // 削除
		}
		return true;
	});

	// クールダウン
	CooldownTimer.Value = FMath::Max(0.f, CooldownTimer.Value - Dt);
	if (!CooldownTimer.IsReady()) return;

	CooldownTimer = FCooldownSeconds(CachedCooldown * PE.CooldownMult);

	const float EffSpeed    = CachedSpeed * PE.SpeedMult;
	const float LifeTime    = 1.2f * PE.DurationMult;

	// 最近傍敵の方向へ発射
	FVector2D Dir = FVector2D(0.f, 1.f);
	float MinDistSq = MAX_FLT;
	for (const FEnemyState& E : Game->Enemies)
	{
		if (E.bPendingRemove) continue;
		const float Dsq = (E.Pos - Game->PlayerPos).SizeSquared();
		if (Dsq < MinDistSq)
		{
			MinDistSq = Dsq;
			Dir = (E.Pos - Game->PlayerPos).GetSafeNormal();
		}
	}

	FProjectileState P;
	P.Pos           = Game->PlayerPos;
	P.Vel           = Dir * EffSpeed;
	P.Radius        = FSimRadius(8.f);
	P.Damage        = FDamage(0.f);  // 直撃ダメージなし（爆発で処理）
	P.WeaponType    = WeaponType;
	P.WeaponSlotIdx = SlotIdx;
	P.LifeTime      = FProjectileLifeTime(LifeTime);
	P.bPiercing     = false;
	WeaponComp->SpawnProjectile(P);
}
