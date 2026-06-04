#include "Survivors/Logic/Weapons/Projectile/SurvivorsWhipWeapon.h"

#include "Survivors/Logic/SurvivorsCollisionComponent.h"
#include "Survivors/Logic/SurvivorsGame.h"
#include "Survivors/Logic/SurvivorsGameConstants.h"
#include "Survivors/Logic/Weapons/SurvivorsWeaponComponent.h"

void USurvivorsWhipWeapon::OnLevelChanged(FWeaponLevel NewLevel)
{
	CacheParams();
}

void USurvivorsWhipWeapon::CacheParams()
{
	const int32 Lv  = FMath::Clamp(Level.Value, 1, SurvivorsGameConstants::MaxWeaponLevel);
	const int32 Idx = Lv - 1;

	if (WeaponType == EWeaponType::BloodyTear)
	{
		const SurvivorsGameConstants::FWhipParams& P = SurvivorsGameConstants::BloodyTearTable[Idx];
		CachedDamage   = P.Damage;
		CachedCooldown = P.Cooldown;
		CachedWidth    = P.Width;
		CachedHeight   = P.Height;
	}
	else
	{
		const SurvivorsGameConstants::FWhipParams& P = SurvivorsGameConstants::WhipTable[Idx];
		CachedDamage   = P.Damage;
		CachedCooldown = P.Cooldown;
		CachedWidth    = P.Width;
		CachedHeight   = P.Height;
	}
}

void USurvivorsWhipWeapon::Tick(float Dt)
{
	if (!Game || !WeaponComp) return;

	CooldownTimer.Value = FMath::Max(0.f, CooldownTimer.Value - Dt);

	if (!CooldownTimer.IsReady()) return;

	const FPassiveEffects& PE = GetPassiveEffects();
	const float EffCooldown = CachedCooldown * PE.CooldownMult;
	CooldownTimer = FCooldownSeconds(EffCooldown);

	const float EffDamage = CachedDamage * PE.DamageMult;
	const float EffWidth  = CachedWidth  * PE.AreaMult;
	const float EffHeight = CachedHeight * PE.AreaMult;
	const float LifeTime  = 0.20f * PE.DurationMult;

	// 左右 2 方向に piercing 矩形プロジェクタイル（近似: 横長楕円として Radius=Width で扱う）
	// 左方向
	{
		FProjectileState P;
		P.Pos         = Game->PlayerPos;
		P.Vel         = FVector2D(-EffWidth * 2.f / LifeTime, 0.f);
		P.Radius      = FSimRadius(EffHeight);
		P.Damage      = FDamage(EffDamage);
		P.WeaponType  = WeaponType;
		P.WeaponSlotIdx = SlotIdx;
		P.LifeTime    = FProjectileLifeTime(LifeTime);
		P.bPiercing   = true;
		WeaponComp->SpawnProjectile(P);
	}
	// 右方向
	{
		FProjectileState P;
		P.Pos         = Game->PlayerPos;
		P.Vel         = FVector2D(EffWidth * 2.f / LifeTime, 0.f);
		P.Radius      = FSimRadius(EffHeight);
		P.Damage      = FDamage(EffDamage);
		P.WeaponType  = WeaponType;
		P.WeaponSlotIdx = SlotIdx;
		P.LifeTime    = FProjectileLifeTime(LifeTime);
		P.bPiercing   = true;
		WeaponComp->SpawnProjectile(P);
	}
}

void USurvivorsWhipWeapon::ComputeHits(USurvivorsCollisionComponent* CollComp, FSurvivorsHitFrame& HitFrame)
{
	// プロジェクタイルの当たり判定は WeaponComponent::ComputeProjectileHits で一括処理されるため
	// Whip では何もしない（基底クラス実装と同様）
}
