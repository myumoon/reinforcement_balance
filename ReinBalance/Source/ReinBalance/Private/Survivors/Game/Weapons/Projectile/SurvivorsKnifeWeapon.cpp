#include "Survivors/Game/Weapons/Projectile/SurvivorsKnifeWeapon.h"

#include "Survivors/Game/SurvivorsGame.h"
#include "Survivors/Game/SurvivorsGameConstants.h"
#include "Survivors/Game/Weapons/SurvivorsWeaponComponent.h"

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

	if (!Game->PlayerVel.IsNearlyZero())
	{
		LastFacingDir = Game->PlayerVel.GetSafeNormal();
	}

	CooldownTimer.Value = FMath::Max(0.f, CooldownTimer.Value - Dt);

	if (PendingKnifeShots > 0)
	{
		KnifeBurstTimer -= Dt;
		while (PendingKnifeShots > 0 && KnifeBurstTimer <= 0.f)
		{
			SpawnKnifeShot();
			--PendingKnifeShots;
			if (PendingKnifeShots > 0)
			{
				KnifeBurstTimer += 0.10f;
			}
		}
	}

	if (!CooldownTimer.IsReady() || PendingKnifeShots > 0) return;

	StartBurst();
}

void USurvivorsKnifeWeapon::StartBurst()
{
	const FPassiveEffects& PE = GetPassiveEffects();
	CooldownTimer = FCooldownSeconds(CachedCooldown * PE.CooldownMult);

	BurstDamage = CachedDamage * PE.DamageMult;
	BurstSpeed = CachedSpeed * PE.SpeedMult;
	BurstRadius = 6.f * PE.AreaMult;
	BurstLifeTime = 1.5f * PE.DurationMult;
	BurstPierce = CachedPierce;
	PendingKnifeShots = FMath::Max(1, CachedAmount + static_cast<int32>(PE.ExtraAmount));
	KnifeBurstTimer = 0.f;
	if (PendingKnifeShots > 0)
	{
		SpawnKnifeShot();
		--PendingKnifeShots;
		KnifeBurstTimer = (PendingKnifeShots > 0) ? 0.10f : 0.f;
	}
}

void USurvivorsKnifeWeapon::SpawnKnifeShot()
{
	const FVector2D Dir  = LastFacingDir.IsNearlyZero() ? FVector2D(1.f, 0.f) : LastFacingDir.GetSafeNormal();
	// 発射方向に垂直な方向へ PlayerRadius 以内でランダムにスポーン位置をずらす（動画観測: 左右ブレ）
	const FVector2D Perp   = FVector2D(-Dir.Y, Dir.X);
	const float     Offset = Game->RandStream.FRandRange(-Game->PlayerRadius, Game->PlayerRadius);

	FProjectileState P;
	P.Pos               = Game->PlayerPos + Perp * Offset;
	P.Vel               = Dir * BurstSpeed;
	P.Radius            = FSimRadius(BurstRadius);
	P.Damage            = FDamage(BurstDamage);
	P.WeaponType        = WeaponType;
	P.WeaponSlotIdx     = SlotIdx;
	P.LifeTime          = FProjectileLifeTime(BurstLifeTime);
	P.bPiercing         = false;
	P.MaxPierceCount    = BurstPierce;  // Pierce=1 から最大3まで増加
	P.KnockbackStrength = SurvivorsGameConstants::KnockbackSim_Half;  // Knockback=0.5
	WeaponComp->SpawnProjectile(P);
}
