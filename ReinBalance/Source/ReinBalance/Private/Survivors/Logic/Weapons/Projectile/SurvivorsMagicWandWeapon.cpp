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

	// 順次発射: 0.1s 間隔でバースト残弾を発射（wiki: projectile interval = 0.1s）
	if (PendingWandShots > 0)
	{
		WandBurstTimer -= Dt;
		while (PendingWandShots > 0 && WandBurstTimer <= 0.f)
		{
			SpawnWandShot();
			--PendingWandShots;
			if (PendingWandShots > 0)
				WandBurstTimer += 0.10f;
		}
	}

	CooldownTimer.Value = FMath::Max(0.f, CooldownTimer.Value - Dt);
	if (!CooldownTimer.IsReady() || PendingWandShots > 0) return;

	StartBurst();
}

void USurvivorsMagicWandWeapon::StartBurst()
{
	const FPassiveEffects& PE = GetPassiveEffects();
	CooldownTimer = FCooldownSeconds(CachedCooldown * PE.CooldownMult);

	BurstDamage   = CachedDamage * PE.DamageMult;
	BurstSpeed    = CachedSpeed  * PE.SpeedMult;
	BurstRadius   = 8.f * PE.AreaMult;
	BurstLifeTime = 1.5f * PE.DurationMult;
	BurstPierce   = CachedPierce;

	PendingWandShots = CachedAmount + static_cast<int32>(PE.ExtraAmount);
	WandBurstTimer   = 0.f;

	if (PendingWandShots > 0)
	{
		SpawnWandShot();
		--PendingWandShots;
		WandBurstTimer = (PendingWandShots > 0) ? 0.10f : 0.f;
	}
}

void USurvivorsMagicWandWeapon::SpawnWandShot()
{
	// 発射時点で最近傍敵を個別に探索（wiki: "Each missile aims at the closest enemy at the moment it is spawned"）
	FVector2D Dir = FVector2D(0.f, 1.f);
	float MinDistSq = MAX_FLT;
	for (const FEnemyState& E : Game->Enemies)
	{
		if (E.bPendingRemove) continue;
		const float Dsq = (E.Pos - Game->PlayerPos).SizeSquared();
		if (Dsq < MinDistSq)
		{
			MinDistSq = Dsq;
			Dir       = (E.Pos - Game->PlayerPos).GetSafeNormal();
		}
	}

	FProjectileState P;
	P.Pos               = Game->PlayerPos;
	P.Vel               = Dir * BurstSpeed;
	P.Radius            = FSimRadius(BurstRadius);
	P.Damage            = FDamage(BurstDamage);
	P.WeaponType        = WeaponType;
	P.WeaponSlotIdx     = SlotIdx;
	P.LifeTime          = FProjectileLifeTime(BurstLifeTime);
	P.bPiercing         = false;
	P.MaxPierceCount    = BurstPierce;
	P.KnockbackStrength = SurvivorsGameConstants::KnockbackSim_1;
	WeaponComp->SpawnProjectile(P);
}
