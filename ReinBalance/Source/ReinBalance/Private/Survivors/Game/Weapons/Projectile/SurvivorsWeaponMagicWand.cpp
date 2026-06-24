#include "Survivors/Game/Weapons/Projectile/SurvivorsWeaponMagicWand.h"

#include "Survivors/Game/SurvivorsCollisionComponent.h"
#include "Survivors/Game/SurvivorsGame.h"
#include "Survivors/Logic/SurvivorsGameConstants.h"
#include "Survivors/Game/Weapons/SurvivorsWeaponComponent.h"

void USurvivorsWeaponMagicWand::OnLevelChanged(FWeaponLevel NewLevel)
{
	CacheParams();
}

void USurvivorsWeaponMagicWand::CacheParams()
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

void USurvivorsWeaponMagicWand::Tick(float Dt)
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

void USurvivorsWeaponMagicWand::StartBurst()
{
	const FPassiveEffects& PE = GetPassiveEffects();
	CooldownTimer = FCooldownSeconds(CachedCooldown * PE.CooldownMult);

	BurstDamage   = CachedDamage * PE.DamageMult;
	BurstSpeed    = CachedSpeed  * PE.SpeedMult;
	BurstRadius   = 8.f * PE.AreaMult;
	// on-screen targeting 前提で画面横断(800u÷140u/s≈5.7s)をカバーする寿命に延長。
	BurstLifeTime = 6.0f * PE.DurationMult;
	BurstPierce   = CachedPierce;

	// 画面内に敵がいない場合のバースト共通ランダム方向を事前決定（全弾が同一方向に飛ぶ）
	bool bHasOnScreenEnemy = false;
	for (const FEnemyState& E : Game->Enemies)
	{
		if (!E.bPendingRemove && Game->IsOnScreen(E.Pos)) { bHasOnScreenEnemy = true; break; }
	}
	if (!bHasOnScreenEnemy)
	{
		const float Angle = Game->RandStream.FRand() * TWO_PI;
		BurstNoTargetDir = FVector2D(FMath::Cos(Angle), FMath::Sin(Angle));
	}

	PendingWandShots = CachedAmount + static_cast<int32>(PE.ExtraAmount);
	WandBurstTimer   = 0.f;

	if (PendingWandShots > 0)
	{
		SpawnWandShot();
		--PendingWandShots;
		WandBurstTimer = (PendingWandShots > 0) ? 0.10f : 0.f;
	}
}

void USurvivorsWeaponMagicWand::SpawnWandShot()
{
	// 発射時点で画面内最近傍敵を探索（画面外の敵は対象外）。
	// 画面内に敵がいない場合は StartBurst() で決定した BurstNoTargetDir を使う（全弾同一方向）。
	FVector2D Dir = FVector2D::ZeroVector;
	float MinDistSq = MAX_FLT;
	for (const FEnemyState& E : Game->Enemies)
	{
		if (E.bPendingRemove) continue;
		if (!Game->IsOnScreen(E.Pos)) continue;
		const float Dsq = (E.Pos - Game->PlayerPos).SizeSquared();
		if (Dsq < MinDistSq)
		{
			MinDistSq = Dsq;
			Dir       = (E.Pos - Game->PlayerPos).GetSafeNormal();
		}
	}
	if (Dir.IsNearlyZero())
		Dir = BurstNoTargetDir;

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
