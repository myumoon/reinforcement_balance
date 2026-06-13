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
	// 1.5s(旧値)では画面内最大距離(400u)を140u/sで到達できない(2.86s必要)。
	// on-screen targeting 前提で画面横断(800u÷140u/s≈5.7s)をカバーする寿命に延長。
	BurstLifeTime = 6.0f * PE.DurationMult;
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
	// 発射時点で画面内最近傍敵を探索（画面外の敵は対象外）。
	// 画面内に敵がいない場合はランダム方向に発射する。
	FVector2D Dir = FVector2D::ZeroVector;  // ZeroVector = 敵未発見を示す
	float MinDistSq = MAX_FLT;
	for (const FEnemyState& E : Game->Enemies)
	{
		if (E.bPendingRemove) continue;
		if (!Game->IsOnScreen(E.Pos)) continue;  // 画面外の敵は無視
		const float Dsq = (E.Pos - Game->PlayerPos).SizeSquared();
		if (Dsq < MinDistSq)
		{
			MinDistSq = Dsq;
			Dir       = (E.Pos - Game->PlayerPos).GetSafeNormal();
		}
	}
	// 画面内に敵がいない場合はランダム方向
	if (Dir.IsNearlyZero())
	{
		const float Angle = Game->RandStream.FRand() * TWO_PI;
		Dir = FVector2D(FMath::Cos(Angle), FMath::Sin(Angle));
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
