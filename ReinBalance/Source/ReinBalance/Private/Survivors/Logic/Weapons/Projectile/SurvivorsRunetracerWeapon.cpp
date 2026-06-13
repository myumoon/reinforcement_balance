#include "Survivors/Logic/Weapons/Projectile/SurvivorsRunetracerWeapon.h"

#include "Survivors/Logic/SurvivorsCollisionComponent.h"
#include "Survivors/Logic/SurvivorsGame.h"
#include "Survivors/Logic/SurvivorsGameConstants.h"
#include "Survivors/Logic/Weapons/SurvivorsWeaponComponent.h"

void USurvivorsRunetracerWeapon::OnLevelChanged(FWeaponLevel NewLevel)
{
	CacheParams();
}

void USurvivorsRunetracerWeapon::CacheParams()
{
	const int32 Lv  = FMath::Clamp(Level.Value, 1, SurvivorsGameConstants::MaxWeaponLevel);
	const int32 Idx = Lv - 1;

	if (WeaponType == EWeaponType::NoFuture)
	{
		const SurvivorsGameConstants::FRunetracerParams& P = SurvivorsGameConstants::NoFutureTable[Idx];
		CachedDamage    = P.Damage;
		CachedCooldown  = P.Cooldown;
		CachedSpeed     = P.Speed;
		CachedDuration  = P.Duration;
		CachedAmount    = P.Amount;
		CachedMaxBounce = P.MaxBounce;
	}
	else
	{
		const SurvivorsGameConstants::FRunetracerParams& P = SurvivorsGameConstants::RunetracerTable[Idx];
		CachedDamage    = P.Damage;
		CachedCooldown  = P.Cooldown;
		CachedSpeed     = P.Speed;
		CachedDuration  = P.Duration;
		CachedAmount    = P.Amount;
		CachedMaxBounce = P.MaxBounce;
	}
}

void USurvivorsRunetracerWeapon::Tick(float Dt)
{
	if (!Game || !WeaponComp) return;

	// --- 既存プロジェクタイルの壁反射処理 ---
	USurvivorsCollisionComponent* CollComp = Game->CollisionComponent;
	if (CollComp)
	{
		WeaponComp->UpdateProjectilesBySlot(SlotIdx, Dt, [CollComp](FProjectileState& P, float InDt) -> bool
		{
			if (P.BounceCount.HasBounces())
			{
				FVector2D NewPos = P.Pos;
				FVector2D NewVel = P.Vel;
				if (CollComp->ReflectOffWall(NewPos, NewVel, P.Radius.Value))
				{
					P.Pos = NewPos;
					P.Vel = NewVel;
					P.BounceCount.Consume();
				}
			}
			return true;
		});
	}

	// --- 順次発射バースト（wiki/OBSERVED: ~0.2s projectile interval）---
	if (PendingRuneShots > 0)
	{
		RuneBurstTimer -= Dt;
		while (PendingRuneShots > 0 && RuneBurstTimer <= 0.f)
		{
			SpawnRuneShot();
			--PendingRuneShots;
			RuneBurstTimer += SurvivorsGameConstants::RunetracerProjectileInterval;
		}
	}

	// --- クールダウン ---
	CooldownTimer.Value = FMath::Max(0.f, CooldownTimer.Value - Dt);
	if (!CooldownTimer.IsReady() || PendingRuneShots > 0) return;

	StartBurst();
}

void USurvivorsRunetracerWeapon::StartBurst()
{
	const FPassiveEffects& PE = GetPassiveEffects();

	const float CooldownInterval = (WeaponType == EWeaponType::NoFuture)
		? CachedCooldown * PE.CooldownMult + CachedDuration
		: CachedCooldown * PE.CooldownMult;
	CooldownTimer = FCooldownSeconds(CooldownInterval);

	BurstDamage    = CachedDamage   * PE.DamageMult;
	BurstSpeed     = CachedSpeed    * PE.SpeedMult;
	BurstDuration  = CachedDuration * PE.DurationMult;
	BurstRadius    = 10.f           * PE.AreaMult;
	BurstMaxBounce = CachedMaxBounce;

	PendingRuneShots = FMath::Max(1, CachedAmount + static_cast<int32>(PE.ExtraAmount));
	RuneBurstTimer   = 0.f;

	if (PendingRuneShots > 0)
	{
		SpawnRuneShot();
		--PendingRuneShots;
		RuneBurstTimer = (PendingRuneShots > 0) ? SurvivorsGameConstants::RunetracerProjectileInterval : 0.f;
	}
}

void USurvivorsRunetracerWeapon::SpawnRuneShot()
{
	// 各弾のランダム方向は発射時点で決定（OBSERVED: weapon_runetracer.md）
	const float Angle = Game->RandStream.FRand() * TWO_PI;
	const FVector2D Dir = FVector2D(FMath::Cos(Angle), FMath::Sin(Angle));

	FProjectileState P;
	P.Pos               = Game->PlayerPos;
	P.Vel               = Dir * BurstSpeed;
	P.Radius            = FSimRadius(BurstRadius);
	P.Damage            = FDamage(BurstDamage);
	P.WeaponType        = WeaponType;
	P.WeaponSlotIdx     = SlotIdx;
	P.LifeTime          = FProjectileLifeTime(BurstDuration);
	P.bPiercing         = true;   // AoE: 無限貫通
	P.BounceCount       = FBounceCount(BurstMaxBounce);
	P.KnockbackStrength = SurvivorsGameConstants::KnockbackSim_1;
	WeaponComp->SpawnProjectile(P);
}
