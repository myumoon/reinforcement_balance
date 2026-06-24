#include "Survivors/Game/Weapons/Projectile/SurvivorsWeaponRunetracer.h"

#include "Survivors/Game/SurvivorsCollisionComponent.h"
#include "Survivors/Game/SurvivorsGame.h"
#include "Survivors/SurvivorsGameConstants.h"
#include "Survivors/Game/Weapons/SurvivorsWeaponComponent.h"

void USurvivorsWeaponRunetracer::OnLevelChanged(FWeaponLevel NewLevel)
{
	CacheParams();
}

void USurvivorsWeaponRunetracer::CacheParams()
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

void USurvivorsWeaponRunetracer::Tick(float Dt)
{
	if (!Game || !WeaponComp) return;

	// --- 壁反射 + スクリーンエッジバウンス ---
	// ReflectOffWall は AWallActor のみ対象。それに加えて可視スクリーン端でも反射する。
	USurvivorsCollisionComponent* CollComp = Game->CollisionComponent;
	const FVector2D PlayerPos = Game->PlayerPos;
	const float ScrW = SurvivorsGameConstants::ScreenHalfWidthU;
	const float ScrH = SurvivorsGameConstants::ScreenHalfHeightU;
	WeaponComp->UpdateProjectilesBySlot(SlotIdx, Dt, [CollComp, PlayerPos, ScrW, ScrH](FProjectileState& P, float InDt) -> bool
	{
		if (!P.BounceCount.HasBounces()) return true;

		bool bBounced = false;

		// AWallActor 反射
		if (CollComp)
		{
			FVector2D NewPos = P.Pos;
			FVector2D NewVel = P.Vel;
			if (CollComp->ReflectOffWall(NewPos, NewVel, P.Radius.Value))
			{
				P.Pos    = NewPos;
				P.Vel    = NewVel;
				bBounced = true;
			}
		}

		// スクリーンエッジ反射（PlayerPos 相対で可視領域端）
		// TickProjectiles が Pos += Vel * Dt を実行するのは Tick() より後なので、
		// 次フレーム位置 = Pos + Vel * InDt で判定して Vel のみ修正する。
		if (!bBounced)
		{
			const float MinX = PlayerPos.X - ScrW;
			const float MaxX = PlayerPos.X + ScrW;
			const float MinY = PlayerPos.Y - ScrH;
			const float MaxY = PlayerPos.Y + ScrH;
			const FVector2D NextPos = P.Pos + P.Vel * InDt;
			const float     R       = P.Radius.Value;
			if      (NextPos.X + R > MaxX) { P.Vel.X = -FMath::Abs(P.Vel.X); bBounced = true; }
			else if (NextPos.X - R < MinX) { P.Vel.X =  FMath::Abs(P.Vel.X); bBounced = true; }
			if      (NextPos.Y + R > MaxY) { P.Vel.Y = -FMath::Abs(P.Vel.Y); bBounced = true; }
			else if (NextPos.Y - R < MinY) { P.Vel.Y =  FMath::Abs(P.Vel.Y); bBounced = true; }
		}

		if (bBounced) P.BounceCount.Consume();
		return true;
	});

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

void USurvivorsWeaponRunetracer::StartBurst()
{
	const FPassiveEffects& PE = GetPassiveEffects();

	const float CooldownInterval = (WeaponType == EWeaponType::NoFuture)
		? CachedCooldown * PE.CooldownMult + CachedDuration
		: CachedCooldown * PE.CooldownMult;
	CooldownTimer = FCooldownSeconds(CooldownInterval);

	BurstDamage    = CachedDamage   * PE.DamageMult;
	BurstSpeed     = CachedSpeed    * PE.SpeedMult;
	BurstDuration  = CachedDuration * PE.DurationMult;
	BurstRadius    = 6.7f           * PE.AreaMult;  // 10u × 2/3 ≈ 6.7u
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

void USurvivorsWeaponRunetracer::SpawnRuneShot()
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
