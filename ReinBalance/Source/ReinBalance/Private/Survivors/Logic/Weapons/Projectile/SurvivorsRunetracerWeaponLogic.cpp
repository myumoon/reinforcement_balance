#include "Survivors/Logic/Weapons/Projectile/SurvivorsRunetracerWeaponLogic.h"
#include "Survivors/Logic/SurvivorsGameLogic.h"
#include "Survivors/Logic/SurvivorsGameConstants.h"

void FSurvivorsRunetracerWeaponLogic::OnLevelChanged(FWeaponLevel NewLevel)
{
	CacheParams();
}

void FSurvivorsRunetracerWeaponLogic::CacheParams()
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

void FSurvivorsRunetracerWeaponLogic::Tick(float Dt)
{
	if (!Logic) return;

	const FVector2D PlayerPos = Logic->PlayerPos;
	const float ScrW = SurvivorsGameConstants::ScreenHalfWidthU;
	const float ScrH = SurvivorsGameConstants::ScreenHalfHeightU;

	Logic->UpdateProjectilesBySlot(SlotIdx, Dt, [this, PlayerPos, ScrW, ScrH](FProjectileState& P, float InDt) -> bool
	{
		if (!P.BounceCount.HasBounces()) return true;

		bool bBounced = false;

		FVector2D NewPos = P.Pos;
		FVector2D NewVel = P.Vel;
		if (Logic->ReflectOffWall(NewPos, NewVel, P.Radius.Value))
		{
			P.Pos    = NewPos;
			P.Vel    = NewVel;
			bBounced = true;
		}

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

	CooldownTimer.Value = FMath::Max(0.f, CooldownTimer.Value - Dt);
	if (!CooldownTimer.IsReady() || PendingRuneShots > 0) return;

	StartBurst();
}

void FSurvivorsRunetracerWeaponLogic::StartBurst()
{
	const FPassiveEffects& PE = GetPassiveEffects();

	const float CooldownInterval = (WeaponType == EWeaponType::NoFuture)
		? CachedCooldown * PE.CooldownMult + CachedDuration
		: CachedCooldown * PE.CooldownMult;
	CooldownTimer = FCooldownSeconds(CooldownInterval);

	BurstDamage    = CachedDamage   * PE.DamageMult;
	BurstSpeed     = CachedSpeed    * PE.SpeedMult;
	BurstDuration  = CachedDuration * PE.DurationMult;
	BurstRadius    = 6.7f           * PE.AreaMult;
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

void FSurvivorsRunetracerWeaponLogic::SpawnRuneShot()
{
	const float Angle = Logic->RandStream.FRand() * TWO_PI;
	const FVector2D Dir = FVector2D(FMath::Cos(Angle), FMath::Sin(Angle));

	FProjectileState P;
	P.Pos               = Logic->PlayerPos;
	P.Vel               = Dir * BurstSpeed;
	P.Radius            = FSimRadius(BurstRadius);
	P.Damage            = FDamage(BurstDamage);
	P.WeaponType        = WeaponType;
	P.WeaponSlotIdx     = SlotIdx;
	P.LifeTime          = FProjectileLifeTime(BurstDuration);
	P.bPiercing         = true;
	P.BounceCount       = FBounceCount(BurstMaxBounce);
	P.KnockbackStrength = SurvivorsGameConstants::KnockbackSim_1;
	Logic->SpawnProjectile(P);
}
