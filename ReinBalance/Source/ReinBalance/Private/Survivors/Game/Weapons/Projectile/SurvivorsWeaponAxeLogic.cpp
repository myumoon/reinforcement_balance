#include "Survivors/Game/Weapons/Projectile/SurvivorsWeaponAxeLogic.h"
#include "Survivors/Game/SurvivorsGameLogic.h"
#include "Survivors/Game/SurvivorsGameConstants.h"

void FSurvivorsWeaponAxeLogic::OnLevelChanged(FWeaponLevel NewLevel)
{
	CacheParams();
}

void FSurvivorsWeaponAxeLogic::CacheParams()
{
	const int32 Lv  = FMath::Clamp(Level.Value, 1, SurvivorsGameConstants::MaxWeaponLevel);
	const int32 Idx = Lv - 1;

	if (WeaponType == EWeaponType::DeathSpiral)
	{
		const SurvivorsGameConstants::FAxeParams& P = SurvivorsGameConstants::DeathSpiralTable[Idx];
		CachedDamage    = P.Damage;
		CachedCooldown  = P.Cooldown;
		CachedSpeed     = P.Speed;
		CachedArcHeight = P.ArcHeight;
		CachedAmount    = P.Amount;
		CachedPierce    = P.Pierce;
	}
	else
	{
		const SurvivorsGameConstants::FAxeParams& P = SurvivorsGameConstants::AxeTable[Idx];
		CachedDamage    = P.Damage;
		CachedCooldown  = P.Cooldown;
		CachedSpeed     = P.Speed;
		CachedArcHeight = P.ArcHeight;
		CachedAmount    = P.Amount;
		CachedPierce    = P.Pierce;
	}
}

void FSurvivorsWeaponAxeLogic::Tick(float Dt)
{
	if (!Logic) return;

	const float FieldLimit  = Logic->CurrentConfig.FieldHalfSize + 20.f;
	const float FieldBottom = -FieldLimit;
	Logic->UpdateProjectilesBySlot(SlotIdx, Dt, [FieldBottom, FieldLimit](FProjectileState& P, float InDt) -> bool
	{
		P.Vel.Y += P.AngleRad.Value * InDt;
		if (P.WeaponType == EWeaponType::DeathSpiral)
			return FMath::Abs(P.Pos.X) <= FieldLimit && FMath::Abs(P.Pos.Y) <= FieldLimit;
		return P.Pos.Y > FieldBottom;
	});

	if (WeaponType == EWeaponType::DeathSpiral)
	{
		CooldownTimer.Value = FMath::Max(0.f, CooldownTimer.Value - Dt);
		if (!CooldownTimer.IsReady()) return;

		const FPassiveEffects& PE = GetPassiveEffects();
		CooldownTimer = FCooldownSeconds(CachedCooldown * PE.CooldownMult);

		const float EffDamage = CachedDamage * PE.DamageMult;
		const float EffSpeed  = CachedSpeed  * PE.SpeedMult;
		const int32 EffAmount = FMath::Max(1, CachedAmount + static_cast<int32>(PE.ExtraAmount));

		for (int32 i = 0; i < EffAmount; ++i)
		{
			const float Angle = TWO_PI * static_cast<float>(i) / static_cast<float>(EffAmount);
			FProjectileState P;
			P.Pos               = Logic->PlayerPos;
			P.Vel               = FVector2D(FMath::Cos(Angle), FMath::Sin(Angle)) * EffSpeed;
			P.Radius            = FSimRadius(12.f * PE.AreaMult);
			P.Damage            = FDamage(EffDamage);
			P.WeaponType        = WeaponType;
			P.WeaponSlotIdx     = SlotIdx;
			P.LifeTime          = FProjectileLifeTime(20.f);
			P.bPiercing         = true;
			P.MaxPierceCount    = CachedPierce;
			P.KnockbackStrength = SurvivorsGameConstants::KnockbackSim_1;
			P.AngleRad          = FOrbitAngleRad(0.f);
			Logic->SpawnProjectile(P);
		}
		return;
	}

	if (PendingAxeShots > 0)
	{
		AxeBurstTimer -= Dt;
		while (PendingAxeShots > 0 && AxeBurstTimer <= 0.f)
		{
			SpawnAxeShot();
			--PendingAxeShots;
			AxeBurstTimer += SurvivorsGameConstants::AxeProjectileInterval;
		}
	}

	CooldownTimer.Value = FMath::Max(0.f, CooldownTimer.Value - Dt);
	if (!CooldownTimer.IsReady() || PendingAxeShots > 0) return;

	StartBurst();
}

void FSurvivorsWeaponAxeLogic::StartBurst()
{
	const FPassiveEffects& PE = GetPassiveEffects();
	CooldownTimer = FCooldownSeconds(CachedCooldown * PE.CooldownMult);

	BurstDamage    = CachedDamage * PE.DamageMult;
	BurstSpeed     = CachedSpeed  * PE.SpeedMult;
	BurstRadius    = 10.f * SurvivorsGameConstants::AxeAreaScaleFactor * PE.AreaMult;
	BurstArcHeight = CachedArcHeight * PE.SpeedMult;
	BurstPierce    = CachedPierce;
	BurstHorizDir  = 0.f;

	PendingAxeShots      = FMath::Max(1, CachedAmount + static_cast<int32>(PE.ExtraAmount));
	AxeBurstTimer        = 0.f;
	BurstShotsFiredCount = 0;

	if (PendingAxeShots > 0)
	{
		SpawnAxeShot();
		--PendingAxeShots;
		AxeBurstTimer = (PendingAxeShots > 0) ? SurvivorsGameConstants::AxeProjectileInterval : 0.f;
	}
}

void FSurvivorsWeaponAxeLogic::SpawnAxeShot()
{
	const float InitVelY = BurstSpeed;
	const float HalfTime = (InitVelY > KINDA_SMALL_NUMBER) ? (BurstArcHeight / InitVelY) : 0.333f;
	const float GravityY = -InitVelY / HalfTime;

	const float RandomOffset = (BurstShotsFiredCount == 0)
		? 0.f
		: Logic->RandStream.FRandRange(
			-SurvivorsGameConstants::AxeRandomConeHalfAngle,
			 SurvivorsGameConstants::AxeRandomConeHalfAngle);
	const float HorizScale = FMath::Sin(RandomOffset);

	FProjectileState P;
	P.Pos               = Logic->PlayerPos;
	P.Vel               = FVector2D(BurstSpeed * HorizScale, BurstSpeed * FMath::Cos(RandomOffset));
	P.Radius            = FSimRadius(BurstRadius);
	P.Damage            = FDamage(BurstDamage);
	P.WeaponType        = WeaponType;
	P.WeaponSlotIdx     = SlotIdx;
	P.LifeTime          = FProjectileLifeTime(20.f);
	P.bPiercing         = false;
	P.MaxPierceCount    = BurstPierce;
	P.KnockbackStrength = SurvivorsGameConstants::KnockbackSim_1;
	P.AngleRad          = FOrbitAngleRad(GravityY);
	Logic->SpawnProjectile(P);

	++BurstShotsFiredCount;
}
