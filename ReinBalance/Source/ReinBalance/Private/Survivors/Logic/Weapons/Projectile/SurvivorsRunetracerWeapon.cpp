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
		CachedMaxBounce = P.MaxBounce;
	}
	else
	{
		const SurvivorsGameConstants::FRunetracerParams& P = SurvivorsGameConstants::RunetracerTable[Idx];
		CachedDamage    = P.Damage;
		CachedCooldown  = P.Cooldown;
		CachedSpeed     = P.Speed;
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

	// --- クールダウン ---
	CooldownTimer.Value = FMath::Max(0.f, CooldownTimer.Value - Dt);
	if (!CooldownTimer.IsReady()) return;

	const FPassiveEffects& PE = GetPassiveEffects();
	CooldownTimer = FCooldownSeconds(CachedCooldown * PE.CooldownMult);

	const float EffDamage = CachedDamage * PE.DamageMult;
	const float EffSpeed  = CachedSpeed  * PE.SpeedMult;
	const float LifeTime  = 5.0f * PE.DurationMult;

	// ランダム方向に発射
	const float Angle = Game->RandStream.FRand() * TWO_PI;
	const FVector2D Dir = FVector2D(FMath::Cos(Angle), FMath::Sin(Angle));

	FProjectileState P;
	P.Pos           = Game->PlayerPos;
	P.Vel           = Dir * EffSpeed;
	P.Radius        = FSimRadius(10.f);
	P.Damage        = FDamage(EffDamage);
	P.WeaponType    = WeaponType;
	P.WeaponSlotIdx = SlotIdx;
	P.LifeTime      = FProjectileLifeTime(LifeTime);
	P.bPiercing         = true;  // AoE: 無限貫通
	P.BounceCount       = FBounceCount(CachedMaxBounce + static_cast<int32>(PE.ExtraAmount));
	P.KnockbackStrength = SurvivorsGameConstants::KnockbackSim_1;  // Knockback=1
	WeaponComp->SpawnProjectile(P);
}
