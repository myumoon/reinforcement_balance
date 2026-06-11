#include "Survivors/Logic/Weapons/Projectile/SurvivorsSantaWaterWeapon.h"

#include "Survivors/Logic/SurvivorsGame.h"
#include "Survivors/Logic/SurvivorsGameConstants.h"
#include "Survivors/Logic/Weapons/SurvivorsWeaponComponent.h"

void USurvivorsSantaWaterWeapon::OnLevelChanged(FWeaponLevel NewLevel)
{
	CacheParams();
}

void USurvivorsSantaWaterWeapon::CacheParams()
{
	const int32 Lv  = FMath::Clamp(Level.Value, 1, SurvivorsGameConstants::MaxWeaponLevel);
	const int32 Idx = Lv - 1;

	if (WeaponType == EWeaponType::LaBorra)
	{
		const SurvivorsGameConstants::FSantaWaterParams& P = SurvivorsGameConstants::LaBorraTable[Idx];
		CachedDamage   = P.Damage;
		CachedCooldown = P.Cooldown;
		CachedRadius   = P.Radius;
		CachedDuration = P.Duration;
		CachedAmount   = P.Amount;
	}
	else
	{
		const SurvivorsGameConstants::FSantaWaterParams& P = SurvivorsGameConstants::SantaWaterTable[Idx];
		CachedDamage   = P.Damage;
		CachedCooldown = P.Cooldown;
		CachedRadius   = P.Radius;
		CachedDuration = P.Duration;
		CachedAmount   = P.Amount;
	}
}

void USurvivorsSantaWaterWeapon::Tick(float Dt)
{
	if (!Game || !WeaponComp) return;

	CooldownTimer.Value = FMath::Max(0.f, CooldownTimer.Value - Dt);
	if (!CooldownTimer.IsReady()) return;

	const FPassiveEffects& PE = GetPassiveEffects();
	CooldownTimer = FCooldownSeconds(CachedCooldown * PE.CooldownMult);

	const float EffDamage   = CachedDamage   * PE.DamageMult;
	const float EffRadius   = CachedRadius   * PE.AreaMult;
	const float EffDuration = CachedDuration * PE.DurationMult;
	const int32 EffAmount   = CachedAmount   + static_cast<int32>(PE.ExtraAmount);

	TArray<int32> EnemyIdx;
	EnemyIdx.Reserve(Game->Enemies.Num());
	for (int32 EIdx = 0; EIdx < Game->Enemies.Num(); ++EIdx)
	{
		if (!Game->Enemies[EIdx].bPendingRemove) EnemyIdx.Add(EIdx);
	}
	EnemyIdx.Sort([&](int32 A, int32 B)
	{
		return FVector2D::DistSquared(Game->Enemies[A].Pos, Game->PlayerPos)
			 < FVector2D::DistSquared(Game->Enemies[B].Pos, Game->PlayerPos);
	});

	for (int32 i = 0; i < EffAmount; ++i)
	{
		FVector2D DropPos = Game->PlayerPos;
		if (EnemyIdx.IsValidIndex(i))
		{
			DropPos = Game->Enemies[EnemyIdx[i]].Pos;
		}

		FGroundZoneState Z;
		Z.Pos          = DropPos;
		Z.Radius       = EffRadius;
		Z.Damage       = EffDamage;
		Z.LifeTime     = SurvivorsGameConstants::SantaWaterWarningTime + EffDuration;
		Z.WarningTime  = SurvivorsGameConstants::SantaWaterWarningTime;
		Z.HitCooldown  = 0.5f;
		Z.WeaponSlotIdx = SlotIdx;
		Z.WeaponType   = WeaponType;
		Z.bIsWarning   = true;
		WeaponComp->SpawnGroundZone(Z);
	}
}
