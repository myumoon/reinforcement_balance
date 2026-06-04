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

	for (int32 i = 0; i < EffAmount; ++i)
	{
		// 配置位置: ランダムにプレイヤー周辺（または最近傍敵の足元）
		// 簡略: 最近傍敵の位置に生成、複数なら近い順
		FVector2D DropPos = Game->PlayerPos;
		int32 FoundCount = 0;
		for (const FEnemyState& E : Game->Enemies)
		{
			if (E.bPendingRemove) continue;
			if (FoundCount == i)
			{
				DropPos = E.Pos;
				break;
			}
			++FoundCount;
		}

		FGroundZoneState Z;
		Z.Pos          = DropPos;
		Z.Radius       = EffRadius;
		Z.Damage       = EffDamage;
		Z.LifeTime     = EffDuration;
		Z.HitCooldown  = 0.5f;
		Z.WeaponSlotIdx = SlotIdx;
		Z.WeaponType   = WeaponType;
		WeaponComp->SpawnGroundZone(Z);
	}
}
