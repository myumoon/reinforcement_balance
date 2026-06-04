#include "Survivors/Logic/Weapons/Projectile/SurvivorsGarlicWeapon.h"

#include "Survivors/Logic/SurvivorsGame.h"
#include "Survivors/Logic/SurvivorsGameConstants.h"
#include "Survivors/Logic/SurvivorsGemComponent.h"
#include "Survivors/Logic/Weapons/SurvivorsWeaponComponent.h"

void USurvivorsGarlicWeapon::OnLevelChanged(FWeaponLevel NewLevel)
{
	CacheParams();
}

void USurvivorsGarlicWeapon::CacheParams()
{
	const int32 Lv = FMath::Clamp(Level.Value, 1, SurvivorsGameConstants::MaxWeaponLevel);
	const int32 Idx = Lv - 1;

	if (WeaponType == EWeaponType::SoulEater)
	{
		const FGarlicParams& P = SurvivorsGameConstants::SoulEaterTable[Idx];
		CachedDamage      = P.Damage.Value;
		CachedHitInterval = P.HitInterval;
		CachedAreaRadius  = P.AreaRadius.Value;
	}
	else
	{
		// Garlic（デフォルト）
		const FGarlicParams& P = SurvivorsGameConstants::GarlicTable[Idx];
		CachedDamage      = P.Damage.Value;
		CachedHitInterval = P.HitInterval;
		CachedAreaRadius  = P.AreaRadius.Value;
	}
}

void USurvivorsGarlicWeapon::Tick(float Dt)
{
	if (!Game) return;

	// パッシブ効果を適用（エリア半径 × AreaMult）
	const FPassiveEffects& PE = GetPassiveEffects();
	const float EffectiveRadius = CachedAreaRadius * PE.AreaMult;
	const float EffectiveDamage = CachedDamage     * PE.DamageMult;

	for (int32 i = Game->Enemies.Num() - 1; i >= 0; --i)
	{
		FEnemyState& E = Game->Enemies[i];
		const float Dist = FVector2D::Distance(Game->PlayerPos, E.Pos);
		if (Dist <= EffectiveRadius + E.CollisionRadius)
		{
			// スロット別ヒット時刻でクールダウン管理
			const float LastHit = E.WeaponLastHitTime[SlotIdx].Seconds;
			if (Game->ElapsedTime - LastHit >= CachedHitInterval)
			{
				E.HP -= EffectiveDamage;
				E.WeaponLastHitTime[SlotIdx] = FSurvivorsElapsedTime(Game->ElapsedTime);

				// ノックバック
				if (Game->EnemyTypeTable.IsValidIndex(E.TypeId))
				{
					const float Resistance = Game->EnemyTypeTable[E.TypeId].KnockbackResistance;
					if (Resistance < 1.f)
					{
						const FVector2D Dir = (E.Pos - Game->PlayerPos).GetSafeNormal();
						E.Pos += Dir * SurvivorsGameConstants::GarlicKnockbackStrength * (1.f - Resistance);
					}
				}

				if (E.HP <= 0.f)
				{
					Game->GemComponent->DropGem(E.TypeId, E.Pos);
					Game->Enemies.RemoveAt(i);
					Game->LastReward += Game->KillReward;
				}
			}
		}
	}
}
