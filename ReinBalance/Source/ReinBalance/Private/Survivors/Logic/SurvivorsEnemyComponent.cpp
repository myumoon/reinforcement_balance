#include "Survivors/Logic/SurvivorsEnemyComponent.h"

#include "Survivors/Logic/SurvivorsGame.h"
#include "Survivors/Logic/SurvivorsGemComponent.h"

USurvivorsEnemyComponent::USurvivorsEnemyComponent()
{
	PrimaryComponentTick.bCanEverTick = false;
}

void USurvivorsEnemyComponent::Initialize(ASurvivorsGame* InGame)
{
	Game = InGame;
}

void USurvivorsEnemyComponent::Reset()
{
	if (!Game) return;
	Game->Enemies.Empty();
}

void USurvivorsEnemyComponent::UpdateEnemies()
{
	if (!Game) return;

	for (FEnemyState& E : Game->Enemies)
	{
		E.Vel = (Game->PlayerPos - E.Pos).GetSafeNormal() * GetEnemySpeed(E.TypeId);
		E.Pos += E.Vel * SurvivorsGameConstants::PhysicsDt;
	}
}

void USurvivorsEnemyComponent::ApplyAuraDamage()
{
	if (!Game || !Game->GemComponent) return;

	const int32 GarlicLv = FMath::Clamp(Game->WeaponSlots[0].Level, 1, SurvivorsGameConstants::MaxWeaponLevel);
	const FGarlicParams& GP = SurvivorsGameConstants::GarlicTable[GarlicLv - 1];

	for (int32 i = Game->Enemies.Num() - 1; i >= 0; --i)
	{
		FEnemyState& E = Game->Enemies[i];
		const float Dist = FVector2D::Distance(Game->PlayerPos, E.Pos);
		if (Dist <= GP.AreaRadius + E.CollisionRadius)
		{
			if (Game->ElapsedTime - E.GarlicLastHitTime >= GP.HitInterval)
			{
				E.HP -= GP.Damage;
				E.GarlicLastHitTime = Game->ElapsedTime;

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

void USurvivorsEnemyComponent::ApplyContactDamage()
{
	if (!Game) return;

	for (FEnemyState& E : Game->Enemies)
	{
		const float HitR = Game->PlayerRadius + E.CollisionRadius;
		if (FVector2D::DistSquared(Game->PlayerPos, E.Pos) < HitR * HitR)
		{
			if (Game->ElapsedTime - E.PlayerLastHitTime >= SurvivorsGameConstants::ContactHitInterval)
			{
				Game->PlayerHP -= E.ContactDamage;
				E.PlayerLastHitTime = Game->ElapsedTime;
			}
		}
	}
	Game->PlayerHP = FMath::Max(Game->PlayerHP, 0.f);
}

float USurvivorsEnemyComponent::GetEnemySpeed(int32 TypeId) const
{
	if (!Game || !Game->EnemyTypeTable.IsValidIndex(TypeId)) return 50.f * (Game ? Game->EnemySpeedMult : 1.f);
	return Game->EnemyTypeTable[TypeId].Speed * Game->EnemySpeedMult;
}

float USurvivorsEnemyComponent::GetEnemyTypeMaxHP(int32 TypeId) const
{
	if (!Game || !Game->EnemyTypeTable.IsValidIndex(TypeId)) return 1.f;
	return Game->EnemyTypeTable[TypeId].BaseHP;
}
