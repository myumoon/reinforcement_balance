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

// ApplyAuraDamage() は USurvivorsWeaponComponent::TickAllWeapons() に移管済み。
// 後方互換のため宣言は残すが、内部実装は空にしてコンパイルのみ通す。
void USurvivorsEnemyComponent::ApplyAuraDamage()
{
	// WeaponComponent に移管済み。PhysicsStep から直接呼ばれないが
	// SurvivorsGame.cpp の互換シムから呼ばれる可能性があるため空実装を残す。
}

void USurvivorsEnemyComponent::ApplyContactDamage()
{
	if (!Game) return;

	for (FEnemyState& E : Game->Enemies)
	{
		// Laurel シールドが有効な場合は接触ダメージを受けない
		if (Game->bShieldActive) continue;

		const float HitR = Game->PlayerRadius + E.CollisionRadius;
		if (FVector2D::DistSquared(Game->PlayerPos, E.Pos) < HitR * HitR)
		{
			if (Game->ElapsedTime - E.PlayerLastHitTime >= SurvivorsGameConstants::ContactHitInterval)
			{
				Game->PlayerHP -= E.ContactDamage * Game->EnemyDamageScale;
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
