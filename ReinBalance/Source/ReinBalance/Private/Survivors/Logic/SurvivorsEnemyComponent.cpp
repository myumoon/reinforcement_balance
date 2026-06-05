#include "Survivors/Logic/SurvivorsEnemyComponent.h"

#include "Survivors/Logic/SurvivorsCollisionComponent.h"
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

	const bool bGlobalFreeze = (Game->ElapsedTime < Game->GlobalFreezeUntilTime);
	for (FEnemyState& E : Game->Enemies)
	{
		// 個別フリーズまたはグローバルフリーズ（Orologion）中は移動スキップ
		if (E.bFrozen || bGlobalFreeze) continue;

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
	ensureMsgf(false, TEXT("ApplyContactDamage() は非推奨。PhysicsStep 内の ComputeContactHits/ApplyContactHits を使うこと。"));
}

void USurvivorsEnemyComponent::ComputeContactHits(USurvivorsCollisionComponent* CollComp, FSurvivorsHitFrame& HitFrame)
{
	if (!Game || !CollComp) return;

	TArray<const FSurvivorsTargetProxy*> Contacts;
	CollComp->QueryEnemyContacts(Game->PlayerPos, Game->PlayerRadius, Contacts);

	for (const FSurvivorsTargetProxy* Proxy : Contacts)
	{
		// narrowphase
		if ((Game->PlayerPos - Proxy->Pos).SizeSquared() > FMath::Square(Game->PlayerRadius + Proxy->Radius)) continue;

		const int32 EIdx = Proxy->Ref.IndexAtBuildTime;
		if (!Game->Enemies.IsValidIndex(EIdx) || Game->Enemies[EIdx].UniqueId != Proxy->Ref.UniqueId) continue;
		const FEnemyState& E = Game->Enemies[EIdx];
		if (E.bPendingRemove) continue;

		if (Game->ElapsedTime - E.PlayerLastHitTime >= SurvivorsGameConstants::ContactHitInterval)
		{
			FSurvivorsHitEvent Ev;
			Ev.Type = ESurvivorsHitType::ContactDamage;
			Ev.Target = Proxy->Ref;
			Ev.Damage = E.ContactDamage;
			HitFrame.Events.Add(Ev);
		}
	}
}

void USurvivorsEnemyComponent::ApplyContactHits(FSurvivorsHitFrame& HitFrame)
{
	if (!Game) return;

	for (const FSurvivorsHitEvent& Ev : HitFrame.Events)
	{
		if (Ev.Type != ESurvivorsHitType::ContactDamage) continue;

		// Laurel シールドが有効な場合は接触ダメージを受けない
		if (Game->bShieldActive) continue;

		const int32 EIdx = Ev.Target.IndexAtBuildTime;
		if (!Game->Enemies.IsValidIndex(EIdx)) continue;
		FEnemyState& E = Game->Enemies[EIdx];
		if (E.UniqueId != Ev.Target.UniqueId) continue;
		if (E.bPendingRemove) continue;

		Game->PlayerHP -= Ev.Damage;  // ContactDamage はスポーン時に EnemyDamageScale 適用済み
		E.PlayerLastHitTime = Game->ElapsedTime;
	}

	Game->PlayerHP = FMath::Max(0.f, Game->PlayerHP);
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
