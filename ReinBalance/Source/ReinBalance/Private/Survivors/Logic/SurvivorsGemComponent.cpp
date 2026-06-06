#include "Survivors/Logic/SurvivorsGemComponent.h"

#include "Survivors/Logic/SurvivorsCollisionComponent.h"
#include "Survivors/Logic/SurvivorsGame.h"
#include "Survivors/Logic/SurvivorsGameConstants.h"
#include "Survivors/Logic/SurvivorsPlayerComponent.h"
#include "Survivors/Logic/SurvivorsWikiSpec.h"

USurvivorsGemComponent::USurvivorsGemComponent()
{
	PrimaryComponentTick.bCanEverTick = false;
}

void USurvivorsGemComponent::Initialize(ASurvivorsGame* InGame)
{
	Game = InGame;
}

void USurvivorsGemComponent::Reset()
{
	if (!Game) return;
	Game->Gems.Empty();
}

void USurvivorsGemComponent::DropGem(int32 TypeId, FVector2D Pos)
{
	if (!Game) return;

	const EGemType ConfiguredType = (TypeId >= 0 && TypeId < UE_ARRAY_COUNT(SurvivorsGameConstants::GemDropTable))
		? SurvivorsGameConstants::GemDropTable[TypeId]
		: EGemType::Blue;
	const float EnemyXPDrop = Game->EnemyTypeTable.IsValidIndex(TypeId)
		? Game->EnemyTypeTable[TypeId].XPDrop
		: SurvivorsGameConstants::GemXPValues[0];
	const EGemType Type = ConfiguredType == EGemType::Red
		? EGemType::Red
		: SurvivorsGameConstants::GemTypeForExperience(EnemyXPDrop);
	float BaseExperienceValue = SurvivorsGameConstants::GemXPValues[static_cast<uint8>(Type)];
	if (Game->EnemyTypeTable.IsValidIndex(TypeId))
	{
		if (Type == EGemType::Red)
		{
			BaseExperienceValue = FMath::Max(1.f, EnemyXPDrop);
		}
		else if (Type == EGemType::Blue)
		{
			BaseExperienceValue = FMath::Clamp(EnemyXPDrop, 1.f, SurvivorsGameConstants::BlueGemMaxXP);
		}
		else
		{
			BaseExperienceValue = FMath::Clamp(EnemyXPDrop, SurvivorsGameConstants::BlueGemMaxXP + 1.f, SurvivorsGameConstants::GreenGemMaxXP);
		}
	}
	FGemState G;
	G.Pos = Pos;
	G.Type = Type;
	G.BaseExperienceValue = BaseExperienceValue;
	G.UniqueId = ++Game->NextGemId;
	G.bPendingRemove = false;
	Game->Gems.Add(G);
}

void USurvivorsGemComponent::CheckCollections()
{
	// ComputePickupHits / ApplyPickupHits に移管済み。後方互換のため空実装を残す。
}

void USurvivorsGemComponent::ComputePickupHits(USurvivorsCollisionComponent* CollComp, FSurvivorsHitFrame& HitFrame)
{
	if (!Game || !CollComp) return;

	TArray<const FSurvivorsTargetProxy*> Contacts;
	CollComp->QueryPickupContacts(Game->PlayerPos, Game->GemPickupRadius, Contacts);

	for (const FSurvivorsTargetProxy* Proxy : Contacts)
	{
		// narrowphase
		if ((Game->PlayerPos - Proxy->Pos).SizeSquared() > FMath::Square(Game->GemPickupRadius)) continue;

		const int32 GIdx = Proxy->Ref.IndexAtBuildTime;
		if (!Game->Gems.IsValidIndex(GIdx) || Game->Gems[GIdx].UniqueId != Proxy->Ref.UniqueId) continue;
		if (Game->Gems[GIdx].bPendingRemove) continue;

		FSurvivorsHitEvent Ev;
		Ev.Type = ESurvivorsHitType::PickupCollect;
		Ev.Target = Proxy->Ref;
		HitFrame.Events.Add(Ev);
	}
}

void USurvivorsGemComponent::ApplyPickupHits(FSurvivorsHitFrame& HitFrame)
{
	if (!Game || !Game->PlayerComponent) return;

	for (const FSurvivorsHitEvent& Ev : HitFrame.Events)
	{
		if (Ev.Type != ESurvivorsHitType::PickupCollect) continue;

		const int32 GIdx = Ev.Target.IndexAtBuildTime;
		if (!Game->Gems.IsValidIndex(GIdx) || Game->Gems[GIdx].UniqueId != Ev.Target.UniqueId) continue;
		if (Game->Gems[GIdx].bPendingRemove) continue;

		Game->Gems[GIdx].bPendingRemove = true;
		float XPGain = Game->Gems[GIdx].BaseExperienceValue;
		if (Game->Gems[GIdx].Type == EGemType::Red)
		{
			const int32 Mult = Game->RandStream.RandRange(
				SurvivorsGameConstants::RedGemMinMultiplier,
				SurvivorsGameConstants::RedGemMaxMultiplier);
			XPGain = SurvivorsWikiSpec::RedGemExperienceForMultiplier(XPGain, Mult);
		}
		Game->PlayerComponent->ProcessXPGain(XPGain);
		Game->LastReward += Game->ItemReward;
	}
}
