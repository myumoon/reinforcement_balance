#include "Survivors/Logic/SurvivorsGemComponent.h"

#include "Survivors/Logic/SurvivorsGame.h"
#include "Survivors/Logic/SurvivorsPlayerComponent.h"

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

	const EGemType Type = (TypeId >= 0 && TypeId < UE_ARRAY_COUNT(SurvivorsGameConstants::GemDropTable))
		? SurvivorsGameConstants::GemDropTable[TypeId]
		: EGemType::Blue;
	Game->Gems.Add({ Pos, Type });
}

void USurvivorsGemComponent::CheckCollections()
{
	if (!Game || !Game->PlayerComponent) return;

	const float RadSq = Game->GemPickupRadius * Game->GemPickupRadius;
	for (int32 i = Game->Gems.Num() - 1; i >= 0; --i)
	{
		if (FVector2D::DistSquared(Game->PlayerPos, Game->Gems[i].Pos) <= RadSq)
		{
			const float XPGain = SurvivorsGameConstants::GemXPValues[static_cast<uint8>(Game->Gems[i].Type)];
			Game->PlayerComponent->ProcessXPGain(XPGain);
			Game->LastReward += Game->ItemReward;
			Game->Gems.RemoveAt(i);
		}
	}
}
