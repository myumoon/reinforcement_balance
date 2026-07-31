/**
 * legacy ObservationComponentをcanonical Logicへの薄い委譲として実装する。
 * 初心者向け: 観測layoutや数式をComponentへ複製せず、Game facadeから同じ値を受け取る。
 */
#include "Survivors/Game/SurvivorsObservationComponent.h"

#include "Survivors/Game/SurvivorsGame.h"

USurvivorsObservationComponent::USurvivorsObservationComponent()
{
	PrimaryComponentTick.bCanEverTick = false;
}

void USurvivorsObservationComponent::Initialize(ASurvivorsGame* InGame)
{
	Game = InGame;
}

TArray<FSurvivorsObsSegment> USurvivorsObservationComponent::GetObsSchema() const
{
	return Game ? Game->GetObsSchema() : TArray<FSurvivorsObsSegment>();
}

FString USurvivorsObservationComponent::GetObsSchemaHash() const
{
	return Game ? Game->GetObsSchemaHash() : FString();
}

TArray<float> USurvivorsObservationComponent::GetObservation() const
{
	return Game ? Game->GetObservation() : TArray<float>();
}
