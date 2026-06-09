// Fill out your copyright notice in the Description page of Project Settings.

#include "Survivors/Logic/SurvivorsDebugRegistry.h"

#if WITH_EDITOR

ISurvivorsDebugSlot* FSurvivorsDebugRegistry::ActiveSlotComponent = nullptr;

void FSurvivorsDebugRegistry::RegisterSlotComponent(ISurvivorsDebugSlot* Component)
{
	ActiveSlotComponent = Component;
}

void FSurvivorsDebugRegistry::UnregisterSlotComponent(ISurvivorsDebugSlot* Component)
{
	if (ActiveSlotComponent == Component)
	{
		ActiveSlotComponent = nullptr;
	}
}

ISurvivorsDebugSlot* FSurvivorsDebugRegistry::GetSlotComponent()
{
	return ActiveSlotComponent;
}

#endif // WITH_EDITOR
