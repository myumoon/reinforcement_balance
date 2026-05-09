#include "Survivors/View/SurvivorsGUIViewComponent.h"

#include "Survivors/Logic/SurvivorsGame.h"
#include "Survivors/View/SurvivorsHPWidget.h"
#include "Components/SceneComponent.h"
#include "Components/WidgetComponent.h"

USurvivorsGUIViewComponent::USurvivorsGUIViewComponent()
{
	PrimaryComponentTick.bCanEverTick = false;
}

void USurvivorsGUIViewComponent::Initialize(ASurvivorsGame* InGame, USceneComponent* AttachParent)
{
	Game = InGame;
	if (!AttachParent || HPWidgetComp) return;

	HPWidgetComp = NewObject<UWidgetComponent>(GetOwner(), TEXT("SurvivorsHPWidget"));
	HPWidgetComp->RegisterComponent();
	HPWidgetComp->AttachToComponent(AttachParent, FAttachmentTransformRules::KeepRelativeTransform);
	HPWidgetComp->SetWidgetClass(USurvivorsHPWidget::StaticClass());
	HPWidgetComp->SetWidgetSpace(EWidgetSpace::Screen);
	HPWidgetComp->SetDrawSize(FVector2D(120.f, 20.f));
}

void USurvivorsGUIViewComponent::UpdateView()
{
	if (!Game || !HPWidgetComp) return;

	const FVector2D PPos = Game->GetPlayerPos();
	HPWidgetComp->SetRelativeLocation(FVector(PPos.X * Game->SimToUE, PPos.Y * Game->SimToUE, 80.f));
	if (USurvivorsHPWidget* Widget = Cast<USurvivorsHPWidget>(HPWidgetComp->GetUserWidgetObject()))
	{
		Widget->UpdateDisplay(Game->GetPlayerHP(), Game->GetMaxPlayerHP());
	}
}
