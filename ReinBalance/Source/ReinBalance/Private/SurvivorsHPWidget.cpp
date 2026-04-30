#include "SurvivorsHPWidget.h"
#include "Blueprint/WidgetTree.h"
#include "Components/ProgressBar.h"
#include "Components/CanvasPanel.h"
#include "Components/CanvasPanelSlot.h"

void USurvivorsHPWidget::NativeConstruct()
{
	Super::NativeConstruct();

	if (!WidgetTree || WidgetTree->RootWidget) return;

	UCanvasPanel* Root = WidgetTree->ConstructWidget<UCanvasPanel>(
		UCanvasPanel::StaticClass(), TEXT("Root"));
	WidgetTree->RootWidget = Root;

	HPBar = WidgetTree->ConstructWidget<UProgressBar>(
		UProgressBar::StaticClass(), TEXT("HPBar"));

	if (UCanvasPanelSlot* BarSlot = Root->AddChildToCanvas(HPBar))
	{
		FAnchors FullFill;
		FullFill.Minimum = FVector2D(0.f, 0.f);
		FullFill.Maximum = FVector2D(1.f, 1.f);
		BarSlot->SetAnchors(FullFill);
		BarSlot->SetOffsets(FMargin(0.f));
	}

	HPBar->SetPercent(1.f);
	HPBar->SetFillColorAndOpacity(FLinearColor(0.f, 0.8f, 0.f, 1.f));
}

void USurvivorsHPWidget::UpdateDisplay(float HP, float MaxHP)
{
	if (!HPBar) return;

	const float Ratio = FMath::Clamp(MaxHP > 0.f ? HP / MaxHP : 0.f, 0.f, 1.f);
	HPBar->SetPercent(Ratio);

	FLinearColor Color;
	if (Ratio > 0.5f)
	{
		// 黄 → 緑: R が 1→0, G は 0.8 固定
		const float T = (Ratio - 0.5f) * 2.f;
		Color = FLinearColor(1.f - T, 0.8f, 0.f, 1.f);
	}
	else
	{
		// 赤 → 黄: G が 0→0.8
		const float T = Ratio * 2.f;
		Color = FLinearColor(0.8f, T * 0.8f, 0.f, 1.f);
	}
	HPBar->SetFillColorAndOpacity(Color);
}
