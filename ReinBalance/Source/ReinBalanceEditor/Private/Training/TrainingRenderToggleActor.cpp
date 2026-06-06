#include "Training/TrainingRenderToggleActor.h"
#include "Engine/GameViewportClient.h"
#include "GameFramework/PlayerController.h"

ATrainingRenderToggleActor::ATrainingRenderToggleActor()
{
	PrimaryActorTick.bCanEverTick = true;
	ToggleKey = EKeys::F10;
}

void ATrainingRenderToggleActor::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	UWorld* World = GetWorld();
	if (!World) return;

	APlayerController* PC = World->GetFirstPlayerController();

	const bool bCtrlDown  = PC && (PC->IsInputKeyDown(EKeys::LeftControl) || PC->IsInputKeyDown(EKeys::RightControl));
	const bool bShiftDown = PC && (PC->IsInputKeyDown(EKeys::LeftShift)   || PC->IsInputKeyDown(EKeys::RightShift));
	const bool bKeyDown   = PC && PC->IsInputKeyDown(ToggleKey);
	// Ctrl または Shift が押されている場合はデバッグオーバーレイトグル（Ctrl+F10）に委譲
	const bool bTriggered = bKeyDown && !bCtrlDown && !bShiftDown;

	if (bTriggered && !bKeyWasDown)
	{
		ToggleRendering();
	}
	bKeyWasDown = bTriggered;
}

void ATrainingRenderToggleActor::ToggleRendering()
{
	bRenderEnabled = !bRenderEnabled;

	if (GEngine && GEngine->GameViewport)
	{
		GEngine->GameViewport->bDisableWorldRendering = !bRenderEnabled;
	}

	const FString Message = bRenderEnabled
		? TEXT("[Training] Render ON  (F10 to toggle)")
		: TEXT("[Training] Render OFF (F10 to toggle)");
	const FColor  Color   = bRenderEnabled ? FColor::Green : FColor::Yellow;
	GEngine->AddOnScreenDebugMessage(-1, 3.f, Color, Message);

	UE_LOG(LogTemp, Log, TEXT("ATrainingRenderToggleActor: World rendering %s"),
		bRenderEnabled ? TEXT("enabled") : TEXT("disabled"));
}
