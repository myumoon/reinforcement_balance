#include "Survivors/View/SurvivorsGameView.h"

#include "Survivors/View/SurvivorsEnemyViewComponent.h"
#include "Survivors/View/SurvivorsGUIViewComponent.h"
#include "Survivors/View/SurvivorsGemViewComponent.h"
#include "Survivors/View/SurvivorsPlayerViewComponent.h"
#include "Components/SceneComponent.h"

ASurvivorsGameView::ASurvivorsGameView()
{
	PrimaryActorTick.bCanEverTick = true;

	USceneComponent* SceneRoot = CreateDefaultSubobject<USceneComponent>(TEXT("SceneRoot"));
	SetRootComponent(SceneRoot);

	PlayerView = CreateDefaultSubobject<USurvivorsPlayerViewComponent>(TEXT("PlayerView"));
	GemView = CreateDefaultSubobject<USurvivorsGemViewComponent>(TEXT("GemView"));
	EnemyView = CreateDefaultSubobject<USurvivorsEnemyViewComponent>(TEXT("EnemyView"));
	GUIView = CreateDefaultSubobject<USurvivorsGUIViewComponent>(TEXT("GUIView"));
}

void ASurvivorsGameView::BeginPlay()
{
	Super::BeginPlay();

	if (!Game)
	{
		UE_LOG(LogTemp, Warning,
			TEXT("ASurvivorsGameView: Game is not assigned. Set an ASurvivorsGame reference in Details."));
		SetActorTickEnabled(false);
		return;
	}

	USceneComponent* AttachParent = RootComponent;
	PlayerView->Initialize(Game, AttachParent);
	GemView->Initialize(Game, AttachParent);
	EnemyView->Initialize(Game, AttachParent);
	GUIView->Initialize(Game, AttachParent);
}

void ASurvivorsGameView::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	if (!Game) return;

	PlayerView->UpdateView();
	GemView->UpdateView();
	EnemyView->UpdateView();
	GUIView->UpdateView();
}
