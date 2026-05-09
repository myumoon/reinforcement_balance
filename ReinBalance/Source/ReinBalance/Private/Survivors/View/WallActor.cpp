#include "Survivors/View/WallActor.h"
#include "Engine/StaticMesh.h"
#include "UObject/ConstructorHelpers.h"

AWallActor::AWallActor()
{
	PrimaryActorTick.bCanEverTick = false;

	WallMesh = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("WallMesh"));
	SetRootComponent(WallMesh);

	static ConstructorHelpers::FObjectFinder<UStaticMesh> CubeAsset(
		TEXT("/Engine/BasicShapes/Cube.Cube"));
	if (CubeAsset.Succeeded())
		WallMesh->SetStaticMesh(CubeAsset.Object);

	WallMesh->SetCollisionProfileName(TEXT("BlockAll"));
	WallMesh->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
}

FBox2D AWallActor::GetSimBounds(float InSimToUE) const
{
	const FBox B = GetComponentsBoundingBox(true);
	const float Inv = 1.f / InSimToUE;
	return FBox2D(
		FVector2D(B.Min.X * Inv, B.Min.Y * Inv),
		FVector2D(B.Max.X * Inv, B.Max.Y * Inv));
}
