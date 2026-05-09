#include "Survivors/Logic/SurvivorsCollisionComponent.h"

#include "Survivors/Logic/SurvivorsGame.h"
#include "Survivors/View/WallActor.h"
#include "Kismet/GameplayStatics.h"

USurvivorsCollisionComponent::USurvivorsCollisionComponent()
{
	PrimaryComponentTick.bCanEverTick = false;
}

void USurvivorsCollisionComponent::Initialize(ASurvivorsGame* InGame)
{
	Game = InGame;
}

void USurvivorsCollisionComponent::CollectWallActors()
{
	if (!Game) return;

	Game->WallActors.Empty();
	TArray<AActor*> Found;
	UGameplayStatics::GetAllActorsOfClass(Game->GetWorld(), AWallActor::StaticClass(), Found);
	for (AActor* A : Found)
	{
		if (AWallActor* W = Cast<AWallActor>(A))
		{
			Game->WallActors.Add(W);
		}
	}

	UE_LOG(LogTemp, Log, TEXT("[SurvivorsGame] WallActors found: %d"), Game->WallActors.Num());
}

void USurvivorsCollisionComponent::ResolveWallCollisions()
{
	if (!Game) return;

	for (const TObjectPtr<AWallActor>& Wall : Game->WallActors)
	{
		if (!Wall) continue;
		const FBox2D Box = Wall->GetSimBounds(Game->SimToUE);

		const FVector2D Closest(
			FMath::Clamp(Game->PlayerPos.X, Box.Min.X, Box.Max.X),
			FMath::Clamp(Game->PlayerPos.Y, Box.Min.Y, Box.Max.Y));
		const FVector2D Delta = Game->PlayerPos - Closest;
		const float DistSq = Delta.SizeSquared();

		if (DistSq < Game->PlayerRadius * Game->PlayerRadius && DistSq > KINDA_SMALL_NUMBER)
		{
			const float Dist = FMath::Sqrt(DistSq);
			const FVector2D N = Delta / Dist;
			Game->PlayerPos = Closest + N * Game->PlayerRadius;
			const float VdotN = FVector2D::DotProduct(Game->PlayerVel, N);
			if (VdotN < 0.f) Game->PlayerVel -= N * VdotN;
		}
		else if (DistSq <= KINDA_SMALL_NUMBER)
		{
			const float px1 = Game->PlayerPos.X - Box.Min.X;
			const float px2 = Box.Max.X - Game->PlayerPos.X;
			const float py1 = Game->PlayerPos.Y - Box.Min.Y;
			const float py2 = Box.Max.Y - Game->PlayerPos.Y;
			const float m = FMath::Min(FMath::Min(px1, px2), FMath::Min(py1, py2));
			if      (m == px1) { Game->PlayerPos.X = Box.Min.X - Game->PlayerRadius; Game->PlayerVel.X = FMath::Min(Game->PlayerVel.X, 0.f); }
			else if (m == px2) { Game->PlayerPos.X = Box.Max.X + Game->PlayerRadius; Game->PlayerVel.X = FMath::Max(Game->PlayerVel.X, 0.f); }
			else if (m == py1) { Game->PlayerPos.Y = Box.Min.Y - Game->PlayerRadius; Game->PlayerVel.Y = FMath::Min(Game->PlayerVel.Y, 0.f); }
			else               { Game->PlayerPos.Y = Box.Max.Y + Game->PlayerRadius; Game->PlayerVel.Y = FMath::Max(Game->PlayerVel.Y, 0.f); }
		}
	}
}

float USurvivorsCollisionComponent::CastRayToObstacles(FVector2D Origin, FVector2D Dir) const
{
	if (!Game) return 0.f;

	float tMin = TNumericLimits<float>::Max();
	if (Dir.X >  1e-6f) tMin = FMath::Min(tMin, ( Game->FieldHalfSize - Origin.X) / Dir.X);
	if (Dir.X < -1e-6f) tMin = FMath::Min(tMin, (-Game->FieldHalfSize - Origin.X) / Dir.X);
	if (Dir.Y >  1e-6f) tMin = FMath::Min(tMin, ( Game->FieldHalfSize - Origin.Y) / Dir.Y);
	if (Dir.Y < -1e-6f) tMin = FMath::Min(tMin, (-Game->FieldHalfSize - Origin.Y) / Dir.Y);

	for (const TObjectPtr<AWallActor>& Wall : Game->WallActors)
	{
		if (!Wall) continue;
		const FBox2D Box = Wall->GetSimBounds(Game->SimToUE);

		float tNear = 0.f;
		float tFar = TNumericLimits<float>::Max();
		if (FMath::Abs(Dir.X) > 1e-6f)
		{
			float t1 = (Box.Min.X - Origin.X) / Dir.X;
			float t2 = (Box.Max.X - Origin.X) / Dir.X;
			if (t1 > t2) Swap(t1, t2);
			tNear = FMath::Max(tNear, t1);
			tFar = FMath::Min(tFar, t2);
		}
		else if (Origin.X < Box.Min.X || Origin.X > Box.Max.X) continue;

		if (FMath::Abs(Dir.Y) > 1e-6f)
		{
			float t1 = (Box.Min.Y - Origin.Y) / Dir.Y;
			float t2 = (Box.Max.Y - Origin.Y) / Dir.Y;
			if (t1 > t2) Swap(t1, t2);
			tNear = FMath::Max(tNear, t1);
			tFar = FMath::Min(tFar, t2);
		}
		else if (Origin.Y < Box.Min.Y || Origin.Y > Box.Max.Y) continue;

		if (tNear < tFar && tNear > 0.f)
		{
			tMin = FMath::Min(tMin, tNear);
		}
	}

	return tMin < TNumericLimits<float>::Max() ? tMin : 0.f;
}
