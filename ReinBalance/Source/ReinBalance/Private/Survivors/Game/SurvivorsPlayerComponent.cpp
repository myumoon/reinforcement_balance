/**
 * legacy PlayerComponent を canonical Logic への薄い委譲として実装する。
 * 初心者向け: ここでは XP 閾値や候補選択を計算せず、Game facade に処理を渡す。
 */
#include "Survivors/Game/SurvivorsPlayerComponent.h"

#include "Survivors/Game/SurvivorsGame.h"
#include "Survivors/SurvivorsWikiSpec.h"

USurvivorsPlayerComponent::USurvivorsPlayerComponent()
{
	PrimaryComponentTick.bCanEverTick = false;
}

void USurvivorsPlayerComponent::Initialize(ASurvivorsGame* InGame)
{
	Game = InGame;
}

void USurvivorsPlayerComponent::Reset()
{
	if (!Game) return;
	// legacy debug mirror だけを初期化し、production の level-up state には触れない。
	// 初心者向け: 実際の訓練状態は先に FSurvivorsGameLogic::Reset が初期化しています。
	Game->PlayerPos = FVector2D::ZeroVector;
	Game->PlayerVel = FVector2D::ZeroVector;
	Game->PlayerHP = Game->MaxPlayerHP;
	Game->PlayerXP = 0.f;
	Game->PlayerLevel = 1;
}

void USurvivorsPlayerComponent::ApplyAction(int32 ActionIdx)
{
	if (!Game) return;
	FVector2D Direction = FVector2D::ZeroVector;
	switch (ActionIdx)
	{
	case 0: Direction = FVector2D(0.f, 1.f); break;
	case 1: Direction = FVector2D(1.f, 1.f).GetSafeNormal(); break;
	case 2: Direction = FVector2D(1.f, 0.f); break;
	case 3: Direction = FVector2D(1.f, -1.f).GetSafeNormal(); break;
	case 4: Direction = FVector2D(0.f, -1.f); break;
	case 5: Direction = FVector2D(-1.f, -1.f).GetSafeNormal(); break;
	case 6: Direction = FVector2D(-1.f, 0.f); break;
	case 7: Direction = FVector2D(-1.f, 1.f).GetSafeNormal(); break;
	default: break;
	}
	Game->PlayerVel = Direction * Game->MoveSpeed;
	Game->PlayerPos += Game->PlayerVel * SurvivorsGameConstants::PhysicsDt;
}

float USurvivorsPlayerComponent::XPRequiredForLevel(int32 Level) const
{
	return SurvivorsWikiSpec::XPRequiredForLevel(Level);
}

float USurvivorsPlayerComponent::CumulativeXPForLevel(int32 Level) const
{
	return SurvivorsWikiSpec::CumulativeXPForLevel(Level);
}

void USurvivorsPlayerComponent::ProcessXPGain(float Amount)
{
	if (Game)
	{
		Game->AddExperience(Amount);
	}
}

TArray<int32> USurvivorsPlayerComponent::GetEvolvableWeapons() const
{
	return Game ? Game->GetEvolvableWeaponsForChest() : TArray<int32>();
}

void USurvivorsPlayerComponent::EvolveWeapon(int32 SlotIdx, EWeaponType EvolvedType)
{
	if (Game)
	{
		Game->EvolveWeaponFromChest(SlotIdx, EvolvedType);
	}
}
