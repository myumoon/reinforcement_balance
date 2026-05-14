#include "Survivors/Logic/SurvivorsPlayerComponent.h"

#include "Survivors/Logic/SurvivorsGame.h"

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

	Game->PlayerPos = FVector2D::ZeroVector;
	Game->PlayerVel = FVector2D::ZeroVector;
	Game->PlayerHP = Game->MaxPlayerHP;
	Game->PlayerXP = 0.f;
	Game->PlayerLevel = 1;

	Game->WeaponSlots[0].Type = EWeaponType::Aura;
	Game->WeaponSlots[0].Level = 1;
	Game->AuraRadius = SurvivorsGameConstants::GarlicTable[0].AreaRadius;
}

void USurvivorsPlayerComponent::ApplyAction(int32 ActionIdx)
{
	if (!Game) return;

	// アクション番号定義（対称性拡張・Python/EUREKA prompt と共通仕様）
	// 0=北(+Y), 1=北東, 2=東(+X), 3=南東, 4=南(-Y), 5=南西, 6=西(-X), 7=北西, 8=静止
	FVector2D MoveDir = FVector2D::ZeroVector;
	switch (ActionIdx)
	{
		case 0: MoveDir = FVector2D( 0.f,  1.f);                        break; // 北
		case 1: MoveDir = FVector2D( 1.f,  1.f).GetSafeNormal(); break; // 北東
		case 2: MoveDir = FVector2D( 1.f,  0.f);                        break; // 東
		case 3: MoveDir = FVector2D( 1.f, -1.f).GetSafeNormal(); break; // 南東
		case 4: MoveDir = FVector2D( 0.f, -1.f);                        break; // 南
		case 5: MoveDir = FVector2D(-1.f, -1.f).GetSafeNormal(); break; // 南西
		case 6: MoveDir = FVector2D(-1.f,  0.f);                        break; // 西
		case 7: MoveDir = FVector2D(-1.f,  1.f).GetSafeNormal(); break; // 北西
		default: break; // 8=静止
	}
	Game->PlayerVel = MoveDir * Game->MoveSpeed;
	Game->PlayerPos += Game->PlayerVel * SurvivorsGameConstants::PhysicsDt;
}

float USurvivorsPlayerComponent::XPRequiredForLevel(int32 Level) const
{
	if (Level <= 1) return 0.f;
	if (Level == 2) return 5.f;
	float Value = 5.f;
	for (int32 Lv = 3; Lv <= Level; ++Lv)
	{
		if      (Lv <= 20) Value += 10.f;
		else if (Lv <= 40) Value += 13.f;
		else if (Lv <= 60) Value += 16.f;
		else if (Lv <= 80) Value += 19.f;
		else               Value += 22.f;
	}
	return Value;
}

float USurvivorsPlayerComponent::CumulativeXPForLevel(int32 Level) const
{
	float Total = 0.f;
	for (int32 Lv = 2; Lv <= Level; ++Lv)
	{
		Total += XPRequiredForLevel(Lv);
	}
	return Total;
}

void USurvivorsPlayerComponent::ProcessXPGain(float Amount)
{
	if (!Game) return;

	Game->PlayerXP += Amount;
	while (true)
	{
		const float NextThreshold = CumulativeXPForLevel(Game->PlayerLevel + 1);
		if (Game->PlayerXP < NextThreshold) break;
		Game->PlayerLevel++;
		OnLevelUp(Game->PlayerLevel);
	}
}

void USurvivorsPlayerComponent::OnLevelUp(int32 NextLevel)
{
	if (!Game) return;

	const int32 NewGarlicLv = FMath::Min(Game->WeaponSlots[0].Level + 1, SurvivorsGameConstants::MaxWeaponLevel);
	Game->WeaponSlots[0].Level = NewGarlicLv;
	Game->AuraRadius = SurvivorsGameConstants::GarlicTable[NewGarlicLv - 1].AreaRadius;
}
