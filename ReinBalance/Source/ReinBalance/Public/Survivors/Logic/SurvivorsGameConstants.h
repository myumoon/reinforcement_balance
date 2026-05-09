#pragma once

#include "CoreMinimal.h"
#include "Survivors/Logic/SurvivorsTypes.h"

struct FGarlicParams
{
	float Damage;
	float HitInterval;
	float AreaRadius;
};

namespace SurvivorsGameConstants
{
	static constexpr int32 NumGemObs = 20;
	static constexpr int32 MaxEnemyObs = 20;
	static constexpr int32 MaxWeaponSlots = 3;
	static constexpr int32 MaxWeaponTypeSlots = 8;
	static constexpr int32 MaxWeaponLevel = 8;
	static constexpr int32 MaxPlayerLevel = 100;
	static constexpr float PhysicsDt = 1.f / 60.f;
	static constexpr float MaxGameTime = 1800.f;
	static constexpr float ContactHitInterval = 0.5f;
	static constexpr float GarlicKnockbackStrength = 10.f;

	inline const FVector2D RayDirs[8] = {
		FVector2D( 1.f,      0.f     ),
		FVector2D( 0.7071f,  0.7071f ),
		FVector2D( 0.f,      1.f     ),
		FVector2D(-0.7071f,  0.7071f ),
		FVector2D(-1.f,      0.f     ),
		FVector2D(-0.7071f, -0.7071f ),
		FVector2D( 0.f,     -1.f     ),
		FVector2D( 0.7071f, -0.7071f ),
	};

	inline constexpr FGarlicParams GarlicTable[MaxWeaponLevel] = {
		{  5.f, 1.30f, 25.f },
		{  5.f, 1.25f, 30.f },
		{ 10.f, 1.20f, 35.f },
		{ 10.f, 1.15f, 40.f },
		{ 15.f, 1.10f, 45.f },
		{ 15.f, 1.05f, 50.f },
		{ 20.f, 1.00f, 55.f },
		{ 20.f, 0.95f, 60.f },
	};

	inline constexpr float GemXPValues[3] = { 1.f, 5.f, 10.f };

	inline constexpr EGemType GemDropTable[11] = {
		EGemType::Blue,
		EGemType::Blue,
		EGemType::Blue,
		EGemType::Blue,
		EGemType::Green,
		EGemType::Green,
		EGemType::Green,
		EGemType::Blue,
		EGemType::Green,
		EGemType::Green,
		EGemType::Red,
	};
}
