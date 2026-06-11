#pragma once

#include "CoreMinimal.h"

namespace SurvivorsWikiSpec
{
	enum class EGemColor : uint8
	{
		Blue,
		Green,
		Red,
	};

	static constexpr float StandardMaxPlayerHP = 100.f;
	static constexpr float StandardMoveSpeed = 80.f;
	static constexpr float BaseGemPickupRadius = 30.f;

	static constexpr float PummarolaRecoveryPerLevel = 0.2f;
	static constexpr float SpellbinderDurationPerLevel = 0.10f;
	static constexpr float WingsMoveSpeedPerLevel = 0.10f;
	static constexpr float CrownGrowthPerLevel = 0.08f;
	static constexpr float SkullCursePerLevel = 0.10f;
	static constexpr float TorronasLevel1Omni = 0.04f;
	static constexpr float TorronasLevel2To8OmniPerLevel = 0.03f;
	static constexpr float TorronasLevel9CurseBonus = 1.0f;

	static constexpr int32 PassiveTypeCount = 18;
	inline constexpr int32 PassiveMaxLevel[PassiveTypeCount] = {
		0, 5, 5, 5, 5, 5, 5, 5, 5, 2, 5, 5, 5, 5, 0, 5, 2, 9
	};

	inline constexpr float AttractorbPickupRadiusMult[5] = {
		1.5f,
		1.995f,
		2.49375f,
		2.9925f,
		3.980025f,
	};

	inline constexpr float GemXPValues[3] = { 2.f, 9.f, 10.f };
	static constexpr float BlueGemMaxXP = 2.f;
	static constexpr float GreenGemMaxXP = 9.f;
	static constexpr float RedGemMinXP = 10.f;
	static constexpr int32 RedGemMinMultiplier = 10;
	static constexpr int32 RedGemMaxMultiplier = 50;

	static constexpr int32 BaseWeaponMaxLevel = 8;
	static constexpr int32 LaurelMaxLevel = 7;
	static constexpr int32 EvolvedWeaponMaxLevel = 1;

	inline constexpr float XPRequiredForLevel(int32 Level)
	{
		if (Level <= 1) return 0.f;
		if (Level == 2) return 5.f;
		if (Level <= 20)
		{
			return 5.f + 10.f * static_cast<float>(Level - 2);
		}
		if (Level <= 40)
		{
			const float Base = 195.f + 13.f * static_cast<float>(Level - 21);
			return Level == 21 ? Base + 600.f : Base;
		}

		const float Base = 455.f + 16.f * static_cast<float>(Level - 41);
		return Level == 41 ? Base + 2400.f : Base;
	}

	inline float CumulativeXPForLevel(int32 Level)
	{
		float Total = 0.f;
		for (int32 Lv = 2; Lv <= Level; ++Lv)
		{
			Total += XPRequiredForLevel(Lv);
		}
		return Total;
	}

	inline constexpr float TemporaryGrowthBonusForPlayerLevel(int32 PlayerLevel)
	{
		return (PlayerLevel == 20 || PlayerLevel == 40) ? 1.f : 0.f;
	}

	inline constexpr float EffectiveGrowthMultiplier(float GrowthMult, int32 PlayerLevel)
	{
		const float WithTemporaryBonus = GrowthMult + TemporaryGrowthBonusForPlayerLevel(PlayerLevel);
		return WithTemporaryBonus > 0.f ? WithTemporaryBonus : 0.f;
	}

	inline constexpr float EffectiveXPGain(float BaseXP, float GrowthMult, int32 PlayerLevel)
	{
		return BaseXP * EffectiveGrowthMultiplier(GrowthMult, PlayerLevel);
	}

	inline constexpr float CrownGrowthMultiplierForLevel(int32 Level)
	{
		return 1.f + CrownGrowthPerLevel * static_cast<float>(Level);
	}

	inline constexpr float AttractorbPickupRadiusMultiplierForLevel(int32 Level)
	{
		if (Level <= 0) return 1.f;
		const int32 ClampedLevel = Level < 5 ? Level : 5;
		return AttractorbPickupRadiusMult[ClampedLevel - 1];
	}

	inline constexpr float PummarolaRecoveryForLevel(int32 Level)
	{
		return PummarolaRecoveryPerLevel * static_cast<float>(Level);
	}

	inline constexpr float SpellbinderDurationBonusForLevel(int32 Level)
	{
		return SpellbinderDurationPerLevel * static_cast<float>(Level);
	}

	inline constexpr float WingsMoveSpeedBonusForLevel(int32 Level)
	{
		return WingsMoveSpeedPerLevel * static_cast<float>(Level);
	}

	inline constexpr float SkullCurseBonusForLevel(int32 Level)
	{
		return SkullCursePerLevel * static_cast<float>(Level);
	}

	inline constexpr float TorronasOmniBonusForLevel(int32 Level)
	{
		if (Level <= 0) return 0.f;
		const int32 ExtraLevels = Level - 1 < 7 ? Level - 1 : 7;
		return TorronasLevel1Omni + TorronasLevel2To8OmniPerLevel * static_cast<float>(ExtraLevels);
	}

	inline constexpr float TorronasCurseBonusForLevel(int32 Level)
	{
		return Level >= 9 ? TorronasLevel9CurseBonus : 0.f;
	}

	inline constexpr EGemColor GemColorForExperience(float XP)
	{
		return XP <= BlueGemMaxXP ? EGemColor::Blue
			: XP <= GreenGemMaxXP ? EGemColor::Green
			: EGemColor::Red;
	}

	inline constexpr float RedGemExperienceForMultiplier(float BaseXP, int32 Multiplier)
	{
		return BaseXP * static_cast<float>(Multiplier);
	}
}
