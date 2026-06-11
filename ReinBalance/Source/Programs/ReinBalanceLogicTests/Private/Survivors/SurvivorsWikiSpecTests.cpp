#include "TestHarness.h"

#include "Survivors/Logic/SurvivorsWikiSpec.h"

namespace
{
bool NearlyEqual(float Actual, float Expected, float Tolerance = 0.001f)
{
	return FMath::IsNearlyEqual(Actual, Expected, Tolerance);
}
}

TEST_CASE("Survivors wiki base player stats use Poe baseline values", "[unit][survivors][logic][wiki-spec]")
{
	CHECK(SurvivorsWikiSpec::StandardMaxPlayerHP == 70.f);
	CHECK(SurvivorsWikiSpec::StandardMoveSpeed == 80.f);
	CHECK(SurvivorsWikiSpec::BaseGemPickupRadius == 30.f);
}

TEST_CASE("Survivors wiki XP requirement starts at zero before level two", "[unit][survivors][logic][wiki-spec][level]")
{
	CHECK(SurvivorsWikiSpec::XPRequiredForLevel(0) == 0.f);
	CHECK(SurvivorsWikiSpec::XPRequiredForLevel(1) == 0.f);
	CHECK(SurvivorsWikiSpec::XPRequiredForLevel(2) == 5.f);
}

TEST_CASE("Survivors wiki XP requirement grows by ten through level twenty", "[unit][survivors][logic][wiki-spec][level]")
{
	CHECK(SurvivorsWikiSpec::XPRequiredForLevel(3) == 15.f);
	CHECK(SurvivorsWikiSpec::XPRequiredForLevel(10) == 85.f);
	CHECK(SurvivorsWikiSpec::XPRequiredForLevel(20) == 185.f);
}

TEST_CASE("Survivors wiki XP requirement has a one-level wall at level twenty one", "[unit][survivors][logic][wiki-spec][level]")
{
	CHECK(SurvivorsWikiSpec::XPRequiredForLevel(21) == 795.f);
	CHECK(SurvivorsWikiSpec::XPRequiredForLevel(22) == 208.f);
	CHECK(SurvivorsWikiSpec::XPRequiredForLevel(40) == 442.f);
}

TEST_CASE("Survivors wiki XP requirement has a one-level wall at level forty one", "[unit][survivors][logic][wiki-spec][level]")
{
	CHECK(SurvivorsWikiSpec::XPRequiredForLevel(41) == 2855.f);
	CHECK(SurvivorsWikiSpec::XPRequiredForLevel(42) == 471.f);
	CHECK(SurvivorsWikiSpec::XPRequiredForLevel(100) == 1399.f);
}

TEST_CASE("Survivors wiki cumulative XP is the sum of per-level requirements", "[unit][survivors][logic][wiki-spec][level]")
{
	CHECK(SurvivorsWikiSpec::CumulativeXPForLevel(1) == 0.f);
	CHECK(SurvivorsWikiSpec::CumulativeXPForLevel(2) == 5.f);
	CHECK(SurvivorsWikiSpec::CumulativeXPForLevel(3) == 20.f);
	CHECK(SurvivorsWikiSpec::CumulativeXPForLevel(20) == 1805.f);
	CHECK(SurvivorsWikiSpec::CumulativeXPForLevel(21) == 2600.f);
	CHECK(SurvivorsWikiSpec::CumulativeXPForLevel(41) == 11630.f);
}

TEST_CASE("Survivors wiki temporary growth bonus applies only at level twenty and forty", "[unit][survivors][logic][wiki-spec][growth]")
{
	CHECK(SurvivorsWikiSpec::TemporaryGrowthBonusForPlayerLevel(19) == 0.f);
	CHECK(SurvivorsWikiSpec::TemporaryGrowthBonusForPlayerLevel(20) == 1.f);
	CHECK(SurvivorsWikiSpec::TemporaryGrowthBonusForPlayerLevel(21) == 0.f);
	CHECK(SurvivorsWikiSpec::TemporaryGrowthBonusForPlayerLevel(40) == 1.f);
	CHECK(SurvivorsWikiSpec::TemporaryGrowthBonusForPlayerLevel(41) == 0.f);
}

TEST_CASE("Survivors wiki growth multiplier applies to XP gain after temporary bonus", "[unit][survivors][logic][wiki-spec][growth]")
{
	CHECK(NearlyEqual(SurvivorsWikiSpec::EffectiveXPGain(10.f, 1.0f, 1), 10.f));
	CHECK(NearlyEqual(SurvivorsWikiSpec::EffectiveXPGain(10.f, 1.4f, 1), 14.f));
	CHECK(NearlyEqual(SurvivorsWikiSpec::EffectiveXPGain(10.f, 1.0f, 20), 20.f));
	CHECK(NearlyEqual(SurvivorsWikiSpec::EffectiveXPGain(10.f, 1.4f, 20), 24.f));
	CHECK(NearlyEqual(SurvivorsWikiSpec::EffectiveXPGain(10.f, -2.0f, 1), 0.f));
}

TEST_CASE("Survivors wiki Spellbinder gives ten percent duration per level", "[unit][survivors][logic][wiki-spec][passive]")
{
	CHECK(NearlyEqual(SurvivorsWikiSpec::SpellbinderDurationBonusForLevel(1), 0.10f));
	CHECK(NearlyEqual(SurvivorsWikiSpec::SpellbinderDurationBonusForLevel(3), 0.30f));
	CHECK(NearlyEqual(SurvivorsWikiSpec::SpellbinderDurationBonusForLevel(5), 0.50f));
}

TEST_CASE("Survivors wiki Wings gives ten percent move speed per level", "[unit][survivors][logic][wiki-spec][passive]")
{
	CHECK(NearlyEqual(SurvivorsWikiSpec::WingsMoveSpeedBonusForLevel(1), 0.10f));
	CHECK(NearlyEqual(SurvivorsWikiSpec::WingsMoveSpeedBonusForLevel(3), 0.30f));
	CHECK(NearlyEqual(SurvivorsWikiSpec::WingsMoveSpeedBonusForLevel(5), 0.50f));
}

TEST_CASE("Survivors wiki Pummarola gives zero point two recovery per level", "[unit][survivors][logic][wiki-spec][passive]")
{
	CHECK(NearlyEqual(SurvivorsWikiSpec::PummarolaRecoveryForLevel(1), 0.2f));
	CHECK(NearlyEqual(SurvivorsWikiSpec::PummarolaRecoveryForLevel(3), 0.6f));
	CHECK(NearlyEqual(SurvivorsWikiSpec::PummarolaRecoveryForLevel(5), 1.0f));
}

TEST_CASE("Survivors wiki Crown gives eight percent growth per level", "[unit][survivors][logic][wiki-spec][passive]")
{
	CHECK(NearlyEqual(SurvivorsWikiSpec::CrownGrowthMultiplierForLevel(1), 1.08f));
	CHECK(NearlyEqual(SurvivorsWikiSpec::CrownGrowthMultiplierForLevel(3), 1.24f));
	CHECK(NearlyEqual(SurvivorsWikiSpec::CrownGrowthMultiplierForLevel(5), 1.40f));
}

TEST_CASE("Survivors wiki Attractorb uses multiplicative pickup radius totals", "[unit][survivors][logic][wiki-spec][passive]")
{
	CHECK(NearlyEqual(SurvivorsWikiSpec::AttractorbPickupRadiusMultiplierForLevel(0), 1.0f));
	CHECK(NearlyEqual(SurvivorsWikiSpec::AttractorbPickupRadiusMultiplierForLevel(1), 1.5f));
	CHECK(NearlyEqual(SurvivorsWikiSpec::AttractorbPickupRadiusMultiplierForLevel(2), 1.995f));
	CHECK(NearlyEqual(SurvivorsWikiSpec::AttractorbPickupRadiusMultiplierForLevel(3), 2.49375f));
	CHECK(NearlyEqual(SurvivorsWikiSpec::AttractorbPickupRadiusMultiplierForLevel(4), 2.9925f));
	CHECK(NearlyEqual(SurvivorsWikiSpec::AttractorbPickupRadiusMultiplierForLevel(5), 3.980025f));
	CHECK(NearlyEqual(SurvivorsWikiSpec::AttractorbPickupRadiusMultiplierForLevel(6), 3.980025f));
}

TEST_CASE("Survivors wiki Torrona's Box gives omni through level eight and curse at level nine", "[unit][survivors][logic][wiki-spec][passive]")
{
	CHECK(NearlyEqual(SurvivorsWikiSpec::TorronasOmniBonusForLevel(0), 0.f));
	CHECK(NearlyEqual(SurvivorsWikiSpec::TorronasOmniBonusForLevel(1), 0.04f));
	CHECK(NearlyEqual(SurvivorsWikiSpec::TorronasOmniBonusForLevel(2), 0.07f));
	CHECK(NearlyEqual(SurvivorsWikiSpec::TorronasOmniBonusForLevel(8), 0.25f));
	CHECK(NearlyEqual(SurvivorsWikiSpec::TorronasOmniBonusForLevel(9), 0.25f));
	CHECK(NearlyEqual(SurvivorsWikiSpec::TorronasCurseBonusForLevel(8), 0.f));
	CHECK(NearlyEqual(SurvivorsWikiSpec::TorronasCurseBonusForLevel(9), 1.f));
}

TEST_CASE("Survivors wiki passive max levels exclude coin-only Stone Mask but keep Clover as an evolution key", "[unit][survivors][logic][wiki-spec][passive]")
{
	CHECK(SurvivorsWikiSpec::PassiveTypeCount == 18);
	CHECK(SurvivorsWikiSpec::PassiveMaxLevel[0] == 0);
	CHECK(SurvivorsWikiSpec::PassiveMaxLevel[12] == 5);
	CHECK(SurvivorsWikiSpec::PassiveMaxLevel[13] == 5);
	CHECK(SurvivorsWikiSpec::PassiveMaxLevel[14] == 0);
	CHECK(SurvivorsWikiSpec::PassiveMaxLevel[16] == 2);
	CHECK(SurvivorsWikiSpec::PassiveMaxLevel[17] == 9);
}

TEST_CASE("Survivors wiki gem colors are selected by contained XP value", "[unit][survivors][logic][wiki-spec][gem]")
{
	CHECK(SurvivorsWikiSpec::BlueGemMaxXP == 2.f);
	CHECK(SurvivorsWikiSpec::GreenGemMaxXP == 9.f);
	CHECK(SurvivorsWikiSpec::RedGemMinXP == 10.f);
	CHECK(SurvivorsWikiSpec::GemColorForExperience(1.f) == SurvivorsWikiSpec::EGemColor::Blue);
	CHECK(SurvivorsWikiSpec::GemColorForExperience(2.f) == SurvivorsWikiSpec::EGemColor::Blue);
	CHECK(SurvivorsWikiSpec::GemColorForExperience(3.f) == SurvivorsWikiSpec::EGemColor::Green);
	CHECK(SurvivorsWikiSpec::GemColorForExperience(9.f) == SurvivorsWikiSpec::EGemColor::Green);
	CHECK(SurvivorsWikiSpec::GemColorForExperience(10.f) == SurvivorsWikiSpec::EGemColor::Red);
}

TEST_CASE("Survivors red gem pickup uses ten to fifty times the normal XP", "[unit][survivors][logic][wiki-spec][gem]")
{
	CHECK(SurvivorsWikiSpec::RedGemMinMultiplier == 10);
	CHECK(SurvivorsWikiSpec::RedGemMaxMultiplier == 50);
	CHECK(SurvivorsWikiSpec::RedGemExperienceForMultiplier(2.f, SurvivorsWikiSpec::RedGemMinMultiplier) == 20.f);
	CHECK(SurvivorsWikiSpec::RedGemExperienceForMultiplier(2.f, SurvivorsWikiSpec::RedGemMaxMultiplier) == 100.f);
	CHECK(SurvivorsWikiSpec::RedGemExperienceForMultiplier(9.f, SurvivorsWikiSpec::RedGemMinMultiplier) == 90.f);
	CHECK(SurvivorsWikiSpec::RedGemExperienceForMultiplier(9.f, SurvivorsWikiSpec::RedGemMaxMultiplier) == 450.f);
}

TEST_CASE("Survivors wiki weapon level caps distinguish base, Laurel, and evolved weapons", "[unit][survivors][logic][wiki-spec][weapon]")
{
	CHECK(SurvivorsWikiSpec::BaseWeaponMaxLevel == 8);
	CHECK(SurvivorsWikiSpec::LaurelMaxLevel == 7);
	CHECK(SurvivorsWikiSpec::EvolvedWeaponMaxLevel == 1);
}
