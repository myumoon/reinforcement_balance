/**
 * Survivors Logic の one-way content/action schema を検証する LLT。
 * Windows の Low Level Tests で schema が enum/table/config 由来の主要項目を含むことを確認する。
 */
#include "TestHarness.h"

#include "Misc/MD5.h"
#include "Survivors/Game/SurvivorsGame.h"
#include "Survivors/SurvivorsGameLogic.h"
#include "UObject/UObjectGlobals.h"

/**
 * facade→Logic の schema bytes と hash 整合を検証する。
 * HTTP が利用する facade が独自 schema を生成せず、Logic の同じ内容をそのまま公開することを確認する。
 */
TEST_CASE("Survivors facade delegates canonical fidelity schema to logic", "[unit][survivors][logic][fidelity]")
{
	FSurvivorsGameLogic Logic;
	FSurvivorsGameLogicConfig Config;
	Logic.Initialize(Config);

	const ASurvivorsGame* Facade = GetDefault<ASurvivorsGame>();
	REQUIRE(Facade != nullptr);

	const FString LogicContent = Logic.GetContentSchema();
	const FString LogicActionTime = Logic.GetActionTimeSchema();
	const FString FacadeContent = Facade->GetContentSchema();
	const FString FacadeActionTime = Facade->GetActionTimeSchema();

	CHECK(FacadeContent == LogicContent);
	CHECK(FacadeActionTime == LogicActionTime);
	CHECK(FMD5::HashAnsiString(*FacadeContent) == FMD5::HashAnsiString(*LogicContent));
	CHECK(FMD5::HashAnsiString(*FacadeActionTime) == FMD5::HashAnsiString(*LogicActionTime));
	CHECK(FacadeContent.Contains(TEXT("\"xp_curve\"")));
	CHECK(FacadeActionTime.Contains(TEXT("\"directions\"")));
}
