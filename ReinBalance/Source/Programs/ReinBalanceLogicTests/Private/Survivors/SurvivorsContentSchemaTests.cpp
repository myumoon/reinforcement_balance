/**
 * Survivors Logic の one-way content/action schema を検証する LLT。
 * Windows の Low Level Tests で schema が enum/table/config 由来の主要項目を含むことを確認する。
 */
#include "TestHarness.h"

#include "HAL/PlatformFileManager.h"
#include "Misc/FileHelper.h"
#include "Misc/MD5.h"
#include "Misc/Paths.h"
#include "Survivors/Game/SurvivorsGame.h"
#include "Survivors/SurvivorsGameConstants.h"
#include "Survivors/SurvivorsGameLogic.h"
#include "UObject/UObjectGlobals.h"

/**
 * JSON の構造を変えず formatting whitespace だけを除去する。
 * 初心者向け:
 * capture の改行やインデントには依存せず、C++ が出した全 key と値を同じ順で比較します。
 */
static FString CompactCapturedSchema(FString Value)
{
	Value.ReplaceInline(TEXT("\r"), TEXT(""));
	Value.ReplaceInline(TEXT("\n"), TEXT(""));
	Value.ReplaceInline(TEXT("\t"), TEXT(""));
	return Value;
}

/**
 * committed capture と live Logic export の完全一致を検証する。
 * 初心者向け:
 * C++ の content を変えたのに Python 用 capture を更新し忘れると、Windows LLT が失敗します。
 */
TEST_CASE("Survivors live content schema equals committed capture", "[unit][survivors][content][capture]")
{
	FSurvivorsGameLogic Logic;
	FSurvivorsGameLogicConfig Config;
	Logic.Initialize(Config);

	const FString CapturePath = FPaths::ConvertRelativePathToFull(
		FPaths::ProjectDir(),
		TEXT("../Tools/Training/configs/survivors_content_schema_capture_v1.json"));
	FString Captured;
	REQUIRE(FPlatformFileManager::Get().GetPlatformFile().FileExists(*CapturePath));
	REQUIRE(FFileHelper::LoadFileToString(Captured, *CapturePath));

	const FString Live = FString::Printf(
		TEXT("{\"schema_version\":\"survivors.content_schema.v1\",\"content\":%s,\"action_time\":%s}"),
		*Logic.GetContentSchema(),
		*Logic.GetActionTimeSchema());
	CHECK(CompactCapturedSchema(Captured) == Live);
}

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
	CHECK(FacadeContent.Contains(*FString::Printf(
		TEXT("\"id\":\"blue\",\"xp\":%.9g"), SurvivorsGameConstants::GemXPValues[0])));
	CHECK(FacadeContent.Contains(*FString::Printf(
		TEXT("\"id\":\"green\",\"xp\":%.9g"), SurvivorsGameConstants::GemXPValues[1])));
	CHECK(FacadeContent.Contains(*FString::Printf(
		TEXT("\"id\":\"red\",\"xp\":%.9g"), SurvivorsGameConstants::GemXPValues[2])));
}
