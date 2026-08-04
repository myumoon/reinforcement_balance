using UnrealBuildTool;

[SupportedPlatforms("Win64")]
public class ReinBalanceLogicTestsTarget : TestTargetRules
{
	public ReinBalanceLogicTestsTarget(TargetInfo Target) : base(Target)
	{
		// ASurvivorsGame facade の委譲を実オブジェクトで検証するため Engine module を有効化する。
		// schema の文字列断片だけでなく、production facade と純粋 Logic の同一性を LLT で確認する。
		bCompileAgainstEngine = true;
		bCompileAgainstApplicationCore = true;
		bCompileAgainstCoreUObject = true;
		bUsesSlate = false;
		bUsePlatformFileStub = true;
		bMockEngineDefaults = true;
		GlobalDefinitions.Add("WITH_AUTOMATION_TESTS=1");
		GlobalDefinitions.Add("WITH_REINBALANCE_LOGIC_TESTS=1");
	}
}
