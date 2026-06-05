using UnrealBuildTool;

[SupportedPlatforms("Win64")]
public class ReinBalanceLogicTestsTarget : TestTargetRules
{
	public ReinBalanceLogicTestsTarget(TargetInfo Target) : base(Target)
	{
		bNeverCompileAgainstEngine = true;
	}
}
