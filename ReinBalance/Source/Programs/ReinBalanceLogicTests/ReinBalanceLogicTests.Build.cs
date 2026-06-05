using UnrealBuildTool;

public class ReinBalanceLogicTests : TestModuleRules
{
	public ReinBalanceLogicTests(ReadOnlyTargetRules Target) : base(Target)
	{
		PrivateDependencyModuleNames.AddRange(new string[]
		{
			"ReinBalanceLogic",
		});
	}
}
