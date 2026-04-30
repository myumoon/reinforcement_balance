using UnrealBuildTool;

public class ReinBalanceEditor : ModuleRules
{
	public ReinBalanceEditor(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[]
		{
			"Core",
			"CoreUObject",
			"Engine",
			"ReinBalance",
			"PythonTrainingComm",
		});

		PrivateDependencyModuleNames.AddRange(new string[]
		{
			"UnrealEd",
			"HTTPServer", // CoinHttpEnvService で FHttpPath / FHttpRouteHandle を使用
			"Json",       // SurvivorsHttpEnvService の /params JSON パース用
		});
	}
}
