using UnrealBuildTool;

public class PythonTrainingComm : ModuleRules
{
	public PythonTrainingComm(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[]
		{
			"Core",
			"CoreUObject",
			"Engine",
		});

		PrivateDependencyModuleNames.AddRange(new string[]
		{
			"HTTPServer",
			"HTTP",
			"Json",
			"JsonUtilities",
		});
	}
}
