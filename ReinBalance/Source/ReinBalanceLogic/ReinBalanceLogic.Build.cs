using UnrealBuildTool;

public class ReinBalanceLogic : ModuleRules
{
	public ReinBalanceLogic(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[]
		{
			"Core",
			"CoreUObject",
		});

		// Engine / UMG / Slate / NNE / UnrealEd / HTTPServer は依存禁止
	}
}
