#include "Misc/AutomationTest.h"
#include "SurvivorsTestHelpers.h"

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsDebugFacadeSpawnDebugUsesLogic,
	"ReinBalance.Survivors.Debug.FacadeSpawnDebugUsesLogic",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FSurvivorsDebugFacadeSpawnDebugUsesLogic::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!S.Create()) return false;

	S.Game->ResetState(TOptional<int32>(123));
	S.Game->PhysicsStep(8);

	const FSurvivorsSpawnDebug SpawnDebug = S.Game->GetSpawnDebug();
	TestTrue("Spawn debug elapsed time follows logic", SpawnDebug.ElapsedTime > 0.f);
	TestEqual("Spawn debug wave is first wave", SpawnDebug.CurrentWaveIndex, 0);

	S.Destroy();
	return true;
}
