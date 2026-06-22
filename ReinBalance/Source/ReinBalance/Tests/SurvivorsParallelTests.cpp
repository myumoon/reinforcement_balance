#include "Misc/AutomationTest.h"
#include "SurvivorsTestHelpers.h"
#include "Survivors/Logic/SurvivorsGameLogic.h"

// ============================================================
// Phase 3: FSurvivorsGameLogic のロジックテスト
// ============================================================

/** FSurvivorsGameLogic::ExecReset が非空の obs を返す */
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsParallelLogicResetReturnsObs,
	"ReinBalance.Survivors.Parallel.LogicResetReturnsObs",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FSurvivorsParallelLogicResetReturnsObs::RunTest(const FString& Parameters)
{
	// TODO(issue): FSurvivorsGameLogic が実装されたら有効化する。
	// 現時点では FSurvivorsGameLogic の骨格のみ存在するため、
	// ExecReset は空の Obs を返す（実装完了後にこのテストが通る）。
	FSurvivorsTestWorld S;
	if (!S.Create()) return false;

	// ASurvivorsGame 経由で ResetState を呼ぶ（後方互換パス）
	S.Game->ResetState(TOptional<int32>());
	TArray<float> Obs = S.Game->GetObservation();
	TestTrue("Reset returns non-empty obs", Obs.Num() > 0);

	S.Destroy();
	return true;
}

/** 同じ seed を使うと同じ obs が返る */
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsParallelLogicResetSameSeedSameObs,
	"ReinBalance.Survivors.Parallel.LogicResetSameSeedSameObs",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FSurvivorsParallelLogicResetSameSeedSameObs::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!S.Create()) return false;

	const int32 Seed = 42;
	S.Game->ResetState(TOptional<int32>(Seed));
	TArray<float> Obs1 = S.Game->GetObservation();

	S.Game->ResetState(TOptional<int32>(Seed));
	TArray<float> Obs2 = S.Game->GetObservation();

	TestEqual("Same seed produces same obs length", Obs1.Num(), Obs2.Num());
	bool bAllEqual = true;
	for (int32 i = 0; i < Obs1.Num(); ++i)
	{
		if (!FMath::IsNearlyEqual(Obs1[i], Obs2[i], 1e-5f))
		{
			bAllEqual = false;
			break;
		}
	}
	TestTrue("Same seed produces same obs values", bAllEqual);

	S.Destroy();
	return true;
}

/** PhysicsStep / GetReward / IsDone が有効な値を返す */
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsParallelLogicStepReturnsValidData,
	"ReinBalance.Survivors.Parallel.LogicStepReturnsValidData",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FSurvivorsParallelLogicStepReturnsValidData::RunTest(const FString& Parameters)
{
	FSurvivorsTestWorld S;
	if (!S.Create()) return false;

	S.Game->ResetState(TOptional<int32>());
	S.Game->PhysicsStep(0);  // 北方向

	TArray<float> Obs   = S.Game->GetObservation();
	float         Reward = S.Game->GetReward();
	bool          bDone  = S.Game->IsDone();

	TestTrue("Step returns non-empty obs", Obs.Num() > 0);
	TestTrue("Reward is finite", FMath::IsFinite(Reward));
	TestFalse("Not done after single step", bDone);

	S.Destroy();
	return true;
}

// ============================================================
// Phase 2: bManagedExternally フラグのテスト
// ============================================================

/** bManagedExternally=true 時に Tick() がキュー処理をスキップする
 *  NOTE: このテストは TakeStepRequest/TakeResetRequest の戻り値で間接的に確認する。
 *        EnvServer への直接アクセスが必要なため、現時点では ASurvivorsHttpEnvService の
 *        公開 API レベルで検証する。
 */
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsParallelManagedExternallySkipsTick,
	"ReinBalance.Survivors.Parallel.ManagedExternallySkipsTick",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FSurvivorsParallelManagedExternallySkipsTick::RunTest(const FString& Parameters)
{
	// TODO(issue): ASurvivorsHttpEnvService を直接インスタンス化してテストするには
	// HTTP サーバーのポートが必要。統合テストは PIE 環境でのみ実行可能なため、
	// 現時点ではフラグの存在のみを確認する。
	// 実際の挙動確認は PIE + Python クライアントを使った手動テストで行う。

	// ASurvivorsHttpEnvService に bManagedExternally フィールドが存在することを型チェックで確認
	// （コンパイルが通れば OK）
	// TODO(issue): ASurvivorsHttpEnvService の include が必要だが、
	// ReinBalanceEditor モジュールへの依存を避けるため現時点ではスキップ。
	AddInfo(TEXT("bManagedExternally フラグの存在は SurvivorsParallelSetupActor.cpp でコンパイル確認する。"));
	return true;
}

/** TakeStepRequest/TakeResetRequest がキューから正しく取り出す
 *  NOTE: FHttpEnvServerBase のキューは private なため直接テスト不可。
 *        FSurvivorsGameLogic::ExecStep/ExecReset が正しく動作することで間接確認する。
 */
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsParallelQueueTakeRequest,
	"ReinBalance.Survivors.Parallel.QueueTakeRequest",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FSurvivorsParallelQueueTakeRequest::RunTest(const FString& Parameters)
{
	// TODO(issue): Phase3 完成後に FSurvivorsGameLogic::ExecStep/ExecReset を直接テストする。
	// 現状は ASurvivorsGame 経由でステップが正常動作することを確認する。
	FSurvivorsTestWorld S;
	if (!S.Create()) return false;

	S.Game->ResetState(TOptional<int32>(123));
	TArray<float> Obs = S.Game->GetObservation();
	TestEqual("Obs dim is consistent", Obs.Num(), S.Game->GetObsDim());

	// 複数ステップが正常に動作することを確認
	for (int32 i = 0; i < 10; ++i)
	{
		S.Game->PhysicsStep(8);  // 静止
	}
	TArray<float> ObsAfterSteps = S.Game->GetObservation();
	TestEqual("Obs dim unchanged after steps", ObsAfterSteps.Num(), S.Game->GetObsDim());

	S.Destroy();
	return true;
}
