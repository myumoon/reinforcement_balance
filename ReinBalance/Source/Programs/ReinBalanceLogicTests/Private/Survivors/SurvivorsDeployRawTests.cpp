/**
 * Survivors の deploy raw state と軌跡決定性を検証する LLT。
 *
 * 固定シード・固定行動列で進めたときの flat obs 列を 1 つの digest に畳み込み、
 * 変更前に記録した golden 値と一致することで「乱数消費・ゲーム進行・obs が変わっていない」ことを確かめる。
 */
#include "TestHarness.h"

#include "Misc/Crc.h"
#include "Survivors/SurvivorsGameConstants.h"
#include "Survivors/SurvivorsGameLogic.h"

namespace SurvivorsDeployRawTestsLocal
{
	/** 軌跡 digest の対象ステップ数 */
	constexpr int32 TrajectorySteps = 300;
	/** 軌跡 digest に使う固定シード */
	constexpr int32 TrajectorySeed = 73013;

	/**
	 * 固定シード・固定行動列で TrajectorySteps 進め、各 step の obs を CRC32 で畳み込む。
	 *
	 * 行動は step % 9 の固定列。各 step 後の観測ベクトルのバイト列を前回の CRC に連結して計算し、
	 * どこか 1 step でも obs が変われば最終値が変わる。
	 */
	uint32 ComputeTrajectoryDigest(FSurvivorsGameLogic& Logic, int32 Steps)
	{
		uint32 Digest = 0;
		for (int32 Step = 0; Step < Steps; ++Step)
		{
			Logic.PhysicsStep(Step % 9);
			const TArray<float> Obs = Logic.GetObservation();
			Digest = FCrc::MemCrc32(Obs.GetData(), Obs.Num() * sizeof(float), Digest);
		}
		return Digest;
	}

	/**
	 * 4 種類の武器エフェクト（projectile / zone / orbit / aura）が全て出る初期装備の config を作る。
	 *
	 * Knife・SantaWater・KingBible・Garlic・FireWand・Peachone を持たせ、
	 * id 採番を追加した生成経路が軌跡 digest に必ず含まれるようにする。
	 */
	FSurvivorsGameLogicConfig MakeAllEffectsConfig()
	{
		FSurvivorsGameLogicConfig Config;
		Config.bHasInitialOverride = true;
		Config.InitialWeaponSlots.Add({static_cast<int32>(EWeaponType::Knife), 4});
		Config.InitialWeaponSlots.Add({static_cast<int32>(EWeaponType::SantaWater), 4});
		Config.InitialWeaponSlots.Add({static_cast<int32>(EWeaponType::KingBible), 4});
		Config.InitialWeaponSlots.Add({static_cast<int32>(EWeaponType::Garlic), 4});
		Config.InitialWeaponSlots.Add({static_cast<int32>(EWeaponType::FireWand), 4});
		Config.InitialWeaponSlots.Add({static_cast<int32>(EWeaponType::Peachone), 4});
		return Config;
	}
}

TEST_CASE("Survivors trajectory digest matches golden recorded before deploy raw", "[unit][survivors][deploy-raw][determinism]")
{
	using namespace SurvivorsDeployRawTestsLocal;
	// golden は deploy raw 追加前の origin/main (3a825b5) で記録した値。変えてはならない。
	{
		FSurvivorsGameLogicConfig Config;
		FSurvivorsGameLogic Logic;
		REQUIRE(Logic.Initialize(Config));
		Logic.Reset(TrajectorySeed);
		CHECK(ComputeTrajectoryDigest(Logic, TrajectorySteps) == 1980010005u);
		CHECK(Logic.RandStream.GetCurrentSeed() == 1990668092);
		CHECK(Logic.GetObsSchemaHash() == TEXT("84a7f054d4a4c8fa7b22dba61a95a5b3"));
	}
	{
		FSurvivorsGameLogic Logic;
		REQUIRE(Logic.Initialize(MakeAllEffectsConfig()));
		Logic.Reset(TrajectorySeed);
		CHECK(ComputeTrajectoryDigest(Logic, TrajectorySteps * 2) == 1376127261u);
		CHECK(Logic.RandStream.GetCurrentSeed() == -528431702);
	}
}
