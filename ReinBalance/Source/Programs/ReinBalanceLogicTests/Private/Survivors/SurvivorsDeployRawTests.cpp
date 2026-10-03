/**
 * Survivors の deploy raw state と軌跡決定性を検証する LLT。
 *
 * 固定シード・固定行動列で進めたときの flat obs 列を 1 つの digest に畳み込み、
 * 変更前に記録した golden 値と一致することで「乱数消費・ゲーム進行・obs が変わっていない」ことを確かめる。
 */
#include "TestHarness.h"

#include "HAL/PlatformMisc.h"
#include "Misc/Crc.h"
#include "Misc/Paths.h"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <string>
#include "Survivors/SurvivorsDeployRaw.h"
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
	uint32 ComputeTrajectoryDigest(FSurvivorsGameLogic& Logic, int32 Steps, bool bBuildDeployRaw = false)
	{
		uint32 Digest = 0;
		for (int32 Step = 0; Step < Steps; ++Step)
		{
			if (bBuildDeployRaw)
			{
				// raw state の生成と JSON 化を毎 step 挟んでも軌跡が変わらないことを確かめる
				SurvivorsDeployRaw::ToJson(Logic.BuildDeployRawState());
			}
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

TEST_CASE("Survivors deploy raw generation does not change trajectory", "[unit][survivors][deploy-raw][determinism]")
{
	using namespace SurvivorsDeployRawTestsLocal;
	{
		FSurvivorsGameLogicConfig Config;
		FSurvivorsGameLogic Logic;
		REQUIRE(Logic.Initialize(Config));
		Logic.Reset(TrajectorySeed);
		CHECK(ComputeTrajectoryDigest(Logic, TrajectorySteps, true) == 1980010005u);
		CHECK(Logic.RandStream.GetCurrentSeed() == 1990668092);
	}
	{
		FSurvivorsGameLogic Logic;
		REQUIRE(Logic.Initialize(MakeAllEffectsConfig()));
		Logic.Reset(TrajectorySeed);
		CHECK(ComputeTrajectoryDigest(Logic, TrajectorySteps * 2, true) == 1376127261u);
		CHECK(Logic.RandStream.GetCurrentSeed() == -528431702);
		CHECK(Logic.GetObsSchemaHash() == TEXT("84a7f054d4a4c8fa7b22dba61a95a5b3"));
	}
}

TEST_CASE("Survivors deploy raw ids are stable per object and new per spawn", "[unit][survivors][deploy-raw][ids]")
{
	using namespace SurvivorsDeployRawTestsLocal;
	FSurvivorsGameLogic Logic;
	REQUIRE(Logic.Initialize(MakeAllEffectsConfig()));
	Logic.Reset(TrajectorySeed);

	// 除外を無効化（巨大な余白）して、画面外へ出入りする見かけの消失を id の再生成と区別できるようにする
	constexpr float NoCull = 1.0e9f;
	TMap<int64, TPair<FString, TOptional<int32>>> Previous;
	TSet<int64> Retired;
	TSet<FString> SeenClasses;
	TSet<int64> OrbitCycles;
	TOptional<int64> FirstAuraId;
	int32 RetiredProjectiles = 0;
	for (int32 Step = 0; Step < TrajectorySteps * 2; ++Step)
	{
		Logic.PhysicsStep(Step % 9);
		const FSurvivorsDeployRawState State = Logic.BuildDeployRawState(NoCull);
		TMap<int64, TPair<FString, TOptional<int32>>> Current;
		for (const FSurvivorsDeployRawEntity& E : State.Entities)
		{
			INFO("step " << Step << " id " << E.EntityId);
			// 同 tick 内で id が重複しない
			CHECK(!Current.Contains(E.EntityId));
			// 一度消えた id は二度と現れない（再生成は必ず新 id）
			CHECK(!Retired.Contains(E.EntityId));
			Current.Add(E.EntityId, {E.ClassName, E.Slot});
			SeenClasses.Add(E.ClassName);
			// tick をまたいで同じ id は同じ種類・同じスロット
			if (const auto* Prev = Previous.Find(E.EntityId))
			{
				CHECK(Prev->Key == E.ClassName);
				CHECK(Prev->Value == E.Slot);
			}
			if (E.ClassName == TEXT("weapon_orbit"))
			{
				OrbitCycles.Add((E.EntityId & ((int64(1) << SurvivorsDeployRaw::LocalIdBits) - 1)) / 256);
			}
			if (E.ClassName == TEXT("weapon_aura"))
			{
				if (!FirstAuraId.IsSet()) { FirstAuraId = E.EntityId; }
				CHECK(E.EntityId == FirstAuraId.GetValue());
			}
		}
		for (const auto& Pair : Previous)
		{
			if (!Current.Contains(Pair.Key))
			{
				Retired.Add(Pair.Key);
				RetiredProjectiles += Pair.Value.Key == TEXT("weapon_projectile") ? 1 : 0;
			}
		}
		Previous = MoveTemp(Current);
	}

	// 4 種類の武器エフェクトと敵・ジェムが全て出て、projectile と King Bible の周期が実際に入れ替わった
	for (const TCHAR* Name : {TEXT("weapon_projectile"), TEXT("weapon_zone"), TEXT("weapon_orbit"),
		TEXT("weapon_aura"), TEXT("enemy_normal"), TEXT("gem_blue")})
	{
		INFO(TCHAR_TO_UTF8(Name));
		CHECK(SeenClasses.Contains(Name));
	}
	CHECK(OrbitCycles.Num() >= 2);
	CHECK(RetiredProjectiles > 0);
}

TEST_CASE("Survivors deploy raw ids restart after reset", "[unit][survivors][deploy-raw][ids]")
{
	using namespace SurvivorsDeployRawTestsLocal;
	FSurvivorsGameLogic Logic;
	REQUIRE(Logic.Initialize(MakeAllEffectsConfig()));
	Logic.Reset(TrajectorySeed);
	for (int32 Step = 0; Step < 120; ++Step) { Logic.PhysicsStep(Step % 9); }
	const FString First = SurvivorsDeployRaw::ToJson(Logic.BuildDeployRawState());
	CHECK(Logic.NextEffectId > 0);

	// 初期装備 override は Reset 1 回で消費されるので、同じ装備で再開するため Initialize し直す
	REQUIRE(Logic.Initialize(MakeAllEffectsConfig()));
	Logic.Reset(TrajectorySeed);
	CHECK(Logic.NextEffectId == 0);
	for (int32 Step = 0; Step < 120; ++Step) { Logic.PhysicsStep(Step % 9); }
	CHECK(SurvivorsDeployRaw::ToJson(Logic.BuildDeployRawState()) == First);
}

TEST_CASE("Survivors deploy raw culls outside camera plus margin", "[unit][survivors][deploy-raw][cull]")
{
	using namespace SurvivorsGameConstants;
	FSurvivorsGameLogicConfig Config;
	FSurvivorsGameLogic Logic;
	REQUIRE(Logic.Initialize(Config));
	Logic.Reset(1);
	Logic.PlayerPos = FVector2D(100.0, -50.0);
	Logic.Enemies.Empty();
	Logic.Gems.Empty();
	Logic.Projectiles.Empty();
	Logic.GroundZones.Empty();

	const float M = SurvivorsDeployRaw::CullMarginU;
	CHECK(M == 100.f);
	const float Eps = 0.5f;
	// (相対位置, 範囲内か) の境界ケース。ちょうど境界は含む
	const TArray<TPair<FVector2D, bool>> Cases = {
		{FVector2D(ScreenHalfWidthU + M, 0.0), true},
		{FVector2D(-(ScreenHalfWidthU + M), 0.0), true},
		{FVector2D(ScreenHalfWidthU + M + Eps, 0.0), false},
		{FVector2D(0.0, ScreenHalfHeightU + M), true},
		{FVector2D(0.0, -(ScreenHalfHeightU + M + Eps)), false},
		{FVector2D(ScreenHalfWidthU + M, ScreenHalfHeightU + M), true},
		{FVector2D(ScreenHalfWidthU, ScreenHalfHeightU + M + Eps), false},
	};
	for (int32 i = 0; i < Cases.Num(); ++i)
	{
		const FVector2D Pos = Logic.PlayerPos + Cases[i].Key;
		FEnemyState E; E.Pos = Pos; E.UniqueId = 1000 + i; E.CollisionRadius = 8.f;
		Logic.Enemies.Add(E);
		FGemState G; G.Pos = Pos; G.UniqueId = 2000 + i; G.Type = EGemType::Green;
		Logic.Gems.Add(G);
		FProjectileState P; P.Pos = Pos; P.WeaponSlotIdx = 0;
		Logic.SpawnProjectile(P);
		FGroundZoneState Z; Z.Pos = Pos; Z.WeaponSlotIdx = 1;
		Logic.SpawnGroundZone(Z);
	}

	const FSurvivorsDeployRawState State = Logic.BuildDeployRawState();
	CHECK(State.CameraCenter == Logic.PlayerPos);
	CHECK(State.CameraHalfWidth == ScreenHalfWidthU);
	CHECK(State.CameraHalfHeight == ScreenHalfHeightU);
	TSet<int64> Ids;
	for (const FSurvivorsDeployRawEntity& E : State.Entities) { Ids.Add(E.EntityId); }
	for (int32 i = 0; i < Cases.Num(); ++i)
	{
		INFO("case " << i);
		const bool bIn = Cases[i].Value;
		CHECK(Ids.Contains(SurvivorsDeployRaw::MakeEntityId(ESurvivorsDeployRawIdSpace::Enemy, 1000 + i)) == bIn);
		CHECK(Ids.Contains(SurvivorsDeployRaw::MakeEntityId(ESurvivorsDeployRawIdSpace::Gem, 2000 + i)) == bIn);
		CHECK(Ids.Contains(SurvivorsDeployRaw::MakeEntityId(
			ESurvivorsDeployRawIdSpace::Projectile, Logic.Projectiles[i].EffectId)) == bIn);
		CHECK(Ids.Contains(SurvivorsDeployRaw::MakeEntityId(
			ESurvivorsDeployRawIdSpace::GroundZone, Logic.GroundZones[i].EffectId)) == bIn);
	}
	// 種類別の id 空間は衝突しない（同じローカル番号でも別 id）
	CHECK(SurvivorsDeployRaw::MakeEntityId(ESurvivorsDeployRawIdSpace::Projectile, 0)
		!= SurvivorsDeployRaw::MakeEntityId(ESurvivorsDeployRawIdSpace::GroundZone, 0));
}

TEST_CASE("Survivors deploy raw JSON has fixed key layout", "[unit][survivors][deploy-raw][json]")
{
	FSurvivorsDeployRawState S;
	S.CameraCenter = FVector2D(1.5, -2.0);
	S.CameraHalfWidth = 400.f;
	S.CameraHalfHeight = 225.f;
	S.CullMarginU = 100.f;
	S.PlayerPos = FVector2D(1.5, -2.0);
	S.HpRatio = 0.75f;
	S.PlayerLevel = 3;
	S.ElapsedS = 12.25f;
	S.DurationMult = 1.1f;
	S.WeaponSlots.Add({0, 7, 2});
	S.PassiveSlots.Add({0, 0, 0});
	FSurvivorsDeployRawEntity Enemy;
	Enemy.EntityId = SurvivorsDeployRaw::MakeEntityId(ESurvivorsDeployRawIdSpace::Enemy, 5);
	Enemy.ClassName = TEXT("enemy_boss");
	Enemy.WorldPos = FVector2D(10.0, 20.0);
	Enemy.RadiusWorld = 12.f;
	S.Entities.Add(Enemy);
	FSurvivorsDeployRawEntity Zone;
	Zone.EntityId = SurvivorsDeployRaw::MakeEntityId(ESurvivorsDeployRawIdSpace::GroundZone, 9);
	Zone.ClassName = TEXT("weapon_zone");
	Zone.WorldPos = FVector2D(-3.0, 4.0);
	Zone.RadiusWorld = 30.f;
	Zone.Slot = 1;
	Zone.TtlTrueS = 2.5f;
	Zone.bWarning = true;
	S.Entities.Add(Zone);

	CHECK(SurvivorsDeployRaw::ToJson(S) == TEXT(
		"{\"schema_version\":\"survivors_deploy_raw.v1\",\"elapsed_s\":12.25,"
		"\"camera\":{\"center_x\":1.5,\"center_y\":-2,\"half_width\":400,\"half_height\":225,\"cull_margin\":100},"
		"\"player\":{\"world_x\":1.5,\"world_y\":-2,\"hp_ratio\":0.75,\"level\":3},"
		"\"duration_mult\":1.10000002,"
		"\"weapon_slots\":[{\"index\":0,\"type_id\":7,\"level\":2}],"
		"\"passive_slots\":[{\"index\":0,\"type_id\":0,\"level\":0}],"
		"\"entities\":[{\"entity_id\":1099511627781,\"class_name\":\"enemy_boss\",\"world_x\":10,\"world_y\":20,"
		"\"radius_world\":12,\"slot\":null,\"ttl_true_s\":null,\"warning\":false},"
		"{\"entity_id\":4398046511113,\"class_name\":\"weapon_zone\",\"world_x\":-3,\"world_y\":4,"
		"\"radius_world\":30,\"slot\":1,\"ttl_true_s\":2.5,\"warning\":true}]}"));
}

namespace SurvivorsDeployRawTestsLocal
{
	/**
	 * Python テスト用 fixture のリポジトリ内パスを返す。
	 *
	 * このテストソースの位置（__FILE__）から 6 階層上がリポジトリ root。
	 */
	FString FixturePath()
	{
		const FString Dir = FPaths::GetPath(FString(ANSI_TO_TCHAR(__FILE__)));
		FString Path = FPaths::Combine(
			Dir, TEXT("../../../../../../Tools/Training/tests/survivors/fixtures/deploy_raw_llt_v1.json"));
		FPaths::NormalizeFilename(Path);
		FPaths::CollapseRelativeDirectories(Path);
		return Path;
	}

	/**
	 * HTTP 応答の deploy_raw 部分を模した fixture JSON を作る。
	 *
	 * 4 種類の武器エフェクトが出る初期装備で固定シード・固定行動列を進め、
	 * /reset 応答と、連続 tick を含む数個の /step 応答の deploy_raw を並べる。
	 */
	FString BuildFixtureJson()
	{
		FSurvivorsGameLogic Logic;
		if (!Logic.Initialize(MakeAllEffectsConfig())) { return FString(); }
		Logic.Reset(TrajectorySeed);
		TArray<FString> Responses;
		Responses.Add(FString::Printf(
			TEXT("    {\"endpoint\":\"/reset\",\"step\":0,\"deploy_raw\":%s}"),
			*SurvivorsDeployRaw::ToJson(Logic.BuildDeployRawState())));
		const TSet<int32> CaptureSteps = {60, 61, 62, 63, 180, 181, 360};
		for (int32 Step = 1; Step <= 360; ++Step)
		{
			Logic.PhysicsStep((Step - 1) % 9);
			if (CaptureSteps.Contains(Step))
			{
				Responses.Add(FString::Printf(
					TEXT("    {\"endpoint\":\"/step\",\"step\":%d,\"info\":{\"deploy_raw\":%s}}"),
					Step, *SurvivorsDeployRaw::ToJson(Logic.BuildDeployRawState())));
			}
		}
		return FString::Printf(
			TEXT("{\n  \"description\": \"deploy_raw generated by ReinBalanceLogicTests (SurvivorsDeployRawTests.cpp) "
			     "from FSurvivorsGameLogic + SurvivorsDeployRaw::ToJson, the same functions SurvivorsHttpEnvService uses. "
			     "Not captured from a live UE5 PIE (WAITING_MANUAL). Regenerate with REINBALANCE_WRITE_DEPLOY_RAW_FIXTURE=1.\",\n"
			     "  \"seed\": %d,\n  \"action_rule\": \"action of step k (1-based) = (k - 1) %% 9\",\n"
			     "  \"initial_weapons\": [\"Knife\", \"SantaWater\", \"KingBible\", \"Garlic\", \"FireWand\", \"Peachone\"],\n"
			     "  \"obs_schema_hash\": \"%s\",\n  \"responses\": [\n%s\n  ]\n}\n"),
			TrajectorySeed, *Logic.GetObsSchemaHash(), *FString::Join(Responses, TEXT(",\n")));
	}
}

TEST_CASE("Survivors deploy raw fixture for Python matches current producer", "[unit][survivors][deploy-raw][fixture]")
{
	using namespace SurvivorsDeployRawTestsLocal;
	// LLT では UE の file manager が書き込めないため、標準ライブラリの stream で読み書きする
	const std::string Path(TCHAR_TO_UTF8(*FixturePath()));
	const std::string Expected(TCHAR_TO_UTF8(*BuildFixtureJson()));
	INFO("fixture path " << Path);
	if (!FPlatformMisc::GetEnvironmentVariable(TEXT("REINBALANCE_WRITE_DEPLOY_RAW_FIXTURE")).IsEmpty())
	{
		std::ofstream Out(Path, std::ios::binary | std::ios::trunc);
		REQUIRE(Out.good());
		Out << Expected;
	}
	std::ifstream In(Path, std::ios::binary);
	REQUIRE(In.good());
	std::string Actual((std::istreambuf_iterator<char>(In)), std::istreambuf_iterator<char>());
	// checkout 時の改行変換（CRLF）は無視する
	Actual.erase(std::remove(Actual.begin(), Actual.end(), '\r'), Actual.end());
	CHECK(Actual == Expected);
}
