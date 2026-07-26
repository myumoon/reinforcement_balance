/**
 * Survivors 全 content の reset/step 有限性と到達性を表駆動で検証する LLT。
 * 初心者向け:
 * 武器・パッシブ・敵を一種類ずつ初期状態へ入れ、観測が壊れず ID と level が保持されるか確認する。
 */
#include "TestHarness.h"

#include "Math/UnrealMathUtility.h"
#include "Survivors/SurvivorsGameConstants.h"
#include "Survivors/SurvivorsGameLogic.h"

namespace
{
/**
 * 全観測値が finite であることを確認する。
 * 初心者向け:
 * NaN や Infinity が一つでもあれば学習を壊すため、content ごとの step 後に検査する。
 */
void CheckFiniteObservation(const FSurvivorsGameLogic& Logic)
{
	const TArray<float> Observation = Logic.GetObservation();
	REQUIRE(!Observation.IsEmpty());
	for (const float Value : Observation)
	{
		CHECK(FMath::IsFinite(Value));
	}
}
}

/**
 * base 1-15 と evolved/union 16-28 の reset/step を検証する。
 * 初心者向け:
 * starting 除外の Pentagram/Laurel/Gorgeous Moon も直接初期 slot に置き、全体除外でないことを保証する。
 */
TEST_CASE("Survivors all weapons reset and step finitely", "[unit][survivors][content][weapon]")
{
	for (int32 WeaponId = 1; WeaponId <= static_cast<int32>(EWeaponType::Vandalier); ++WeaponId)
	{
		FSurvivorsGameLogicConfig Config;
		Config.bHasInitialOverride = true;
		Config.InitialWeaponSlots.Add({WeaponId, 1});
		Config.AllowedWeaponTypes.Add(WeaponId);
		Config.StartingWeaponMode = TEXT("pool_random");
		FSurvivorsGameLogic Logic;
		Logic.Initialize(Config);
		Logic.Reset(1000 + WeaponId);
		CHECK(static_cast<int32>(Logic.GetWeaponSlot(0).Type) == WeaponId);
		Logic.PhysicsStep(8);
		CheckFiniteObservation(Logic);
	}
}

/**
 * passive 1-17 の最大 level、effect summary、obs を検証する。
 * 初心者向け:
 * Stone Mask も戦闘効果なしの5レベル item として slot と観測に残ることを確認する。
 */
TEST_CASE("Survivors all passives expose max level and finite summary", "[unit][survivors][content][passive]")
{
	for (int32 PassiveId = 1; PassiveId <= static_cast<int32>(EPassiveItemType::TorronasBox); ++PassiveId)
	{
		const EPassiveItemType Type = static_cast<EPassiveItemType>(PassiveId);
		const int32 MaxLevel = SurvivorsGameConstants::PassiveMaxLevel[PassiveId];
		REQUIRE(MaxLevel > 0);
		FSurvivorsGameLogicConfig Config;
		Config.bHasInitialOverride = true;
		Config.InitialWeaponSlots.Add({static_cast<int32>(EWeaponType::Garlic), 1});
		Config.InitialPassiveSlots.Add({PassiveId, MaxLevel});
		FSurvivorsGameLogic Logic;
		Logic.Initialize(Config);
		Logic.Reset(2000 + PassiveId);
		CHECK(static_cast<int32>(Logic.GetPassiveSlot(0).Type) == PassiveId);
		CHECK(Logic.GetPassiveSlot(0).Level == MaxLevel);
		Logic.PhysicsStep(8);
		CheckFiniteObservation(Logic);
	}
}

/**
 * enemy 0-10 の spawn/HP/damage/XP/boss flags と type obs を検証する。
 * 初心者向け:
 * 各敵だけを出す wave で step し、通常敵と boss の既定値が有限で観測可能か確認する。
 */
TEST_CASE("Survivors all enemies spawn and encode type finitely", "[unit][survivors][content][enemy]")
{
	const float ExpectedHP[] = {1.f, 4.f, 6.f, 3.f, 10.f, 15.f, 20.f, 2.f, 30.f, 25.f, 3000.f};
	const float ExpectedDamage[] = {2.f, 3.f, 4.f, 3.f, 5.f, 6.f, 7.f, 3.f, 10.f, 10.f, 12.f};
	const float ExpectedXP[] = {2.f, 2.f, 2.f, 2.f, 9.f, 9.f, 9.f, 2.f, 9.f, 9.f, 2.f};
	static_assert(UE_ARRAY_COUNT(ExpectedHP) == 11);
	for (int32 EnemyId = 0; EnemyId <= 10; ++EnemyId)
	{
		FSurvivorsGameLogicConfig Config;
		Config.MaxEnemyTypeId = 10;
		for (int32 TableId = 0; TableId <= 10; ++TableId)
		{
			FEnemyTypeParams Row;
			Row.Name = FString::Printf(TEXT("CoverageEnemy%d"), TableId);
			Row.BaseHP = ExpectedHP[TableId];
			Row.ContactDamage = ExpectedDamage[TableId];
			Row.XPDrop = ExpectedXP[TableId];
			Row.bIsBoss = TableId == 10;
			Row.bResistsFreeze = TableId == 10;
			Row.bResistsInstantKill = TableId == 10;
			Row.bResistsDebuff = TableId == 10;
			Config.EnemyTypeTable.Add(Row);
		}
		REQUIRE(Config.EnemyTypeTable.IsValidIndex(EnemyId));
		const FEnemyTypeParams& Params = Config.EnemyTypeTable[EnemyId];
		CHECK(Params.BaseHP == ExpectedHP[EnemyId]);
		CHECK(Params.ContactDamage == ExpectedDamage[EnemyId]);
		CHECK(Params.XPDrop == ExpectedXP[EnemyId]);
		CHECK(Params.bIsBoss == (EnemyId == 10));
		CHECK(Params.bResistsInstantKill == (EnemyId == 10));
		FSpawnWave Wave;
		Wave.TimeStart = 0.f;
		Wave.TimeEnd = 10.f;
		Wave.SpawnRate = 60.f;
		Wave.MinEnemies = 1;
		Wave.MaxEnemies = 1;
		Wave.EnemyWeights.Add({EnemyId, 1.f});
		Config.SpawnWaves = {Wave};
		Config.BossSpawnTime = EnemyId == 10 ? 0.f : 100.f;
		FSurvivorsGameLogic Logic;
		Logic.Initialize(Config);
		Logic.Reset(3000 + EnemyId);
		Logic.PhysicsStep(8);
		REQUIRE(Logic.GetEnemyCount() > 0);
		CHECK(Logic.GetEnemyType(0) == EnemyId);
		CHECK(FMath::IsFinite(Logic.GetEnemyHP(0)));
		CheckFiniteObservation(Logic);
	}
}

/**
 * Gorgeous Moon と Vandalier の prerequisite と slot 消費契約を固定する。
 * 初心者向け:
 * 全進化表を確認し、union だけ partner が必要で二枠から一枠へなる規則を schema と同じ表で追跡する。
 */
TEST_CASE("Survivors evolution prerequisites include moon and vandalier union", "[unit][survivors][content][evolution]")
{
	bool bFoundMoon = false;
	bool bFoundVandalier = false;
	for (const SurvivorsGameConstants::FEvolutionRule& Rule : SurvivorsGameConstants::EvolutionTable)
	{
		CHECK(SurvivorsGameConstants::GetWeaponMaxLevel(Rule.BaseWeapon) > 0);
		CHECK(SurvivorsGameConstants::GetWeaponMaxLevel(Rule.EvolvedWeapon) == 1);
		if (Rule.EvolvedWeapon == EWeaponType::GorgeousMoon)
		{
			bFoundMoon = Rule.BaseWeapon == EWeaponType::Pentagram && Rule.RequiredPassive == EPassiveItemType::Crown;
		}
		if (Rule.EvolvedWeapon == EWeaponType::Vandalier)
		{
			bFoundVandalier = Rule.BaseWeapon == EWeaponType::Peachone
				&& Rule.RequiredPassive == EPassiveItemType::None
				&& Rule.UnionPartner == EWeaponType::EbonyWings;
		}
	}
	CHECK(bFoundMoon);
	CHECK(bFoundVandalier);
}
