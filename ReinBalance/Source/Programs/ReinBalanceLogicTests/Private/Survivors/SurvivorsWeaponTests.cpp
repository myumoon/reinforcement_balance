/**
 * Survivors 全 content の reset/step 有限性と到達性を表駆動で検証する LLT。
 * 初心者向け:
 * 武器・パッシブ・敵を一種類ずつ初期状態へ入れ、観測が壊れず ID と level が保持されるか確認する。
 */
#include "TestHarness.h"

#include "Math/UnrealMathUtility.h"
#include "Survivors/SurvivorsGameConstants.h"
#include "Survivors/SurvivorsGameLogic.h"

/**
 * production の非公開 choice/evolution/default table を LLT から実行する friend adapter。
 * 初心者向け:
 * テスト専用の値を作らず、ゲーム本体が使う処理と既定表をそのまま検査します。
 */
struct FSurvivorsGameTestAccess
{
	static TArray<FLevelUpChoice> BuildChoices(FSurvivorsGameLogic& Logic) { return Logic.BuildLevelUpChoices(); }
	static void ApplyChoice(FSurvivorsGameLogic& Logic, const FLevelUpChoice& Choice)
	{
		Logic.ApplyLevelUpChoice(Choice);
		Logic.RecalcPassiveEffects();
	}
	static const TArray<FEnemyTypeParams>& EnemyTable(const FSurvivorsGameLogic& Logic)
	{
		return Logic.CurrentConfig.EnemyTypeTable;
	}
};

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

/**
 * 指定武器の実在する level-up choice を返す。
 * 初心者向け:
 * 初期 slot へ目的武器を直接置かず、ゲームが提示した取得・強化候補だけを選びます。
 */
FLevelUpChoice RequireWeaponChoice(FSurvivorsGameLogic& Logic, EWeaponType Type, FLevelUpChoice::EChoiceType ChoiceType)
{
	const TArray<FLevelUpChoice> Choices = FSurvivorsGameTestAccess::BuildChoices(Logic);
	for (const FLevelUpChoice& Choice : Choices)
	{
		if (Choice.WeaponType == Type && Choice.ChoiceType == ChoiceType) return Choice;
	}
	FAIL("required production level-up choice was not offered");
	return FLevelUpChoice();
}

/**
 * passive 合成結果が既定の no-combat 値から変化したかを返す。
 * 初心者向け:
 * 各 field を名前で比較し、構造体の並びや padding に依存しないようにします。
 */
bool HasCombatPassiveEffect(const FPassiveEffects& Value)
{
	const FPassiveEffects Default;
	return !FMath::IsNearlyEqual(Value.DamageMult, Default.DamageMult)
		|| !FMath::IsNearlyEqual(Value.CooldownMult, Default.CooldownMult)
		|| !FMath::IsNearlyEqual(Value.AreaMult, Default.AreaMult)
		|| !FMath::IsNearlyEqual(Value.SpeedMult, Default.SpeedMult)
		|| !FMath::IsNearlyEqual(Value.DurationMult, Default.DurationMult)
		|| !FMath::IsNearlyEqual(Value.ExtraAmount, Default.ExtraAmount)
		|| !FMath::IsNearlyEqual(Value.MoveSpeedMult, Default.MoveSpeedMult)
		|| !FMath::IsNearlyEqual(Value.PickupRadiusMult, Default.PickupRadiusMult)
		|| !FMath::IsNearlyEqual(Value.HpMult, Default.HpMult)
		|| !FMath::IsNearlyEqual(Value.GrowthMult, Default.GrowthMult)
		|| !FMath::IsNearlyEqual(Value.CurseMult, Default.CurseMult)
		|| !FMath::IsNearlyEqual(Value.RegenPerSec, Default.RegenPerSec)
		|| !FMath::IsNearlyEqual(Value.ArmorFlat, Default.ArmorFlat)
		|| Value.MaxRevivalCount != Default.MaxRevivalCount;
}

/**
 * observation segment の先頭 offset を schema から解決する。
 * 初心者向け:
 * 固定の数値位置を手書きせず、production schema の並びを使って対象 field を検査します。
 */
int32 FindObservationOffset(const FSurvivorsGameLogic& Logic, const TCHAR* SegmentName)
{
	int32 Offset = 0;
	for (const FSurvivorsObsSegment& Segment : Logic.GetObsSchema())
	{
		if (Segment.Name == SegmentName) return Offset;
		Offset += Segment.Dim;
	}
	FAIL("required observation segment was not found");
	return INDEX_NONE;
}

/**
 * passive 1-17 の production effect summary を種類ごとに検証する。
 * 初心者向け:
 * 「何かが変わった」だけでなく、それぞれが担当する能力値を明示して確認します。
 */
void CheckPassiveEffectSummary(EPassiveItemType Type, const FPassiveEffects& Effects)
{
	switch (Type)
	{
	case EPassiveItemType::Spinach: CHECK(Effects.DamageMult > 1.f); break;
	case EPassiveItemType::Armor: CHECK(Effects.ArmorFlat > 0.f); break;
	case EPassiveItemType::HollowHeart: CHECK(Effects.HpMult > 1.f); break;
	case EPassiveItemType::Pummarola: CHECK(Effects.RegenPerSec > 0.f); break;
	case EPassiveItemType::EmptyTome: CHECK(Effects.CooldownMult < 1.f); break;
	case EPassiveItemType::Candelabrador: CHECK(Effects.AreaMult > 1.f); break;
	case EPassiveItemType::Bracer: CHECK(Effects.SpeedMult > 1.f); break;
	case EPassiveItemType::Spellbinder: CHECK(Effects.DurationMult > 1.f); break;
	case EPassiveItemType::Duplicator: CHECK(Effects.ExtraAmount > 0.f); break;
	case EPassiveItemType::Wings: CHECK(Effects.MoveSpeedMult > 1.f); break;
	case EPassiveItemType::Attractorb: CHECK(Effects.PickupRadiusMult > 1.f); break;
	case EPassiveItemType::Crown: CHECK(Effects.GrowthMult > 1.f); break;
	case EPassiveItemType::SkullOManiac: CHECK(Effects.CurseMult > 1.f); break;
	case EPassiveItemType::Tirajisu: CHECK(Effects.MaxRevivalCount > 0); break;
	case EPassiveItemType::TorronasBox:
		CHECK(Effects.DamageMult > 1.f);
		CHECK(Effects.CurseMult > 1.f);
		break;
	case EPassiveItemType::Clover:
	case EPassiveItemType::StoneMask:
		CHECK_FALSE(HasCombatPassiveEffect(Effects));
		break;
	default: FAIL("unexpected passive type"); break;
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
 * Pentagram/Laurel を starting 除外状態から choice 経由で取得・強化する。
 * 初心者向け:
 * Garlic で開始した後、実際のレベルアップ候補から対象武器を選び、level 2 まで上げます。
 */
TEST_CASE("Survivors excluded starting weapons remain acquirable and upgradeable", "[unit][survivors][content][choice]")
{
	for (const EWeaponType Target : {EWeaponType::Pentagram, EWeaponType::Laurel})
	{
		FSurvivorsGameLogicConfig Config;
		Config.bHasInitialOverride = true;
		Config.InitialWeaponSlots.Add({static_cast<int32>(EWeaponType::Garlic), 1});
		Config.WeaponPoolMode = TEXT("fixed_subset");
		Config.AllowedWeaponTypes.Add(static_cast<int32>(Target));
		Config.bEnablePassives = false;
		FSurvivorsGameLogic Logic;
		Logic.Initialize(Config);
		Logic.Reset(1500 + static_cast<int32>(Target));
		FSurvivorsGameTestAccess::ApplyChoice(
			Logic, RequireWeaponChoice(Logic, Target, FLevelUpChoice::EChoiceType::WeaponNew));
		int32 TargetSlot = INDEX_NONE;
		for (int32 SlotIdx = 0; SlotIdx < SurvivorsGameConstants::MaxWeaponSlots; ++SlotIdx)
			if (Logic.GetWeaponSlot(SlotIdx).Type == Target) TargetSlot = SlotIdx;
		REQUIRE(TargetSlot != INDEX_NONE);
		CHECK(Logic.GetWeaponSlot(TargetSlot).Level.Value == 1);
		FSurvivorsGameTestAccess::ApplyChoice(
			Logic, RequireWeaponChoice(Logic, Target, FLevelUpChoice::EChoiceType::WeaponUpgrade));
		CHECK(Logic.GetWeaponSlot(TargetSlot).Level.Value == 2);
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
		const TArray<float> Observation = Logic.GetObservation();
		const int32 PassiveOffset = FindObservationOffset(Logic, TEXT("passive_slots"));
		REQUIRE(PassiveOffset != INDEX_NONE);
		CHECK(Observation[PassiveOffset] == static_cast<float>(PassiveId)
			/ static_cast<float>(SurvivorsGameConstants::MaxPassiveTypeCountReserved));
		CHECK(Observation[PassiveOffset + 1] == 1.f);
		const FPassiveEffects& Effects = Logic.GetCachedPassiveEffects();
		CheckPassiveEffectSummary(Type, Effects);
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
		FSurvivorsGameLogic Logic;
		Logic.Initialize(Config);
		const TArray<FEnemyTypeParams>& DefaultTable = FSurvivorsGameTestAccess::EnemyTable(Logic);
		REQUIRE(DefaultTable.IsValidIndex(EnemyId));
		const FEnemyTypeParams& Params = DefaultTable[EnemyId];
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
		FSurvivorsGameLogicConfig SpawnConfig = Config;
		SpawnConfig.SpawnWaves = {Wave};
		SpawnConfig.BossSpawnTime = EnemyId == 10 ? 0.f : 100.f;
		Logic.ApplyConfig(SpawnConfig);
		Logic.Reset(3000 + EnemyId);
		Logic.PhysicsStep(8);
		REQUIRE(Logic.GetEnemyCount() > 0);
		CHECK(Logic.GetEnemyType(0) == EnemyId);
		CHECK(FMath::IsFinite(Logic.GetEnemyHP(0)));
		CheckFiniteObservation(Logic);
		const TArray<float> Observation = Logic.GetObservation();
		const int32 EnemyTypeOffset = FindObservationOffset(Logic, TEXT("enemy_type"));
		REQUIRE(EnemyTypeOffset != INDEX_NONE);
		CHECK(Observation[EnemyTypeOffset] == static_cast<float>(EnemyId) / 10.f);
	}
}

/**
 * Gorgeous Moon と Vandalier の prerequisite と slot 消費契約を固定する。
 * 初心者向け:
 * 全進化表を成立させ、production choice を適用して通常置換と union の二枠から一枠への消費を確認する。
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
		FSurvivorsGameLogicConfig Config;
		Config.bHasInitialOverride = true;
		Config.bEnableEvolutions = true;
		Config.InitialWeaponSlots.Add({
			static_cast<int32>(Rule.BaseWeapon), SurvivorsGameConstants::GetWeaponMaxLevel(Rule.BaseWeapon)});
		if (Rule.UnionPartner != EWeaponType::None)
			Config.InitialWeaponSlots.Add({
				static_cast<int32>(Rule.UnionPartner), SurvivorsGameConstants::GetWeaponMaxLevel(Rule.UnionPartner)});
		if (Rule.RequiredPassive != EPassiveItemType::None)
			Config.InitialPassiveSlots.Add({static_cast<int32>(Rule.RequiredPassive), 1});
		FSurvivorsGameLogic Logic;
		Logic.Initialize(Config);
		Logic.Reset(4000 + static_cast<int32>(Rule.EvolvedWeapon));
		FSurvivorsGameTestAccess::ApplyChoice(
			Logic, RequireWeaponChoice(Logic, Rule.EvolvedWeapon, FLevelUpChoice::EChoiceType::WeaponEvolve));
		CHECK(Logic.GetWeaponSlot(0).Type == Rule.EvolvedWeapon);
		CHECK(Logic.GetWeaponSlot(0).Level.Value == 1);
		if (Rule.UnionPartner != EWeaponType::None)
			CHECK(Logic.GetWeaponSlot(1).Type == EWeaponType::None);
	}
	CHECK(bFoundMoon);
	CHECK(bFoundVandalier);
}
