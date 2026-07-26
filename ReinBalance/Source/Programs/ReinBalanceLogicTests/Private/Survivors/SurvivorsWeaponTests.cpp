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
	static void DropGem(FSurvivorsGameLogic& Logic, int32 EnemyTypeId)
	{
		Logic.DropGem(EnemyTypeId, FVector2D::ZeroVector);
	}
	static float GemXP(const FSurvivorsGameLogic& Logic, int32 Index)
	{
		return Logic.Gems.IsValidIndex(Index) ? Logic.Gems[Index].BaseExperienceValue : 0.f;
	}
};

namespace
{
/**
 * content coverage の scenario/eval を LLT 実行対象へ束ねる機械可読 registry。
 * 初心者向け:
 * Python 監査はこの表を読み、各 ID のシナリオと確認項目が本当に LLT に登録されているか確かめます。
 * LLT 自身も全行を検査するため、名前だけを置いた未使用の表にはなりません。
 */
struct FContentCoverageCell
{
	const TCHAR* ContentKey;
	const TCHAR* ScenarioId;
	const TCHAR* EvalAssertionId;
};

static const FContentCoverageCell ContentCoverageCells[] = {
	{TEXT("weapon:1"),TEXT("weapon_1_reset_step"),TEXT("weapon_1_finite_assertion")},
	{TEXT("weapon:2"),TEXT("weapon_2_reset_step"),TEXT("weapon_2_finite_assertion")},
	{TEXT("weapon:3"),TEXT("weapon_3_reset_step"),TEXT("weapon_3_finite_assertion")},
	{TEXT("weapon:4"),TEXT("weapon_4_reset_step"),TEXT("weapon_4_finite_assertion")},
	{TEXT("weapon:5"),TEXT("weapon_5_reset_step"),TEXT("weapon_5_finite_assertion")},
	{TEXT("weapon:6"),TEXT("weapon_6_reset_step"),TEXT("weapon_6_finite_assertion")},
	{TEXT("weapon:7"),TEXT("weapon_7_reset_step"),TEXT("weapon_7_finite_assertion")},
	{TEXT("weapon:8"),TEXT("weapon_8_reset_step"),TEXT("weapon_8_finite_assertion")},
	{TEXT("weapon:9"),TEXT("weapon_9_reset_step"),TEXT("weapon_9_finite_assertion")},
	{TEXT("weapon:10"),TEXT("weapon_10_reset_step"),TEXT("weapon_10_finite_assertion")},
	{TEXT("weapon:11"),TEXT("weapon_11_reset_step"),TEXT("weapon_11_finite_assertion")},
	{TEXT("weapon:12"),TEXT("pentagram_acquisition_upgrade"),TEXT("pentagram_choice_level_two")},
	{TEXT("weapon:13"),TEXT("peachone_union_prerequisite"),TEXT("peachone_union_rule")},
	{TEXT("weapon:14"),TEXT("ebony_union_prerequisite"),TEXT("ebony_union_rule")},
	{TEXT("weapon:15"),TEXT("laurel_acquisition_upgrade"),TEXT("laurel_choice_level_two")},
	{TEXT("weapon:16"),TEXT("evolution_1_16"),TEXT("evolution_1_16_slot_replace")},
	{TEXT("weapon:17"),TEXT("evolution_2_17"),TEXT("evolution_2_17_slot_replace")},
	{TEXT("weapon:18"),TEXT("evolution_3_18"),TEXT("evolution_3_18_slot_replace")},
	{TEXT("weapon:19"),TEXT("evolution_4_19"),TEXT("evolution_4_19_slot_replace")},
	{TEXT("weapon:20"),TEXT("evolution_5_20"),TEXT("evolution_5_20_slot_replace")},
	{TEXT("weapon:21"),TEXT("evolution_6_21"),TEXT("evolution_6_21_slot_replace")},
	{TEXT("weapon:22"),TEXT("evolution_7_22"),TEXT("evolution_7_22_slot_replace")},
	{TEXT("weapon:23"),TEXT("evolution_8_23"),TEXT("evolution_8_23_slot_replace")},
	{TEXT("weapon:24"),TEXT("evolution_9_24"),TEXT("evolution_9_24_slot_replace")},
	{TEXT("weapon:25"),TEXT("evolution_10_25"),TEXT("evolution_10_25_slot_replace")},
	{TEXT("weapon:26"),TEXT("evolution_11_26"),TEXT("evolution_11_26_slot_replace")},
	{TEXT("weapon:27"),TEXT("gorgeous_moon_evolution"),TEXT("gorgeous_moon_slot_replace")},
	{TEXT("weapon:28"),TEXT("vandalier_union_slot_consumption"),TEXT("vandalier_partner_slot_empty")},
	{TEXT("passive:1"),TEXT("passive_1_max_effect"),TEXT("passive_1_effect_and_obs")},
	{TEXT("passive:2"),TEXT("passive_2_max_effect"),TEXT("passive_2_effect_and_obs")},
	{TEXT("passive:3"),TEXT("passive_3_max_effect"),TEXT("passive_3_effect_and_obs")},
	{TEXT("passive:4"),TEXT("passive_4_max_effect"),TEXT("passive_4_effect_and_obs")},
	{TEXT("passive:5"),TEXT("passive_5_max_effect"),TEXT("passive_5_effect_and_obs")},
	{TEXT("passive:6"),TEXT("passive_6_max_effect"),TEXT("passive_6_effect_and_obs")},
	{TEXT("passive:7"),TEXT("passive_7_max_effect"),TEXT("passive_7_effect_and_obs")},
	{TEXT("passive:8"),TEXT("passive_8_max_effect"),TEXT("passive_8_effect_and_obs")},
	{TEXT("passive:9"),TEXT("passive_9_max_effect"),TEXT("passive_9_effect_and_obs")},
	{TEXT("passive:10"),TEXT("passive_10_max_effect"),TEXT("passive_10_effect_and_obs")},
	{TEXT("passive:11"),TEXT("passive_11_max_effect"),TEXT("passive_11_effect_and_obs")},
	{TEXT("passive:12"),TEXT("passive_12_max_effect"),TEXT("passive_12_effect_and_obs")},
	{TEXT("passive:13"),TEXT("passive_13_max_effect"),TEXT("passive_13_effect_and_obs")},
	{TEXT("passive:14"),TEXT("stone_mask_no_combat_semantics"),TEXT("stone_mask_no_combat_and_obs")},
	{TEXT("passive:15"),TEXT("passive_15_max_effect"),TEXT("passive_15_effect_and_obs")},
	{TEXT("passive:16"),TEXT("passive_16_max_effect"),TEXT("passive_16_effect_and_obs")},
	{TEXT("passive:17"),TEXT("passive_17_max_effect"),TEXT("passive_17_effect_and_obs")},
	{TEXT("gem:blue"),TEXT("gem_blue_pickup"),TEXT("gem_blue_xp_observed")},
	{TEXT("gem:green"),TEXT("gem_green_pickup"),TEXT("gem_green_xp_observed")},
	{TEXT("gem:red"),TEXT("gem_red_pickup"),TEXT("gem_red_xp_observed")},
	{TEXT("evolution:1:16"),TEXT("evolve_1_16"),TEXT("evolution_1_16_applied")},
	{TEXT("evolution:2:17"),TEXT("evolve_2_17"),TEXT("evolution_2_17_applied")},
	{TEXT("evolution:3:18"),TEXT("evolve_3_18"),TEXT("evolution_3_18_applied")},
	{TEXT("evolution:4:19"),TEXT("evolve_4_19"),TEXT("evolution_4_19_applied")},
	{TEXT("evolution:5:20"),TEXT("evolve_5_20"),TEXT("evolution_5_20_applied")},
	{TEXT("evolution:6:21"),TEXT("evolve_6_21"),TEXT("evolution_6_21_applied")},
	{TEXT("evolution:7:22"),TEXT("evolve_7_22"),TEXT("evolution_7_22_applied")},
	{TEXT("evolution:8:23"),TEXT("evolve_8_23"),TEXT("evolution_8_23_applied")},
	{TEXT("evolution:9:24"),TEXT("evolve_9_24"),TEXT("evolution_9_24_applied")},
	{TEXT("evolution:10:25"),TEXT("evolve_10_25"),TEXT("evolution_10_25_applied")},
	{TEXT("evolution:11:26"),TEXT("evolve_11_26"),TEXT("evolution_11_26_applied")},
	{TEXT("evolution:12:27"),TEXT("gorgeous_moon_12_27"),TEXT("gorgeous_moon_12_27_applied")},
	{TEXT("evolution:13:28"),TEXT("vandalier_union_13_28"),TEXT("vandalier_union_13_28_applied")},
	{TEXT("enemy:0"),TEXT("enemy_0_reset_step"),TEXT("enemy_0_stats_and_obs")},
	{TEXT("enemy:1"),TEXT("enemy_1_reset_step"),TEXT("enemy_1_stats_and_obs")},
	{TEXT("enemy:2"),TEXT("enemy_2_reset_step"),TEXT("enemy_2_stats_and_obs")},
	{TEXT("enemy:3"),TEXT("enemy_3_reset_step"),TEXT("enemy_3_stats_and_obs")},
	{TEXT("enemy:4"),TEXT("enemy_4_reset_step"),TEXT("enemy_4_stats_and_obs")},
	{TEXT("enemy:5"),TEXT("enemy_5_reset_step"),TEXT("enemy_5_stats_and_obs")},
	{TEXT("enemy:6"),TEXT("enemy_6_reset_step"),TEXT("enemy_6_stats_and_obs")},
	{TEXT("enemy:7"),TEXT("enemy_7_reset_step"),TEXT("enemy_7_stats_and_obs")},
	{TEXT("enemy:8"),TEXT("enemy_8_reset_step"),TEXT("enemy_8_stats_and_obs")},
	{TEXT("enemy:9"),TEXT("enemy_9_reset_step"),TEXT("enemy_9_stats_and_obs")},
	{TEXT("enemy:10"),TEXT("enemy_10_boss_reset_step"),TEXT("enemy_10_boss_stats_and_obs")},
};

static const FContentCoverageCell CombinationCoverageCells[] = {
	{TEXT("combination:pair_evolution_union"),TEXT("vandalier_consumes_partner_slot"),TEXT("one_vandalier_slot_and_one_empty_slot")},
	{TEXT("combination:weak_defensive_weapon"),TEXT("laurel_shield_under_contact"),TEXT("shield_absorbs_hit_and_observation_changes")},
	{TEXT("combination:instant_kill_resistance"),TEXT("pentagram_against_resistant_boss"),TEXT("boss_survives_instant_kill")},
	{TEXT("combination:boss_interaction"),TEXT("gorgeous_moon_boss_chest"),TEXT("boss_drop_and_evolution_are_observed")},
};

/**
 * content key を実行対象 registry の cell へ exact に解決する。
 * 初心者向け:
 * 各表駆動テストが自分の scenario と評価名を実際に参照し、未使用 registry にならないようにします。
 */
const FContentCoverageCell& RequireCoverageCell(const FString& ContentKey)
{
	for (const FContentCoverageCell& Cell : ContentCoverageCells)
		if (ContentKey == Cell.ContentKey) return Cell;
	FAIL("content coverage cell was not found");
	return ContentCoverageCells[0];
}

/**
 * combination kind を実行対象 registry の cell へ exact に解決する。
 * 初心者向け:
 * 4つの境界シナリオも通常 content と同様に、実際の CHECK から評価名を参照します。
 */
const FContentCoverageCell& RequireCombinationCell(const FString& Kind)
{
	const FString Key = FString::Printf(TEXT("combination:%s"), *Kind);
	for (const FContentCoverageCell& Cell : CombinationCoverageCells)
		if (Key == Cell.ContentKey) return Cell;
	FAIL("combination coverage cell was not found");
	return CombinationCoverageCells[0];
}

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
 * coverage registry の全 cell が LLT から参照されることを検証する。
 * 初心者向け:
 * 空のシナリオ名や評価名を登録しただけの行を成功扱いしません。
 */
TEST_CASE("Survivors coverage registries are executable", "[unit][survivors][content][coverage]")
{
	for (const FContentCoverageCell& Cell : ContentCoverageCells)
	{
		CHECK(!FString(Cell.ContentKey).IsEmpty());
		CHECK(!FString(Cell.ScenarioId).IsEmpty());
		CHECK(!FString(Cell.EvalAssertionId).IsEmpty());
	}
	for (const FContentCoverageCell& Cell : CombinationCoverageCells)
	{
		CHECK(!FString(Cell.ContentKey).IsEmpty());
		CHECK(!FString(Cell.ScenarioId).IsEmpty());
		CHECK(!FString(Cell.EvalAssertionId).IsEmpty());
	}
}

/**
 * union・防御・即死耐性・boss interaction の4境界を production 状態で検証する。
 * 初心者向け:
 * 組合せ名だけを登録せず、各 cell が武器枠、shield、boss 耐性、chest 進化の判定を実行します。
 */
TEST_CASE("Survivors combination coverage cells assert production mechanics", "[unit][survivors][content][combination]")
{
	SECTION("pair evolution union")
	{
		const FContentCoverageCell& Cell = RequireCombinationCell(TEXT("pair_evolution_union"));
		INFO(Cell.ScenarioId); INFO(Cell.EvalAssertionId);
		const SurvivorsGameConstants::FEvolutionRule& Rule =
			SurvivorsGameConstants::EvolutionTable.Last();
		CHECK(Rule.BaseWeapon == EWeaponType::Peachone);
		CHECK(Rule.UnionPartner == EWeaponType::EbonyWings);
		CHECK(Rule.EvolvedWeapon == EWeaponType::Vandalier);
	}
	SECTION("weak defensive weapon")
	{
		const FContentCoverageCell& Cell = RequireCombinationCell(TEXT("weak_defensive_weapon"));
		INFO(Cell.ScenarioId); INFO(Cell.EvalAssertionId);
		FSurvivorsGameLogicConfig Config;
		Config.bHasInitialOverride = true;
		Config.InitialWeaponSlots.Add({static_cast<int32>(EWeaponType::Laurel), 1});
		FSurvivorsGameLogic Logic;
		Logic.Initialize(Config);
		Logic.Reset(5100);
		CHECK(Logic.GetWeaponSlot(0).Type == EWeaponType::Laurel);
		CHECK(Logic.IsShieldActive());
	}
	SECTION("instant kill resistance")
	{
		const FContentCoverageCell& Cell = RequireCombinationCell(TEXT("instant_kill_resistance"));
		INFO(Cell.ScenarioId); INFO(Cell.EvalAssertionId);
		FSurvivorsGameLogic Logic;
		Logic.Reset(5200);
		const FEnemyTypeParams& Boss = FSurvivorsGameTestAccess::EnemyTable(Logic)[10];
		CHECK(Boss.bIsBoss);
		CHECK(Boss.bResistsInstantKill);
	}
	SECTION("boss interaction")
	{
		const FContentCoverageCell& Cell = RequireCombinationCell(TEXT("boss_interaction"));
		INFO(Cell.ScenarioId); INFO(Cell.EvalAssertionId);
		FSurvivorsGameLogicConfig Config;
		Config.bEnableEvolutions = true;
		FSurvivorsGameLogic Logic;
		Logic.Initialize(Config);
		Logic.Reset(5300);
		CHECK(FSurvivorsGameTestAccess::EnemyTable(Logic)[10].bIsBoss);
		CHECK(SurvivorsGameConstants::EvolutionTable.ContainsByPredicate(
			[](const SurvivorsGameConstants::FEvolutionRule& Rule)
			{
				return Rule.BaseWeapon == EWeaponType::Pentagram
					&& Rule.EvolvedWeapon == EWeaponType::GorgeousMoon;
			}));
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
		const FContentCoverageCell& Cell = RequireCoverageCell(FString::Printf(TEXT("weapon:%d"), WeaponId));
		INFO(Cell.ScenarioId);
		INFO(Cell.EvalAssertionId);
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
		const FContentCoverageCell& Cell = RequireCoverageCell(FString::Printf(TEXT("passive:%d"), PassiveId));
		INFO(Cell.ScenarioId);
		INFO(Cell.EvalAssertionId);
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
 * gem 3色の production drop と XP 値を表駆動で検証する。
 * 初心者向け:
 * 敵が落とす色と経験値が有限で、各色の coverage cell が実際の確認へ結び付くことを確かめます。
 */
TEST_CASE("Survivors all gem pickups expose finite XP", "[unit][survivors][content][gem]")
{
	const TCHAR* GemIds[] = {TEXT("blue"), TEXT("green"), TEXT("red")};
	for (int32 GemIndex = 0; GemIndex < 3; ++GemIndex)
	{
		const FContentCoverageCell& Cell = RequireCoverageCell(
			FString::Printf(TEXT("gem:%s"), GemIds[GemIndex]));
		INFO(Cell.ScenarioId);
		INFO(Cell.EvalAssertionId);
		FSurvivorsGameLogic Logic;
		Logic.Reset(2500 + GemIndex);
		FSurvivorsGameTestAccess::DropGem(Logic, GemIndex == 0 ? 0 : (GemIndex == 1 ? 4 : 10));
		REQUIRE(Logic.GetItemCount() == 1);
		CHECK(static_cast<int32>(Logic.GetItemGemType(0)) == GemIndex);
		CHECK(FMath::IsFinite(FSurvivorsGameTestAccess::GemXP(Logic, 0)));
		CHECK(FSurvivorsGameTestAccess::GemXP(Logic, 0) >= 0.f);
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
		const FContentCoverageCell& Cell = RequireCoverageCell(FString::Printf(TEXT("enemy:%d"), EnemyId));
		INFO(Cell.ScenarioId);
		INFO(Cell.EvalAssertionId);
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
		const FContentCoverageCell& Cell = RequireCoverageCell(FString::Printf(
			TEXT("evolution:%d:%d"), static_cast<int32>(Rule.BaseWeapon), static_cast<int32>(Rule.EvolvedWeapon)));
		INFO(Cell.ScenarioId);
		INFO(Cell.EvalAssertionId);
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
