#include "Survivors/Logic/SurvivorsPlayerComponent.h"

#include "Survivors/Logic/SurvivorsGame.h"
#include "Survivors/Logic/Weapons/SurvivorsWeaponComponent.h"
#include "Survivors/Logic/Weapons/SurvivorsWeaponBase.h"

USurvivorsPlayerComponent::USurvivorsPlayerComponent()
{
	PrimaryComponentTick.bCanEverTick = false;
}

void USurvivorsPlayerComponent::Initialize(ASurvivorsGame* InGame)
{
	Game = InGame;
}

void USurvivorsPlayerComponent::Reset()
{
	if (!Game) return;

	Game->PlayerPos   = FVector2D::ZeroVector;
	Game->PlayerVel   = FVector2D::ZeroVector;
	Game->PlayerHP    = Game->MaxPlayerHP;
	Game->PlayerXP    = 0.f;
	Game->PlayerLevel = 1;

	// 武器スロット・パッシブスロットのリセットは ASurvivorsGame::ResetState() で実施済み
	// ここでは開始武器の付与は行わない（ResetState で WeaponComponent::EquipWeapon を呼ぶ）
}

void USurvivorsPlayerComponent::ApplyAction(int32 ActionIdx)
{
	if (!Game) return;

	// アクション番号定義（対称性拡張・Python/EUREKA prompt と共通仕様）
	// 0=北(+Y), 1=北東, 2=東(+X), 3=南東, 4=南(-Y), 5=南西, 6=西(-X), 7=北西, 8=静止
	FVector2D MoveDir = FVector2D::ZeroVector;
	switch (ActionIdx)
	{
		case 0: MoveDir = FVector2D( 0.f,  1.f);                        break; // 北
		case 1: MoveDir = FVector2D( 1.f,  1.f).GetSafeNormal(); break; // 北東
		case 2: MoveDir = FVector2D( 1.f,  0.f);                        break; // 東
		case 3: MoveDir = FVector2D( 1.f, -1.f).GetSafeNormal(); break; // 南東
		case 4: MoveDir = FVector2D( 0.f, -1.f);                        break; // 南
		case 5: MoveDir = FVector2D(-1.f, -1.f).GetSafeNormal(); break; // 南西
		case 6: MoveDir = FVector2D(-1.f,  0.f);                        break; // 西
		case 7: MoveDir = FVector2D(-1.f,  1.f).GetSafeNormal(); break; // 北西
		default: break; // 8=静止
	}
	// Wings パッシブによる移動速度補正（MoveSpeedMult は最終倍率: デフォルト 1.0 = 変化なし）
	const float EffectiveMoveSpeed = Game->MoveSpeed * Game->CachedPassiveEffects.MoveSpeedMult;
	Game->PlayerVel = MoveDir * EffectiveMoveSpeed;
	Game->PlayerPos += Game->PlayerVel * SurvivorsGameConstants::PhysicsDt;
}

float USurvivorsPlayerComponent::XPRequiredForLevel(int32 Level) const
{
	if (Level <= 1) return 0.f;
	if (Level == 2) return 5.f;
	float Value = 5.f;
	for (int32 Lv = 3; Lv <= Level; ++Lv)
	{
		if      (Lv <= 20) Value += 10.f;
		else if (Lv <= 40) Value += 13.f;
		else if (Lv <= 60) Value += 16.f;
		else if (Lv <= 80) Value += 19.f;
		else               Value += 22.f;
	}
	return Value;
}

float USurvivorsPlayerComponent::CumulativeXPForLevel(int32 Level) const
{
	float Total = 0.f;
	for (int32 Lv = 2; Lv <= Level; ++Lv)
	{
		Total += XPRequiredForLevel(Lv);
	}
	return Total;
}

void USurvivorsPlayerComponent::ProcessXPGain(float Amount)
{
	if (!Game) return;

	Game->PlayerXP += Amount;
	while (true)
	{
		const float NextThreshold = CumulativeXPForLevel(Game->PlayerLevel + 1);
		if (Game->PlayerXP < NextThreshold) break;
		Game->PlayerLevel++;
		OnLevelUp(Game->PlayerLevel);
	}
}

void USurvivorsPlayerComponent::OnLevelUp(int32 NextLevel)
{
	if (!Game) return;

	// WeaponPoolMode が "garlic_only" の場合は W0 フェーズ互換（Garlic-only）
	// W1〜W3 は enable_passives=false / enable_evolutions=false のまま複数武器を扱うため、
	// bEnablePassives / bEnableEvolutions の組み合わせではなく WeaponPoolMode で判断する。
	if (Game->WeaponPoolMode.Equals(TEXT("garlic_only"), ESearchCase::IgnoreCase))
	{
		// 旧動作: スロット0の武器を単純レベルアップ
		if (Game->WeaponSlots[0].Type != EWeaponType::None)
		{
			const int32 NewLv = FMath::Min(
				Game->WeaponSlots[0].Level.Value + 1,
				SurvivorsGameConstants::MaxWeaponLevel);
			Game->WeaponSlots[0].Level = FWeaponLevel(NewLv);

			if (Game->WeaponComponent)
			{
				USurvivorsWeaponBase* WI = Game->WeaponComponent->GetWeaponInstance(0);
				if (WI) WI->SetLevel(FWeaponLevel(NewLv));
			}
		}
		return;
	}

	// PR2: BuildLevelUpChoices → ランダム選択 → ApplyChoice → RecalcPassiveEffects
	TArray<FLevelUpChoice> Choices = BuildLevelUpChoices();
	if (Choices.Num() == 0) return;

	// ランダムに1択を選ぶ
	const int32 ChoiceIdx = Game->RandStream.RandRange(0, Choices.Num() - 1);
	const FLevelUpChoice& Choice = Choices[ChoiceIdx];

	ApplyLevelUpChoice(Choice);
	RecalcPassiveEffects();
}

void USurvivorsPlayerComponent::ApplyLevelUpChoice(const FLevelUpChoice& Choice)
{
	if (!Game) return;

	switch (Choice.ChoiceType)
	{
	case FLevelUpChoice::EChoiceType::PassiveNew:
		{
			// 空きパッシブスロットに追加
			for (int32 i = 0; i < SurvivorsGameConstants::MaxPassiveSlots; ++i)
			{
				if (Game->PassiveSlots[i].Type == EPassiveItemType::None)
				{
					Game->PassiveSlots[i].Type  = Choice.PassiveType;
					Game->PassiveSlots[i].Level = 1;
					break;
				}
			}
		}
		return;

	case FLevelUpChoice::EChoiceType::PassiveUpgrade:
		{
			// 既存パッシブのレベルアップ
			for (int32 i = 0; i < SurvivorsGameConstants::MaxPassiveSlots; ++i)
			{
				if (Game->PassiveSlots[i].Type == Choice.PassiveType)
				{
					const int32 MaxLv = SurvivorsGameConstants::PassiveMaxLevel[static_cast<int32>(Choice.PassiveType)];
					Game->PassiveSlots[i].Level = FMath::Min(Game->PassiveSlots[i].Level + 1, MaxLv);
					break;
				}
			}
		}
		return;

	case FLevelUpChoice::EChoiceType::WeaponEvolve:
		{
			// 進化: ベース武器スロットを探して進化させる
			for (const SurvivorsGameConstants::FEvolutionRule& Rule : SurvivorsGameConstants::EvolutionTable)
			{
				if (Rule.EvolvedWeapon == Choice.WeaponType)
				{
					for (int32 i = 0; i < SurvivorsGameConstants::MaxWeaponSlots; ++i)
					{
						if (Game->WeaponSlots[i].Type == Rule.BaseWeapon)
						{
							EvolveWeapon(i, Choice.WeaponType);
							return;
						}
					}
					break;
				}
			}
		}
		return;

	case FLevelUpChoice::EChoiceType::WeaponUpgrade:
		{
			if (!Game->WeaponComponent) return;
			for (int32 i = 0; i < SurvivorsGameConstants::MaxWeaponSlots; ++i)
			{
				if (Game->WeaponSlots[i].Type == Choice.WeaponType)
				{
					const int32 NewLv = FMath::Min(Choice.NewLevel, SurvivorsGameConstants::MaxWeaponLevel);
					Game->WeaponSlots[i].Level = FWeaponLevel(NewLv);
					USurvivorsWeaponBase* WI = Game->WeaponComponent->GetWeaponInstance(i);
					if (WI) WI->SetLevel(FWeaponLevel(NewLv));
					return;
				}
			}
		}
		return;

	case FLevelUpChoice::EChoiceType::WeaponNew:
	default:
		{
			if (!Game->WeaponComponent) return;
			// 空きスロットに新規装備
			for (int32 i = 0; i < SurvivorsGameConstants::MaxWeaponSlots; ++i)
			{
				if (Game->WeaponSlots[i].Type == EWeaponType::None)
				{
					Game->WeaponSlots[i].Type  = Choice.WeaponType;
					Game->WeaponSlots[i].Level = FWeaponLevel(1);
					Game->WeaponComponent->EquipWeapon(i, Choice.WeaponType, 1);
					return;
				}
			}
		}
		return;
	}
}

// ---- パッシブ効果計算 --------------------------------------------------------

FPassiveEffects USurvivorsPlayerComponent::ComputePassiveEffects() const
{
	FPassiveEffects PE;
	if (!Game) return PE;

	for (const FPassiveSlot& Slot : Game->PassiveSlots)
	{
		if (Slot.Type == EPassiveItemType::None || Slot.Level <= 0) continue;
		const int32 Lv = Slot.Level;

		switch (Slot.Type)
		{
		case EPassiveItemType::Spinach:
			PE.DamageMult += 0.10f * Lv;
			break;
		case EPassiveItemType::Armor:
			PE.ArmorFlat += 1.f * Lv;
			break;
		case EPassiveItemType::HollowHeart:
			PE.HpMult += 0.20f * Lv;
			break;
		case EPassiveItemType::Pummarola:
			PE.RegenPerSec += 0.5f * Lv;
			break;
		case EPassiveItemType::EmptyTome:
			PE.CooldownMult -= 0.08f * Lv;
			PE.CooldownMult  = FMath::Max(PE.CooldownMult, 0.4f);
			break;
		case EPassiveItemType::Candelabrador:
			PE.AreaMult += 0.10f * Lv;
			break;
		case EPassiveItemType::Bracer:
			PE.SpeedMult += 0.10f * Lv;
			break;
		case EPassiveItemType::Spellbinder:
			PE.DurationMult += 0.15f * Lv;
			break;
		case EPassiveItemType::Duplicator:
			PE.ExtraAmount += 1.f * Lv;
			break;
		case EPassiveItemType::Wings:
			PE.MoveSpeedMult += 0.05f * Lv;
			break;
		case EPassiveItemType::Attractorb:
			PE.PickupRadiusMult += 0.15f * Lv;
			break;
		case EPassiveItemType::Tirajisu:
			// MaxLevel=2
			PE.MaxRevivalCount += FMath::Min(Lv, 2);
			break;
		case EPassiveItemType::TorronasBox:
			PE.AreaMult     += 0.01f * Lv;
			PE.CooldownMult -= 0.01f * Lv;
			PE.DurationMult += 0.01f * Lv;
			break;
		case EPassiveItemType::Crown:
		case EPassiveItemType::StoneMask:
		case EPassiveItemType::SkullOManiac:
		case EPassiveItemType::Clover:
			// スタブ（将来実装用）
			break;
		default:
			break;
		}
	}

	return PE;
}

void USurvivorsPlayerComponent::RecalcPassiveEffects()
{
	if (!Game) return;

	Game->CachedPassiveEffects = ComputePassiveEffects();

	// MaxPlayerHP 更新（BaseMaxPlayerHPConst を基準にすることで累積増幅を防ぐ）
	const float NewMaxHP = ASurvivorsGame::BaseMaxPlayerHPConst * Game->CachedPassiveEffects.HpMult;
	// 現在 HP を比例スケール
	if (Game->MaxPlayerHP > 0.f)
	{
		const float Ratio  = Game->PlayerHP / Game->MaxPlayerHP;
		Game->PlayerHP     = NewMaxHP * Ratio;
	}
	Game->MaxPlayerHP = NewMaxHP;

	// MaxRevivalCount 更新
	Game->MaxRevivalCount = Game->CachedPassiveEffects.MaxRevivalCount;

	// GemPickupRadius 更新（BaseGemPickupRadiusConst を基準にすることで累積増幅を防ぐ）
	Game->GemPickupRadius = ASurvivorsGame::BaseGemPickupRadiusConst * Game->CachedPassiveEffects.PickupRadiusMult;
}

// ---- 進化システム ------------------------------------------------------------

TArray<int32> USurvivorsPlayerComponent::GetEvolvableWeapons() const
{
	TArray<int32> Result;
	if (!Game) return Result;

	for (const SurvivorsGameConstants::FEvolutionRule& Rule : SurvivorsGameConstants::EvolutionTable)
	{
		// ベース武器を持っているスロットを探す
		for (int32 SlotIdx = 0; SlotIdx < SurvivorsGameConstants::MaxWeaponSlots; ++SlotIdx)
		{
			const FWeaponSlot& WSlot = Game->WeaponSlots[SlotIdx];
			if (WSlot.Type != Rule.BaseWeapon) continue;
			if (!WSlot.Level.IsMax()) continue;  // Lv8 必須

			// Union 判定（Vandalier の場合）
			if (Rule.UnionPartner != EWeaponType::None)
			{
				bool bHasPartner = false;
				for (int32 i = 0; i < SurvivorsGameConstants::MaxWeaponSlots; ++i)
				{
					if (Game->WeaponSlots[i].Type == Rule.UnionPartner &&
						Game->WeaponSlots[i].Level.IsMax())
					{
						bHasPartner = true;
						break;
					}
				}
				if (!bHasPartner) continue;
			}

			// 必要パッシブチェック
			if (Rule.RequiredPassive != EPassiveItemType::None)
			{
				bool bHasPassive = false;
				for (const FPassiveSlot& PSlot : Game->PassiveSlots)
				{
					if (PSlot.Type == Rule.RequiredPassive && PSlot.Level > 0)
					{
						bHasPassive = true;
						break;
					}
				}
				if (!bHasPassive) continue;
			}

			Result.Add(SlotIdx);
		}
	}
	return Result;
}

void USurvivorsPlayerComponent::EvolveWeapon(int32 SlotIdx, EWeaponType EvolvedType)
{
	if (!Game) return;
	if (!Game->WeaponSlots[SlotIdx].Level.IsMax()) return;  // Lv8 でなければ進化不可

	// スロットを進化後武器に置き換え・レベルを 1 にリセット
	Game->WeaponSlots[SlotIdx].Type  = EvolvedType;
	Game->WeaponSlots[SlotIdx].Level = FWeaponLevel(1);

	if (Game->WeaponComponent)
	{
		Game->WeaponComponent->EquipWeapon(SlotIdx, EvolvedType, 1);
	}

	// Vandalier Union 進化: EbonyWings スロットを解放
	if (EvolvedType == EWeaponType::Vandalier)
	{
		for (int32 k = 0; k < SurvivorsGameConstants::MaxWeaponSlots; ++k)
		{
			if (Game->WeaponSlots[k].Type == EWeaponType::EbonyWings)
			{
				// FWeaponLevel / FCooldownSeconds は explicit ctor を持つため個別代入
				Game->WeaponSlots[k].Type     = EWeaponType::None;
				Game->WeaponSlots[k].Level    = FWeaponLevel(0);
				Game->WeaponSlots[k].Cooldown = FCooldownSeconds(0.f);
				if (Game->WeaponComponent)
				{
					Game->WeaponComponent->UnequipWeapon(k);
				}
				break;
			}
		}
	}
}

TArray<FLevelUpChoice> USurvivorsPlayerComponent::BuildLevelUpChoices()
{
	if (!Game) return {};

	// 全パッシブタイプを候補として列挙（EPassiveItemType::Spinach=1 〜 TorronasBox=17 の全17値）
	// Clover(12) は Cross→HeavenSword、Crown(13) は Pentagram→GorgeousMoon の進化に必要
	static const EPassiveItemType AllPassiveTypes[] = {
		EPassiveItemType::Spinach,  EPassiveItemType::Armor,        EPassiveItemType::HollowHeart,
		EPassiveItemType::Pummarola, EPassiveItemType::EmptyTome,   EPassiveItemType::Candelabrador,
		EPassiveItemType::Bracer,   EPassiveItemType::Spellbinder,  EPassiveItemType::Duplicator,
		EPassiveItemType::Wings,    EPassiveItemType::Attractorb,   EPassiveItemType::Clover,
		EPassiveItemType::Crown,    EPassiveItemType::StoneMask,    EPassiveItemType::SkullOManiac,
		EPassiveItemType::Tirajisu, EPassiveItemType::TorronasBox,
	};

	// --- 候補プールを事前構築 ---

	// 進化候補
	TArray<FLevelUpChoice> Evolutions;
	if (Game->bEnableEvolutions)
	{
		for (int32 EvolveSlot : GetEvolvableWeapons())
		{
			for (const SurvivorsGameConstants::FEvolutionRule& Rule : SurvivorsGameConstants::EvolutionTable)
			{
				if (Rule.BaseWeapon == Game->WeaponSlots[EvolveSlot].Type)
				{
					FLevelUpChoice C;
					C.ChoiceType = FLevelUpChoice::EChoiceType::WeaponEvolve;
					C.WeaponType = Rule.EvolvedWeapon;
					C.SlotIdx    = EvolveSlot;
					C.NewLevel   = 1;
					Evolutions.Add(C);
					break;
				}
			}
		}
	}

	// 既存武器アップグレード候補
	TArray<FLevelUpChoice> WeaponUpgrades;
	for (int32 i = 0; i < SurvivorsGameConstants::MaxWeaponSlots; ++i)
	{
		if (Game->WeaponSlots[i].Type == EWeaponType::None) continue;
		if (Game->WeaponSlots[i].Level.IsMax()) continue;

		FLevelUpChoice C;
		C.ChoiceType = FLevelUpChoice::EChoiceType::WeaponUpgrade;
		C.WeaponType = Game->WeaponSlots[i].Type;
		C.SlotIdx    = i;
		C.NewLevel   = Game->WeaponSlots[i].Level.Value + 1;
		WeaponUpgrades.Add(C);
	}

	// 新規武器候補（weapon_pool_mode に基づいてフィルタリング）
	// weapon_pool_mode の受け付け値:
	//   "garlic_only"         → Garlic のみ
	//   "fixed_subset"        → AllowedWeaponTypes を使用
	//   "all_base"            → 全基本武器（Garlic〜Laurel）
	//   "all_with_evolutions" → all_base と同じ（進化後武器は進化システムで処理）
	//   "weighted"            → fixed_subset 扱い（weights=0 の武器は Python 側で除外済み）
	//   未知値                → SurvivorsHttpEnvService で "garlic_only" にフォールバック済み
	TArray<FLevelUpChoice> NewWeapons;
	{
		int32 EmptySlots = 0;
		for (int32 i = 0; i < SurvivorsGameConstants::MaxWeaponSlots; ++i)
			if (Game->WeaponSlots[i].Type == EWeaponType::None) ++EmptySlots;

		if (EmptySlots > 0)
		{
			TArray<EWeaponType> AllowedPool;
			if (Game->WeaponPoolMode == TEXT("garlic_only"))
			{
				AllowedPool = { EWeaponType::Garlic };
			}
			else if ((Game->WeaponPoolMode == TEXT("fixed_subset") || Game->WeaponPoolMode == TEXT("weighted"))
				&& Game->AllowedWeaponTypes.Num() > 0)
			{
				// "weighted" は Python 側が weights=0 の武器を除外して allowed_weapon_types を送信する
				for (int32 Id : Game->AllowedWeaponTypes)
					AllowedPool.Add(static_cast<EWeaponType>(Id));
			}
			else  // "all_base" / "all_with_evolutions" / デフォルト（garlic_only フォールバック後）
			{
				AllowedPool = {
					EWeaponType::Garlic,  EWeaponType::Whip,   EWeaponType::MagicWand,
					EWeaponType::Knife,   EWeaponType::Axe,    EWeaponType::Cross,
					EWeaponType::KingBible, EWeaponType::FireWand, EWeaponType::SantaWater,
					EWeaponType::Runetracer, EWeaponType::LightningRing, EWeaponType::Pentagram,
					EWeaponType::Peachone, EWeaponType::EbonyWings, EWeaponType::Laurel,
				};
			}

			for (EWeaponType WT : AllowedPool)
			{
				bool bOwned = false;
				for (int32 i = 0; i < SurvivorsGameConstants::MaxWeaponSlots; ++i)
					if (Game->WeaponSlots[i].Type == WT) { bOwned = true; break; }
				if (bOwned) continue;

				FLevelUpChoice C;
				C.ChoiceType = FLevelUpChoice::EChoiceType::WeaponNew;
				C.WeaponType = WT;
				C.NewLevel   = 1;
				NewWeapons.Add(C);
			}
		}
	}

	// パッシブアップグレード候補（既存パッシブで MaxLevel 未満のもの）
	TArray<FLevelUpChoice> PassiveUpgrades;
	// パッシブ新規候補（空きスロットあり・未所持）
	TArray<FLevelUpChoice> PassiveNew;
	if (Game->bEnablePassives)
	{
		for (int32 i = 0; i < SurvivorsGameConstants::MaxPassiveSlots; ++i)
		{
			const FPassiveSlot& PS = Game->PassiveSlots[i];
			if (PS.Type == EPassiveItemType::None || PS.Level <= 0) continue;
			const int32 MaxLv = SurvivorsGameConstants::PassiveMaxLevel[static_cast<int32>(PS.Type)];
			if (PS.Level >= MaxLv) continue;

			FLevelUpChoice C;
			C.ChoiceType  = FLevelUpChoice::EChoiceType::PassiveUpgrade;
			C.PassiveType = PS.Type;
			C.SlotIdx     = i;
			C.NewLevel    = PS.Level + 1;
			PassiveUpgrades.Add(C);
		}

		int32 EmptyPassiveSlots = 0;
		for (int32 i = 0; i < SurvivorsGameConstants::MaxPassiveSlots; ++i)
			if (Game->PassiveSlots[i].Type == EPassiveItemType::None) ++EmptyPassiveSlots;

		if (EmptyPassiveSlots > 0)
		{
			for (EPassiveItemType PT : AllPassiveTypes)
			{
				bool bOwned = false;
				for (int32 i = 0; i < SurvivorsGameConstants::MaxPassiveSlots; ++i)
					if (Game->PassiveSlots[i].Type == PT) { bOwned = true; break; }
				if (bOwned) continue;

				FLevelUpChoice C;
				C.ChoiceType  = FLevelUpChoice::EChoiceType::PassiveNew;
				C.PassiveType = PT;
				C.NewLevel    = 1;
				PassiveNew.Add(C);
			}
		}
	}

	// --- 優先順位に従って Choices を構築 ---
	TArray<FLevelUpChoice> Choices;

	// 1. 進化を優先（最大1件）
	for (const FLevelUpChoice& C : Evolutions)
	{
		if (Choices.Num() >= 3) break;
		Choices.Add(C);
	}

	// 2. パッシブを最低1件確保（bEnablePassives かつ候補あり）
	if (Game->bEnablePassives && Choices.Num() < 3)
	{
		TArray<FLevelUpChoice> AllPassiveCandidates;
		AllPassiveCandidates.Append(PassiveUpgrades);
		AllPassiveCandidates.Append(PassiveNew);
		if (AllPassiveCandidates.Num() > 0)
		{
			const int32 Idx = Game->RandStream.RandRange(0, AllPassiveCandidates.Num() - 1);
			Choices.Add(AllPassiveCandidates[Idx]);
		}
	}

	// 3. 残りを武器（アップグレード→新規）で補充
	TArray<FLevelUpChoice> WeaponPool;
	WeaponPool.Append(WeaponUpgrades);
	WeaponPool.Append(NewWeapons);
	// 進化・パッシブで既に追加した武器タイプとの重複除去
	for (int32 j = WeaponPool.Num() - 1; j >= 0; --j)
	{
		for (const FLevelUpChoice& Existing : Choices)
		{
			if (Existing.WeaponType == WeaponPool[j].WeaponType && Existing.WeaponType != EWeaponType::None)
			{
				WeaponPool.RemoveAt(j);
				break;
			}
		}
	}
	while (Choices.Num() < 3 && WeaponPool.Num() > 0)
	{
		const int32 Idx = Game->RandStream.RandRange(0, WeaponPool.Num() - 1);
		Choices.Add(WeaponPool[Idx]);
		WeaponPool.RemoveAt(Idx);
	}

	// 4. それでも不足ならパッシブ候補で補充（ステップ2で使ったものとの重複を避ける）
	if (Game->bEnablePassives && Choices.Num() < 3)
	{
		TArray<FLevelUpChoice> AllPassiveCandidates;
		AllPassiveCandidates.Append(PassiveUpgrades);
		AllPassiveCandidates.Append(PassiveNew);
		// ステップ2で追加済みのパッシブを除外
		for (int32 j = AllPassiveCandidates.Num() - 1; j >= 0; --j)
		{
			for (const FLevelUpChoice& Existing : Choices)
			{
				if (Existing.ChoiceType != FLevelUpChoice::EChoiceType::WeaponNew &&
					Existing.ChoiceType != FLevelUpChoice::EChoiceType::WeaponUpgrade &&
					Existing.ChoiceType != FLevelUpChoice::EChoiceType::WeaponEvolve &&
					Existing.PassiveType == AllPassiveCandidates[j].PassiveType)
				{
					AllPassiveCandidates.RemoveAt(j);
					break;
				}
			}
		}
		while (Choices.Num() < 3 && AllPassiveCandidates.Num() > 0)
		{
			const int32 Idx = Game->RandStream.RandRange(0, AllPassiveCandidates.Num() - 1);
			Choices.Add(AllPassiveCandidates[Idx]);
			AllPassiveCandidates.RemoveAt(Idx);
		}
	}

	return Choices;
}
