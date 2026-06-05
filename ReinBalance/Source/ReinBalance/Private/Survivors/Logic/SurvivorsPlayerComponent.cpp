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
	Game->PlayerVel = MoveDir * Game->MoveSpeed;
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

	// bEnableEvolutions / bEnablePassives が無効な場合は W0 フェーズ互換（Garlic-only）
	if (!Game->bEnablePassives && !Game->bEnableEvolutions)
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
}

TArray<FLevelUpChoice> USurvivorsPlayerComponent::BuildLevelUpChoices()
{
	TArray<FLevelUpChoice> Choices;
	if (!Game) return Choices;

	// 1. 進化候補を優先追加（bEnableEvolutions が有効な場合のみ）
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
					Choices.Add(C);
					break;
				}
			}
			if (Choices.Num() >= 3) break;
		}
	}

	// 2. 既存武器のレベルアップ候補
	if (Choices.Num() < 3)
	{
		for (int32 i = 0; i < SurvivorsGameConstants::MaxWeaponSlots && Choices.Num() < 3; ++i)
		{
			if (Game->WeaponSlots[i].Type == EWeaponType::None) continue;
			if (Game->WeaponSlots[i].Level.IsMax()) continue;

			const EWeaponType WType = Game->WeaponSlots[i].Type;
			bool bAlready = false;
			for (const FLevelUpChoice& C : Choices)
				if (C.WeaponType == WType) { bAlready = true; break; }
			if (!bAlready)
			{
				FLevelUpChoice C;
				C.ChoiceType = FLevelUpChoice::EChoiceType::WeaponUpgrade;
				C.WeaponType = WType;
				C.SlotIdx    = i;
				C.NewLevel   = Game->WeaponSlots[i].Level.Value + 1;
				Choices.Add(C);
			}
		}
	}

	// 3. 新規武器候補（weapon_pool_mode に基づいてフィルタリング、空きスロットがある場合）
	if (Choices.Num() < 3)
	{
		int32 EmptySlots = 0;
		for (int32 i = 0; i < SurvivorsGameConstants::MaxWeaponSlots; ++i)
			if (Game->WeaponSlots[i].Type == EWeaponType::None) ++EmptySlots;

		if (EmptySlots > 0)
		{
			// 許可武器リストを weapon_pool_mode に基づいて構築
			TArray<EWeaponType> AllowedPool;
			if (Game->WeaponPoolMode == TEXT("garlic_only"))
			{
				AllowedPool = { EWeaponType::Garlic };
			}
			else if (Game->WeaponPoolMode == TEXT("fixed_subset") && Game->AllowedWeaponTypes.Num() > 0)
			{
				for (int32 Id : Game->AllowedWeaponTypes)
					AllowedPool.Add(static_cast<EWeaponType>(Id));
			}
			else  // "all_base" / "all_with_evolutions" / デフォルト
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
				if (Choices.Num() >= 3) break;
				bool bOwned = false;
				for (int32 i = 0; i < SurvivorsGameConstants::MaxWeaponSlots; ++i)
					if (Game->WeaponSlots[i].Type == WT) { bOwned = true; break; }
				if (bOwned) continue;

				bool bAlready = false;
				for (const FLevelUpChoice& C : Choices)
					if (C.WeaponType == WT) { bAlready = true; break; }
				if (!bAlready)
				{
					FLevelUpChoice C;
					C.ChoiceType = FLevelUpChoice::EChoiceType::WeaponNew;
					C.WeaponType = WT;
					C.NewLevel   = 1;
					Choices.Add(C);
				}
			}
		}
	}

	// 4. パッシブ選択肢で補充（bEnablePassives が有効な場合）
	if (Game->bEnablePassives && Choices.Num() < 3)
	{
		// 全パッシブタイプを候補として列挙
		static const EPassiveItemType AllPassives[] = {
			EPassiveItemType::Spinach,  EPassiveItemType::Armor,        EPassiveItemType::HollowHeart,
			EPassiveItemType::Pummarola, EPassiveItemType::EmptyTome,   EPassiveItemType::Candelabrador,
			EPassiveItemType::Bracer,   EPassiveItemType::Spellbinder,  EPassiveItemType::Duplicator,
			EPassiveItemType::Wings,    EPassiveItemType::Attractorb,   EPassiveItemType::Tirajisu,
			EPassiveItemType::TorronasBox,
		};

		// まず MaxLevel 未満の既存パッシブをレベルアップ候補に追加
		for (int32 i = 0; i < SurvivorsGameConstants::MaxPassiveSlots && Choices.Num() < 3; ++i)
		{
			const FPassiveSlot& PS = Game->PassiveSlots[i];
			if (PS.Type == EPassiveItemType::None || PS.Level <= 0) continue;
			const int32 MaxLv = SurvivorsGameConstants::PassiveMaxLevel[static_cast<int32>(PS.Type)];
			if (PS.Level >= MaxLv) continue;

			// 既に Choices に含まれていないか確認
			bool bAlready = false;
			for (const FLevelUpChoice& C : Choices)
				if (C.ChoiceType == FLevelUpChoice::EChoiceType::PassiveUpgrade && C.PassiveType == PS.Type)
				{ bAlready = true; break; }
			if (!bAlready)
			{
				FLevelUpChoice C;
				C.ChoiceType  = FLevelUpChoice::EChoiceType::PassiveUpgrade;
				C.PassiveType = PS.Type;
				C.SlotIdx     = i;
				C.NewLevel    = PS.Level + 1;
				Choices.Add(C);
			}
		}

		// 空きパッシブスロットがあれば未所持パッシブを追加
		int32 EmptyPassiveSlots = 0;
		for (int32 i = 0; i < SurvivorsGameConstants::MaxPassiveSlots; ++i)
			if (Game->PassiveSlots[i].Type == EPassiveItemType::None) ++EmptyPassiveSlots;

		if (EmptyPassiveSlots > 0)
		{
			for (EPassiveItemType PT : AllPassives)
			{
				if (Choices.Num() >= 3) break;
				bool bOwned = false;
				for (int32 i = 0; i < SurvivorsGameConstants::MaxPassiveSlots; ++i)
					if (Game->PassiveSlots[i].Type == PT) { bOwned = true; break; }
				if (bOwned) continue;

				bool bAlready = false;
				for (const FLevelUpChoice& C : Choices)
					if (C.ChoiceType == FLevelUpChoice::EChoiceType::PassiveNew && C.PassiveType == PT)
					{ bAlready = true; break; }
				if (!bAlready)
				{
					FLevelUpChoice C;
					C.ChoiceType  = FLevelUpChoice::EChoiceType::PassiveNew;
					C.PassiveType = PT;
					C.NewLevel    = 1;
					Choices.Add(C);
				}
			}
		}
	}

	return Choices;
}
