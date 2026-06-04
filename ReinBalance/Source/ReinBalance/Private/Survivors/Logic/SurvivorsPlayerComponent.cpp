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
	TArray<TPair<EWeaponType, int32>> Choices = BuildLevelUpChoices();
	if (Choices.Num() == 0) return;

	// ランダムに1択を選ぶ
	const int32 ChoiceIdx = Game->RandStream.RandRange(0, Choices.Num() - 1);
	const TPair<EWeaponType, int32>& Choice = Choices[ChoiceIdx];

	ApplyLevelUpChoice(Choice.Key, Choice.Value);
	RecalcPassiveEffects();
}

void USurvivorsPlayerComponent::ApplyLevelUpChoice(EWeaponType WeaponType, int32 NewLevel)
{
	if (!Game || !Game->WeaponComponent) return;

	// 進化後武器か確認（EvolutionTable に含まれているか）
	bool bIsEvolution = false;
	int32 BaseSlotIdx = INDEX_NONE;
	for (const SurvivorsGameConstants::FEvolutionRule& Rule : SurvivorsGameConstants::EvolutionTable)
	{
		if (Rule.EvolvedWeapon == WeaponType)
		{
			bIsEvolution = true;
			// ベース武器スロットを探す
			for (int32 i = 0; i < SurvivorsGameConstants::MaxWeaponSlots; ++i)
			{
				if (Game->WeaponSlots[i].Type == Rule.BaseWeapon)
				{
					BaseSlotIdx = i;
					break;
				}
			}
			break;
		}
	}

	if (bIsEvolution && BaseSlotIdx != INDEX_NONE)
	{
		// 進化: ベーススロットを進化後武器に置き換え
		EvolveWeapon(BaseSlotIdx, WeaponType);
		return;
	}

	// 既存武器のレベルアップか確認
	for (int32 i = 0; i < SurvivorsGameConstants::MaxWeaponSlots; ++i)
	{
		if (Game->WeaponSlots[i].Type == WeaponType)
		{
			const int32 NewLv = FMath::Min(NewLevel, SurvivorsGameConstants::MaxWeaponLevel);
			Game->WeaponSlots[i].Level = FWeaponLevel(NewLv);
			USurvivorsWeaponBase* WI = Game->WeaponComponent->GetWeaponInstance(i);
			if (WI) WI->SetLevel(FWeaponLevel(NewLv));
			return;
		}
	}

	// 新規武器取得: 空きスロットに装備
	for (int32 i = 0; i < SurvivorsGameConstants::MaxWeaponSlots; ++i)
	{
		if (Game->WeaponSlots[i].Type == EWeaponType::None)
		{
			Game->WeaponSlots[i].Type  = WeaponType;
			Game->WeaponSlots[i].Level = FWeaponLevel(1);
			Game->WeaponComponent->EquipWeapon(i, WeaponType, 1);
			return;
		}
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

	// MaxPlayerHP 更新
	const float BaseHP   = Game->MaxPlayerHP;  // UPROPERTY 設定値
	const float NewMaxHP = BaseHP * Game->CachedPassiveEffects.HpMult;
	// 現在 HP を比例スケール
	if (Game->MaxPlayerHP > 0.f)
	{
		const float Ratio  = Game->PlayerHP / Game->MaxPlayerHP;
		Game->PlayerHP     = NewMaxHP * Ratio;
	}
	Game->MaxPlayerHP = NewMaxHP;

	// MaxRevivalCount 更新
	Game->MaxRevivalCount = Game->CachedPassiveEffects.MaxRevivalCount;

	// GemPickupRadius 更新
	const float BasePickupRadius = 30.f;  // 基本値
	Game->GemPickupRadius = BasePickupRadius * Game->CachedPassiveEffects.PickupRadiusMult;
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

TArray<TPair<EWeaponType, int32>> USurvivorsPlayerComponent::BuildLevelUpChoices()
{
	TArray<TPair<EWeaponType, int32>> Choices;
	if (!Game) return Choices;

	// 1. 進化候補を優先追加
	for (int32 EvolveSlot : GetEvolvableWeapons())
	{
		// 進化後武器を探す
		for (const SurvivorsGameConstants::FEvolutionRule& Rule : SurvivorsGameConstants::EvolutionTable)
		{
			if (Rule.BaseWeapon == Game->WeaponSlots[EvolveSlot].Type)
			{
				Choices.Add(TPair<EWeaponType, int32>(Rule.EvolvedWeapon, 1));
				break;
			}
		}
		if (Choices.Num() >= 3) break;
	}

	// 2. 既存武器のレベルアップ候補
	if (Choices.Num() < 3)
	{
		for (int32 i = 0; i < SurvivorsGameConstants::MaxWeaponSlots && Choices.Num() < 3; ++i)
		{
			if (Game->WeaponSlots[i].Type == EWeaponType::None) continue;
			if (Game->WeaponSlots[i].Level.IsMax()) continue;

			const EWeaponType WType = Game->WeaponSlots[i].Type;
			// 既に Choices に含まれていなければ追加
			bool bAlready = false;
			for (const auto& C : Choices)
				if (C.Key == WType) { bAlready = true; break; }
			if (!bAlready)
				Choices.Add(TPair<EWeaponType, int32>(WType, Game->WeaponSlots[i].Level.Value + 1));
		}
	}

	// 3. 新規武器候補（空きスロットがある場合）
	if (Choices.Num() < 3)
	{
		// 空きスロット数を確認
		int32 EmptySlots = 0;
		for (int32 i = 0; i < SurvivorsGameConstants::MaxWeaponSlots; ++i)
		{
			if (Game->WeaponSlots[i].Type == EWeaponType::None) ++EmptySlots;
		}

		if (EmptySlots > 0)
		{
			// 全武器タイプから未所持のものを追加
			static const EWeaponType AllWeapons[] = {
				EWeaponType::Whip, EWeaponType::MagicWand, EWeaponType::Knife,
				EWeaponType::Axe, EWeaponType::Cross, EWeaponType::KingBible,
				EWeaponType::FireWand, EWeaponType::SantaWater, EWeaponType::Runetracer,
				EWeaponType::LightningRing, EWeaponType::Pentagram,
				EWeaponType::Peachone, EWeaponType::EbonyWings, EWeaponType::Laurel,
			};

			for (EWeaponType WT : AllWeapons)
			{
				if (Choices.Num() >= 3) break;
				bool bOwned = false;
				for (int32 i = 0; i < SurvivorsGameConstants::MaxWeaponSlots; ++i)
				{
					if (Game->WeaponSlots[i].Type == WT) { bOwned = true; break; }
				}
				if (!bOwned)
				{
					bool bAlready = false;
					for (const auto& C : Choices)
						if (C.Key == WT) { bAlready = true; break; }
					if (!bAlready)
						Choices.Add(TPair<EWeaponType, int32>(WT, 1));
				}
			}
		}
	}

	return Choices;
}
