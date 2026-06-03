#pragma once

#include "CoreMinimal.h"
#include "Survivors/Logic/SurvivorsTypes.h"

/**
 * 武器・アイテムを 12 色パレットグループに集約。
 * ISM コンポーネント 12 個で全オブジェクトを描画（既存の Enemy/Gem ISM は別）。
 */
UENUM()
enum class EViewPalette : uint8
{
	Golden      = 0,   // Garlic, Cross, HeavenSword, UnholyVespers, HolyWand
	WarmOrange  = 1,   // Whip, FireWand, Axe, Hellfire, BloodyTear, DeathSpiral
	CoolBlue    = 2,   // MagicWand, SantaWater, LaBorra
	SilverWhite = 3,   // Knife, KingBible, ThousandEdge
	Purple      = 4,   // Runetracer, Pentagram, NoFuture, GorgeousMoon
	Yellow      = 5,   // LightningRing, ThunderLoop
	Pink        = 6,   // Peachone, Vandalier
	DarkNavy    = 7,   // EbonyWings
	Green       = 8,   // Laurel, SoulEater
	PickupHeal  = 9,   // FloorChicken (暖黄), LittleHeart (同 ISM でスケール差)
	PickupSpec  = 10,  // Rosary (白系), Orologion, Vacuum
	Destructible= 11,  // 破壊可能オブジェクト（橙）
	Count       = 12,
};

/** パレット → 代表色（ISM マテリアル色として使用） */
inline const FLinearColor ViewPaletteColors[static_cast<int32>(EViewPalette::Count)] = {
	FLinearColor(1.00f, 0.86f, 0.31f),  // Golden
	FLinearColor(1.00f, 0.39f, 0.08f),  // WarmOrange
	FLinearColor(0.39f, 0.71f, 1.00f),  // CoolBlue
	FLinearColor(0.84f, 0.84f, 1.00f),  // SilverWhite
	FLinearColor(0.78f, 0.20f, 0.82f),  // Purple
	FLinearColor(1.00f, 0.90f, 0.12f),  // Yellow
	FLinearColor(1.00f, 0.59f, 0.59f),  // Pink
	FLinearColor(0.20f, 0.12f, 0.47f),  // DarkNavy
	FLinearColor(0.31f, 0.86f, 0.24f),  // Green
	FLinearColor(1.00f, 0.71f, 0.20f),  // PickupHeal (暖黄)
	FLinearColor(0.90f, 0.90f, 0.95f),  // PickupSpec (白系)
	FLinearColor(1.00f, 0.59f, 0.20f),  // Destructible (橙)
};

/** 武器タイプ → パレットグループ */
inline EViewPalette GetWeaponPalette(EWeaponType Type)
{
	switch (Type)
	{
	case EWeaponType::Garlic:
	case EWeaponType::Cross:
	case EWeaponType::HeavenSword:
	case EWeaponType::HolyWand:
	case EWeaponType::UnholyVespers:   return EViewPalette::Golden;

	case EWeaponType::Whip:
	case EWeaponType::FireWand:
	case EWeaponType::Axe:
	case EWeaponType::Hellfire:
	case EWeaponType::BloodyTear:
	case EWeaponType::DeathSpiral:     return EViewPalette::WarmOrange;

	case EWeaponType::MagicWand:
	case EWeaponType::SantaWater:
	case EWeaponType::LaBorra:         return EViewPalette::CoolBlue;

	case EWeaponType::Knife:
	case EWeaponType::KingBible:
	case EWeaponType::ThousandEdge:    return EViewPalette::SilverWhite;

	case EWeaponType::Runetracer:
	case EWeaponType::Pentagram:
	case EWeaponType::NoFuture:
	case EWeaponType::GorgeousMoon:    return EViewPalette::Purple;

	case EWeaponType::LightningRing:
	case EWeaponType::ThunderLoop:     return EViewPalette::Yellow;

	case EWeaponType::Peachone:
	case EWeaponType::Vandalier:       return EViewPalette::Pink;

	case EWeaponType::EbonyWings:      return EViewPalette::DarkNavy;

	case EWeaponType::Laurel:
	case EWeaponType::SoulEater:       return EViewPalette::Green;

	default:                           return EViewPalette::SilverWhite;
	}
}
