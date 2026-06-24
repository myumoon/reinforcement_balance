#pragma once

#include "CoreMinimal.h"
#include "Survivors/Game/SurvivorsTypes.h"

/**
 * 武器・アイテムを 12 色パレットグループに集約。
 * ISM コンポーネント 12 個で全オブジェクトを描画（既存の Enemy/Gem ISM は別）。
 */
UENUM()
enum class EViewPalette : uint8
{
	Brown       = 0,   // Whip, BloodyTear
	DarkBlue    = 1,   // MagicWand, HolyWand
	Gray        = 2,   // Knife, ThousandEdge, Axe, DeathSpiral, Runetracer, NoFuture
	Cyan        = 3,   // Cross, HeavenSword, SantaWater, LaBorra
	Blue        = 4,   // KingBible, UnholyVespers
	Orange      = 5,   // FireWand, Hellfire
	Yellow      = 6,   // Garlic, SoulEater, Laurel, LightningRing, ThunderLoop
	Purple      = 7,   // Pentagram, GorgeousMoon, EbonyWings
	BluishWhite = 8,   // Peachone, Vandalier
	PickupHeal  = 9,   // FloorChicken (暖黄), LittleHeart (同 ISM でスケール差)
	PickupSpec  = 10,  // Rosary (白系), Orologion, Vacuum
	Destructible= 11,  // 破壊可能オブジェクト（橙）
	Count       = 12,
};

/** パレット → 代表色（ISM マテリアル色として使用） */
inline const FLinearColor ViewPaletteColors[static_cast<int32>(EViewPalette::Count)] = {
	FLinearColor(0.55f, 0.27f, 0.07f),  // Brown
	FLinearColor(0.09f, 0.28f, 0.71f),  // DarkBlue
	FLinearColor(0.60f, 0.60f, 0.60f),  // Gray
	FLinearColor(0.40f, 0.85f, 1.00f),  // Cyan
	FLinearColor(0.16f, 0.40f, 0.92f),  // Blue
	FLinearColor(1.00f, 0.55f, 0.10f),  // Orange
	FLinearColor(1.00f, 0.90f, 0.10f),  // Yellow
	FLinearColor(0.65f, 0.15f, 0.85f),  // Purple
	FLinearColor(0.82f, 0.88f, 1.00f),  // BluishWhite
	FLinearColor(1.00f, 0.71f, 0.20f),  // PickupHeal (暖黄)
	FLinearColor(0.90f, 0.90f, 0.95f),  // PickupSpec (白系)
	FLinearColor(1.00f, 0.59f, 0.20f),  // Destructible (橙)
};

/** 武器タイプ → パレットグループ */
inline EViewPalette GetWeaponPalette(EWeaponType Type)
{
	switch (Type)
	{
	case EWeaponType::Whip:
	case EWeaponType::BloodyTear:       return EViewPalette::Brown;

	case EWeaponType::MagicWand:
	case EWeaponType::HolyWand:         return EViewPalette::DarkBlue;

	case EWeaponType::Knife:
	case EWeaponType::ThousandEdge:
	case EWeaponType::Axe:
	case EWeaponType::DeathSpiral:
	case EWeaponType::Runetracer:
	case EWeaponType::NoFuture:         return EViewPalette::Gray;

	case EWeaponType::Cross:
	case EWeaponType::HeavenSword:
	case EWeaponType::SantaWater:
	case EWeaponType::LaBorra:          return EViewPalette::Cyan;

	case EWeaponType::KingBible:
	case EWeaponType::UnholyVespers:    return EViewPalette::Blue;

	case EWeaponType::FireWand:
	case EWeaponType::Hellfire:         return EViewPalette::Orange;

	case EWeaponType::Garlic:
	case EWeaponType::SoulEater:
	case EWeaponType::Laurel:
	case EWeaponType::LightningRing:
	case EWeaponType::ThunderLoop:      return EViewPalette::Yellow;

	case EWeaponType::Pentagram:
	case EWeaponType::GorgeousMoon:
	case EWeaponType::EbonyWings:       return EViewPalette::Purple;

	case EWeaponType::Peachone:
	case EWeaponType::Vandalier:        return EViewPalette::BluishWhite;

	default:                            return EViewPalette::Gray;
	}
}

/** 武器タイプ → 代表色（GetWeaponPalette + ViewPaletteColors のラッパー） */
inline FLinearColor GetWeaponColor(EWeaponType Type)
{
	return ViewPaletteColors[static_cast<int32>(GetWeaponPalette(Type))];
}

/** パッシブアイテムタイプ → 代表色 */
inline FLinearColor GetPassiveItemColor(EPassiveItemType Type)
{
	switch (Type)
	{
	case EPassiveItemType::None:          return FLinearColor(0.50f, 0.50f, 0.50f);
	case EPassiveItemType::Spinach:       return FLinearColor(0.31f, 0.86f, 0.24f);
	case EPassiveItemType::Armor:         return FLinearColor(0.84f, 0.84f, 1.00f);
	case EPassiveItemType::HollowHeart:   return FLinearColor(1.00f, 0.59f, 0.59f);
	case EPassiveItemType::Pummarola:     return FLinearColor(1.00f, 0.20f, 0.20f);
	case EPassiveItemType::EmptyTome:     return FLinearColor(0.39f, 0.71f, 1.00f);
	case EPassiveItemType::Candelabrador: return FLinearColor(1.00f, 0.90f, 0.12f);
	case EPassiveItemType::Bracer:        return FLinearColor(1.00f, 0.39f, 0.08f);
	case EPassiveItemType::Spellbinder:   return FLinearColor(0.78f, 0.20f, 0.82f);
	case EPassiveItemType::Duplicator:    return FLinearColor(0.84f, 0.84f, 1.00f);
	case EPassiveItemType::Wings:         return FLinearColor(0.40f, 0.90f, 0.90f);
	case EPassiveItemType::Attractorb:    return FLinearColor(0.31f, 0.86f, 0.24f);
	case EPassiveItemType::Clover:        return FLinearColor(0.31f, 0.86f, 0.24f);
	case EPassiveItemType::Crown:         return FLinearColor(1.00f, 0.86f, 0.31f);
	case EPassiveItemType::StoneMask:     return FLinearColor(0.60f, 0.60f, 0.60f);
	case EPassiveItemType::SkullOManiac:  return FLinearColor(0.78f, 0.20f, 0.82f);
	case EPassiveItemType::Tirajisu:      return FLinearColor(1.00f, 0.59f, 0.59f);
	case EPassiveItemType::TorronasBox:   return FLinearColor(1.00f, 0.39f, 0.08f);
	default:                              return FLinearColor(0.50f, 0.50f, 0.50f);
	}
}

/** 敵タイプID → 代表色（SurvivorsEnemyViewComponent と共有） */
inline FLinearColor GetEnemyTypeColor(int32 Type)
{
	switch (Type)
	{
	case 0:  return FLinearColor(1.f,  0.f,  0.f,  1.f);
	case 1:  return FLinearColor(1.f,  0.4f, 0.f,  1.f);
	case 2:  return FLinearColor(0.7f, 0.f,  0.5f, 1.f);
	case 3:  return FLinearColor(0.2f, 0.6f, 1.f,  1.f);
	case 4:  return FLinearColor(0.8f, 0.8f, 0.2f, 1.f);
	case 5:  return FLinearColor(0.5f, 0.9f, 0.4f, 1.f);
	case 6:  return FLinearColor(0.8f, 0.5f, 0.9f, 1.f);
	case 7:  return FLinearColor(0.4f, 0.9f, 0.9f, 1.f);
	case 8:  return FLinearColor(0.9f, 0.7f, 0.3f, 1.f);
	case 9:  return FLinearColor(0.9f, 0.3f, 0.7f, 1.f);
	case 10: return FLinearColor(0.6f, 0.2f, 1.f,  1.f);
	default: return FLinearColor(0.5f, 0.5f, 0.5f, 1.f);
	}
}
