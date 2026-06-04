#pragma once

#include "CoreMinimal.h"
#include "Survivors/Logic/SurvivorsTypes.h"

// ---- Garlic パラメータ構造体 -------------------------------------------------

struct FGarlicParams
{
	FDamage      Damage;
	float        HitInterval;  // 秒（FCooldownSeconds を constexpr に使うと初期化が複雑になるため float で統一）
	FSimRadius   AreaRadius;
};

// ---- 武器カリキュラム定数 ---------------------------------------------------

namespace SurvivorsGameConstants
{
	// ---- 基本定数 ----
	static constexpr int32 MaxWeaponSlots                = 6;   // 3 → 6 に拡大
	static constexpr int32 MaxPassiveSlots               = 6;   // 新規
	static constexpr int32 MaxWeaponLevel                = 8;
	static constexpr int32 MaxPlayerLevel                = 100;
	static constexpr float PhysicsDt                    = 1.f / 60.f;
	static constexpr float MaxGameTime                  = 1800.f;
	static constexpr float ContactHitInterval           = 0.5f;
	static constexpr float GarlicKnockbackStrength      = 10.f;

	// ---- obs 正規化用予約枠定数 ----
	// obs の type_norm = id / MaxWeaponTypeCountReserved で正規化する
	// 将来の武器追加時に type_norm 値域が変わらないよう余裕を持たせる
	static constexpr int32 MaxWeaponTypeCountReserved    = 64;  // EWeaponType は 29種、29〜63 は予約
	static constexpr int32 MaxPassiveTypeCountReserved   = 32;  // EPassiveItemType は 18種、18〜31 は予約

	// ---- obs バッファサイズ定数 ----
	static constexpr int32 MaxProjectileObs              = 32;  // 新規（5 dim each: dx,dy,r,vx,vy）
	static constexpr int32 MaxEnemyObs                   = 32;  // 20 → 32 に拡大
	static constexpr int32 MaxRedGemObs                  = 10;  // 新規
	static constexpr int32 MaxGreenGemObs                = 12;  // 新規
	static constexpr int32 MaxBlueGemObs                 = 12;  // 新規
	static constexpr int32 MaxFloorPickupObs             = 8;   // 新規
	static constexpr int32 MaxSpecialPickupObs           = 3;   // 新規
	static constexpr int32 MaxDestructibleObs            = 10;  // 新規

	// 後方互換用（旧 NumGemObs / MaxWeaponTypeSlots は廃止し定義を残すのみ）
	static constexpr int32 NumGemObs                    = MaxRedGemObs + MaxGreenGemObs + MaxBlueGemObs;  // 34 → obs 内ではタイプ別に分割
	static constexpr int32 MaxWeaponTypeSlots           = MaxWeaponTypeCountReserved;

	// obs 正規化用 Projectile 半径最大値（シム座標）
	static constexpr float MaxProjectileRadius          = 80.f;

	// Obs の Shield 正規化最大値（秒）
	static constexpr float MaxShieldDuration            = 8.f;

	// Obs の Armor / Regen 正規化最大値
	static constexpr float MaxArmorFlat                 = 60.f;   // Armor × 5Lv × 12f/Lv など想定
	static constexpr float MaxRegenPerSec               = 5.f;    // Pummarola × 5Lv など想定

	// 方向別密度/最近傍距離 obs の定数
	static constexpr int32 EnemyDensityDirCount            = 16;
	static constexpr float EnemyNearestDistanceMax         = 2400.0f;
	static constexpr float EnemyDensityNearDistanceMax     = 600.0f;
	static constexpr float EnemyDensityMidDistanceMax      = 1400.0f;
	static constexpr float EnemyDensityNearNormalizeFactor = 4.0f;
	static constexpr float EnemyDensityMidNormalizeFactor  = 6.0f;

	static constexpr int32 GemDensityDirCount              = 16;
	static constexpr float GemNearestDistanceMax           = 2400.0f;
	static constexpr float GemDensityNearDistanceMax       = 600.0f;
	static constexpr float GemDensityMidDistanceMax        = 1400.0f;
	static constexpr float GemDensityNearNormalizeFactor   = 6.0f;
	static constexpr float GemDensityMidNormalizeFactor    = 10.0f;

	// ---- 方向定数 ----
	inline const FVector2D RayDirs[8] = {
		FVector2D( 1.f,      0.f     ),
		FVector2D( 0.7071f,  0.7071f ),
		FVector2D( 0.f,      1.f     ),
		FVector2D(-0.7071f,  0.7071f ),
		FVector2D(-1.f,      0.f     ),
		FVector2D(-0.7071f, -0.7071f ),
		FVector2D( 0.f,     -1.f     ),
		FVector2D( 0.7071f, -0.7071f ),
	};

	// ---- Garlic / Soul Eater パラメータテーブル --------------------------------
	// 原作 Vampire Survivors Wiki 値（Garlic Lv1〜8）
	// Damage, HitInterval(s), AreaRadius(u)
	// TODO: Soul Eater テーブルは PR2 で Wiki 取得後に完成させる（暫定値）
	inline constexpr FGarlicParams GarlicTable[MaxWeaponLevel] = {
		{ FDamage( 5.f), 1.30f, FSimRadius(25.f) },  // Lv1
		{ FDamage( 5.f), 1.25f, FSimRadius(30.f) },  // Lv2
		{ FDamage(10.f), 1.20f, FSimRadius(35.f) },  // Lv3
		{ FDamage(10.f), 1.15f, FSimRadius(40.f) },  // Lv4
		{ FDamage(15.f), 1.10f, FSimRadius(45.f) },  // Lv5
		{ FDamage(15.f), 1.05f, FSimRadius(50.f) },  // Lv6
		{ FDamage(20.f), 1.00f, FSimRadius(55.f) },  // Lv7
		{ FDamage(20.f), 0.95f, FSimRadius(60.f) },  // Lv8
	};

	// TODO(PR2): Soul Eater（Garlic 進化）テーブル
	// 原作値未確認のため暫定的に Garlic 値を流用（プレースホルダー）
	inline constexpr FGarlicParams SoulEaterTable[MaxWeaponLevel] = {
		{ FDamage(10.f), 1.00f, FSimRadius(30.f) },  // Lv1 (暫定)
		{ FDamage(10.f), 0.95f, FSimRadius(35.f) },  // Lv2 (暫定)
		{ FDamage(15.f), 0.90f, FSimRadius(40.f) },  // Lv3 (暫定)
		{ FDamage(15.f), 0.85f, FSimRadius(45.f) },  // Lv4 (暫定)
		{ FDamage(20.f), 0.80f, FSimRadius(50.f) },  // Lv5 (暫定)
		{ FDamage(20.f), 0.75f, FSimRadius(55.f) },  // Lv6 (暫定)
		{ FDamage(25.f), 0.70f, FSimRadius(60.f) },  // Lv7 (暫定)
		{ FDamage(30.f), 0.65f, FSimRadius(65.f) },  // Lv8 (暫定)
	};

	// ---- 他の武器パラメータ構造体（PR2 で各テーブル完成） ---------------------
	// struct FWhipParams    { FDamage Damage; FCooldownSeconds Cooldown; FSimRadius Width; FSimRadius Height; };
	// ... TODO(PR2): 残り 14 武器のパラメータ収集・テーブル定義

	// ---- パッシブアイテム定数 ------------------------------------------------

	// アイテム別最大レベルテーブル（EPassiveItemType のインデックスと対応）
	// None=0, Spinach=5, Armor=5, HollowHeart=5, Pummarola=5, EmptyTome=5,
	// Candelabrador=5, Bracer=5, Spellbinder=5, Duplicator=2, Wings=5,
	// Attractorb=5, Clover=5, Crown=5, StoneMask=5, SkullOManiac=5, Tirajisu=2, TorronasBox=9
	inline constexpr int32 PassiveMaxLevel[18] = {
		0, 5, 5, 5, 5, 5, 5, 5, 5, 2, 5, 5, 5, 5, 5, 5, 2, 9
	};

	// ---- ジェム・基本定数 ---------------------------------------------------

	inline constexpr float GemXPValues[3] = { 1.f, 5.f, 10.f };

	inline constexpr EGemType GemDropTable[11] = {
		EGemType::Blue,
		EGemType::Blue,
		EGemType::Blue,
		EGemType::Blue,
		EGemType::Green,
		EGemType::Green,
		EGemType::Green,
		EGemType::Blue,
		EGemType::Green,
		EGemType::Green,
		EGemType::Red,
	};

	// ---- 進化条件テーブル ---------------------------------------------------

	struct FEvolutionRule
	{
		EWeaponType      BaseWeapon;
		EWeaponType      EvolvedWeapon;
		EPassiveItemType RequiredPassive;                          // None の場合は Union 判定
		EWeaponType      UnionPartner = EWeaponType::None;         // Vandalier 用
	};

	inline constexpr FEvolutionRule EvolutionTable[] = {
		{ EWeaponType::Garlic,        EWeaponType::SoulEater,     EPassiveItemType::Pummarola  },
		{ EWeaponType::Whip,          EWeaponType::BloodyTear,    EPassiveItemType::HollowHeart},
		{ EWeaponType::MagicWand,     EWeaponType::HolyWand,      EPassiveItemType::EmptyTome  },
		{ EWeaponType::Knife,         EWeaponType::ThousandEdge,  EPassiveItemType::Bracer     },
		{ EWeaponType::Axe,           EWeaponType::DeathSpiral,   EPassiveItemType::Candelabrador},
		{ EWeaponType::Cross,         EWeaponType::HeavenSword,   EPassiveItemType::Clover     },
		{ EWeaponType::KingBible,     EWeaponType::UnholyVespers, EPassiveItemType::Spellbinder},
		{ EWeaponType::FireWand,      EWeaponType::Hellfire,      EPassiveItemType::Spinach    },
		{ EWeaponType::SantaWater,    EWeaponType::LaBorra,       EPassiveItemType::Attractorb },
		{ EWeaponType::Runetracer,    EWeaponType::NoFuture,      EPassiveItemType::Armor      },
		{ EWeaponType::LightningRing, EWeaponType::ThunderLoop,   EPassiveItemType::Duplicator },
		{ EWeaponType::Pentagram,     EWeaponType::GorgeousMoon,  EPassiveItemType::Crown      },
		// Union: Peachone Lv8 + EbonyWings Lv8 → Vandalier
		{ EWeaponType::Peachone,      EWeaponType::Vandalier,     EPassiveItemType::None, EWeaponType::EbonyWings },
	};
}
