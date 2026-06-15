#pragma once

#include "CoreMinimal.h"
#include "Survivors/Logic/SurvivorsWikiSpec.h"
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
	static constexpr int32 MaxWeaponLevel                = SurvivorsWikiSpec::BaseWeaponMaxLevel;
	static constexpr int32 MaxPlayerLevel                = 100;
	static constexpr float PhysicsDt                    = 1.f / 60.f;
	static constexpr float MaxGameTime                  = 1800.f;
	static constexpr float ContactHitInterval           = 0.5f;
	static constexpr float GarlicKnockbackStrength      = 10.f;
	static constexpr float StandardMaxPlayerHP          = SurvivorsWikiSpec::StandardMaxPlayerHP;
	static constexpr float StandardMoveSpeed            = SurvivorsWikiSpec::StandardMoveSpeed;
	static constexpr float BaseGemPickupRadius          = SurvivorsWikiSpec::BaseGemPickupRadius;

	// ---- obs 正規化用予約枠定数 ----
	// obs の type_norm = id / MaxWeaponTypeCountReserved で正規化する
	// 将来の武器追加時に type_norm 値域が変わらないよう余裕を持たせる
	static constexpr int32 MaxWeaponTypeCountReserved    = 64;  // EWeaponType は 29種、29〜63 は予約
	static constexpr int32 MaxPassiveTypeCountReserved   = 32;  // EPassiveItemType は 18種、18〜31 は予約

	// ---- obs バッファサイズ定数 ----
	static constexpr int32 MaxProjectileObs              = 32;  // 新規（6 dim each: dx,dy,r,vx,vy,warning）
	static constexpr int32 ProjectileObsStride           = 6;
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

	// Santa Water / La Borra の着弾予兆時間（秒）
	// 本家VSは予告円なし。UE5では0.1sの短い予告のみ（落下位置が見える程度）。
	static constexpr float SantaWaterWarningTime        = 0.10f;

	// SantaWater low-amount（Amount<4）のランダムドロップ範囲半径
	// 1発目は最近傍敵、残りはこの半径内のランダム位置に落下
	static constexpr float SantaWaterRandomDropRadius   = 150.f;

	// SantaWater high-amount（Amount>=4）の円形配置半径
	// OBSERVED: santa_water.jpg 337.5px × (800u/1920px) ≈ 140.6u → 採用値 140u
	static constexpr float SantaWaterCircleRadius       = 140.f;

	// SantaWater high-amount: 各 drop の角度ステップ（30°固定）
	static constexpr float SantaWaterCircleDropStep     = UE_PI / 6.f;  // 30° in radians

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
	// Damage, HitInterval(s), AreaRadius(u) ※AreaRadius は Area100%=25u 換算
	inline constexpr FGarlicParams GarlicTable[MaxWeaponLevel] = {
		{ FDamage( 5.f), 1.30f, FSimRadius(25.f) },  // Lv1: D=5,  CD=1.30, Area=100%
		{ FDamage( 7.f), 1.30f, FSimRadius(35.f) },  // Lv2: D+2,  Area+40%
		{ FDamage( 8.f), 1.20f, FSimRadius(35.f) },  // Lv3: D+1,  CD-0.10
		{ FDamage( 9.f), 1.20f, FSimRadius(40.f) },  // Lv4: D+1,  Area+20%
		{ FDamage(11.f), 1.10f, FSimRadius(40.f) },  // Lv5: D+2,  CD-0.10
		{ FDamage(12.f), 1.10f, FSimRadius(45.f) },  // Lv6: D+1,  Area+20%
		{ FDamage(13.f), 1.00f, FSimRadius(45.f) },  // Lv7: D+1,  CD-0.10
		{ FDamage(15.f), 1.00f, FSimRadius(50.f) },  // Lv8: D+2,  Area+20%
	};

	// Soul Eater（Garlic 進化）: Damage=20, Area=300%(75u), CD=1.0s
	// 進化武器は MaxLevel=1 のため Lv1 のみ参照される
	inline constexpr FGarlicParams SoulEaterTable[MaxWeaponLevel] = {
		{ FDamage(20.f), 1.00f, FSimRadius(75.f) },  // Lv1
		{ FDamage(20.f), 1.00f, FSimRadius(75.f) },  // Lv2 (未使用)
		{ FDamage(20.f), 1.00f, FSimRadius(75.f) },  // Lv3 (未使用)
		{ FDamage(20.f), 1.00f, FSimRadius(75.f) },  // Lv4 (未使用)
		{ FDamage(20.f), 1.00f, FSimRadius(75.f) },  // Lv5 (未使用)
		{ FDamage(20.f), 1.00f, FSimRadius(75.f) },  // Lv6 (未使用)
		{ FDamage(20.f), 1.00f, FSimRadius(75.f) },  // Lv7 (未使用)
		{ FDamage(20.f), 1.00f, FSimRadius(75.f) },  // Lv8 (未使用)
	};

	// ---- 残り 14 武器パラメータ構造体・テーブル -----------------------------------

	struct FWhipParams
	{
		float Damage;
		float Cooldown;   // 秒
		float Width;      // シム座標（横幅半径）
		float Height;     // シム座標（縦幅半径）
		int32 Amount;
	};

	// Whip: Area100%=50u。CD は全レベル固定 1.35s。
	// Amount: Lv1 baseline=2（左→右交互スイング、0.3s間隔）。Lv2以降 Amount 変化なし。
	inline constexpr FWhipParams WhipTable[MaxWeaponLevel] = {
		{ 10.f, 1.35f, 50.f, 15.f, 2 },  // Lv1: D=10, Area=100%, Amount=2(baseline)
		{ 10.f, 1.35f, 50.f, 15.f, 2 },  // Lv2: D same
		{ 15.f, 1.35f, 50.f, 15.f, 2 },  // Lv3: D+5
		{ 20.f, 1.35f, 55.f, 15.f, 2 },  // Lv4: D+5, Area+10%
		{ 25.f, 1.35f, 55.f, 15.f, 2 },  // Lv5: D+5
		{ 30.f, 1.35f, 60.f, 15.f, 2 },  // Lv6: D+5, Area+10%
		{ 35.f, 1.35f, 60.f, 15.f, 2 },  // Lv7: D+5
		{ 40.f, 1.35f, 60.f, 15.f, 2 },  // Lv8: D+5
	};

	// BloodyTear (Whip 進化): Damage=40, Area=130%(65u), CD=1.35s
	// 進化武器は MaxLevel=1 のため Lv1 のみ参照される
	inline constexpr FWhipParams BloodyTearTable[MaxWeaponLevel] = {
		{ 40.f, 1.35f, 65.f, 15.f, 2 },  // Lv1
		{ 40.f, 1.35f, 65.f, 15.f, 2 },  // Lv2 (未使用)
		{ 40.f, 1.35f, 65.f, 15.f, 2 },  // Lv3 (未使用)
		{ 40.f, 1.35f, 65.f, 15.f, 2 },  // Lv4 (未使用)
		{ 40.f, 1.35f, 65.f, 15.f, 2 },  // Lv5 (未使用)
		{ 40.f, 1.35f, 65.f, 15.f, 2 },  // Lv6 (未使用)
		{ 40.f, 1.35f, 65.f, 15.f, 2 },  // Lv7 (未使用)
		{ 40.f, 1.35f, 65.f, 15.f, 2 },  // Lv8 (未使用)
	};

	// ノックバック強度定数 (シム座標単位): 仕様の Knockback 値をシミュレーション値に変換
	// 仕様 Knockback=1 を基準とし、フィールド幅(2000u)・敵速度(100u/s)に対して妥当な値
	static constexpr float KnockbackSim_Half = 10.f;  // Knockback=0.5（Knife）
	static constexpr float KnockbackSim_1    = 20.f;  // Knockback=1（標準）
	static constexpr float KnockbackSim_2    = 40.f;  // Knockback=2（Peachone/EbonyWings）
	static constexpr float KnockbackSim_4    = 80.f;  // Knockback=4（UnholyVespers）
	static constexpr float KnockbackSim_6    = 120.f; // Knockback=6（HeavenSword）

	struct FMagicWandParams
	{
		float Damage;
		float Cooldown;
		float Speed;
		int32 Amount;
		int32 Pierce;  // 貫通数: 1=単体命中, N=N体貫通
	};

	// MagicWand: Speed は全レベル固定。
	// OBSERVED: magic_wand_bullet2.mp4 frame 289→300, 61.8px/11f × 0.4167u/px ≈ 140u/s
	// acceptance range 120-165u/s。wiki の 100%=600u は Camera Z=2000 基準と合わないため動画値を採用。
	// Amount: Lv1 baseline=2（1発目即時、2発目0.1s後）。Lv2=3, Lv4=4, Lv6=5。
	inline constexpr FMagicWandParams MagicWandTable[MaxWeaponLevel] = {
		{ 10.f, 1.20f, 140.f, 2, 1 },  // Lv1: D=10, CD=1.20, Speed=140u/s, Amount=2(baseline)
		{ 10.f, 1.20f, 140.f, 3, 1 },  // Lv2: Amount+1
		{ 10.f, 1.00f, 140.f, 3, 1 },  // Lv3: CD-0.20
		{ 10.f, 1.00f, 140.f, 4, 1 },  // Lv4: Amount+1
		{ 20.f, 1.00f, 140.f, 4, 1 },  // Lv5: D+10
		{ 20.f, 1.00f, 140.f, 5, 1 },  // Lv6: Amount+1
		{ 20.f, 1.00f, 140.f, 5, 2 },  // Lv7: Pierce+1
		{ 30.f, 1.00f, 140.f, 5, 2 },  // Lv8: D+10
	};

	// HolyWand (MagicWand 進化): Damage=30, CD=0.5s, Amount=4, Pierce=2
	// Speed: wiki では MagicWand の 200%。動画由来 MagicWand 140u/s × 2.0 = 280u/s。
	// HolyWand の動画測定値は未取得のため、MagicWand との相対倍率 2.0x を維持する。
	// 進化武器は MaxLevel=1 のため Lv1 のみ参照される
	inline constexpr FMagicWandParams HolyWandTable[MaxWeaponLevel] = {
		{ 30.f, 0.50f, 280.f, 4, 2 },  // Lv1: Speed=280u/s(MagicWand 140×2.0)
		{ 30.f, 0.50f, 280.f, 4, 2 },  // Lv2 (未使用)
		{ 30.f, 0.50f, 280.f, 4, 2 },  // Lv3 (未使用)
		{ 30.f, 0.50f, 280.f, 4, 2 },  // Lv4 (未使用)
		{ 30.f, 0.50f, 280.f, 4, 2 },  // Lv5 (未使用)
		{ 30.f, 0.50f, 280.f, 4, 2 },  // Lv6 (未使用)
		{ 30.f, 0.50f, 280.f, 4, 2 },  // Lv7 (未使用)
		{ 30.f, 0.50f, 280.f, 4, 2 },  // Lv8 (未使用)
	};

	struct FKnifeParams
	{
		float Damage;
		float Cooldown;
		float Speed;
		int32 Amount;
		int32 Pierce;  // 貫通数: 1=単体命中, 2=2体貫通, 3=3体貫通
	};

	// Knife: CD=1.0s 全レベル固定
	// Speed: OBSERVED: knife_bullet3.mp4 Track1=326.2u/s, Track2=325.8u/s → 採用値326u/s
	// Pierce: Lv1=1体, Lv5=2体, Lv8=3体
	// Amount: Lv1 baseline=2（1発目即時、2発目0.1s後）。Lv2=3, Lv3=4, Lv4=5, Lv6=6, Lv7=7。
	inline constexpr FKnifeParams KnifeTable[MaxWeaponLevel] = {
		{  6.5f, 1.00f, 326.f, 2, 1 },  // Lv1: D=6.5, Speed=326u/s(OBSERVED), Amount=2(baseline), Pierce=1
		{  6.5f, 1.00f, 326.f, 3, 1 },  // Lv2: Amount+1
		{ 11.5f, 1.00f, 326.f, 4, 1 },  // Lv3: D+5, Amount+1
		{ 11.5f, 1.00f, 326.f, 5, 1 },  // Lv4: Amount+1
		{ 11.5f, 1.00f, 326.f, 5, 2 },  // Lv5: Pierce+1
		{ 11.5f, 1.00f, 326.f, 6, 2 },  // Lv6: Amount+1
		{ 16.5f, 1.00f, 326.f, 7, 2 },  // Lv7: D+5, Amount+1
		{ 16.5f, 1.00f, 326.f, 7, 3 },  // Lv8: Pierce+1
	};

	// ThousandEdge (Knife 進化): Damage=16.5, CD=0.35s, Speed=Knife×150%=489u/s, Amount=6, Pierce=3
	// 進化武器は MaxLevel=1 のため Lv1 のみ参照される
	inline constexpr FKnifeParams ThousandEdgeTable[MaxWeaponLevel] = {
		{ 16.5f, 0.35f, 489.f, 6, 3 },  // Lv1: Speed=326×1.5=489u/s
		{ 16.5f, 0.35f, 489.f, 6, 3 },  // Lv2 (未使用)
		{ 16.5f, 0.35f, 489.f, 6, 3 },  // Lv3 (未使用)
		{ 16.5f, 0.35f, 489.f, 6, 3 },  // Lv4 (未使用)
		{ 16.5f, 0.35f, 489.f, 6, 3 },  // Lv5 (未使用)
		{ 16.5f, 0.35f, 489.f, 6, 3 },  // Lv6 (未使用)
		{ 16.5f, 0.35f, 489.f, 6, 3 },  // Lv7 (未使用)
		{ 16.5f, 0.35f, 489.f, 6, 3 },  // Lv8 (未使用)
	};

	struct FAxeParams
	{
		float Damage;
		float Cooldown;
		float Speed;
		float ArcHeight;  // 最大弧高さ（シム座標）
		int32 Amount;     // 発射数
		int32 Pierce;     // 貫通数: Lv1=3, Lv4=5, Lv7=7
	};

	// Axe: CD=4.0s 全レベル固定、Speed=360u(100%) 全レベル固定
	// Amount: Lv1 baseline=2（1発目即時、2発目0.2s後）。Lv2=3, Lv5=4。Pierce: Lv1=3体,Lv4=5体,Lv7=7体。
	// ArcHeight の意味: 重力式 g = InitVelY²/ArcHeight における中間パラメータ。
	// 物理的な頂点高さ ≈ ArcHeight/2 (連続近似)。Lv1 ArcHeight=120 → apex ≈ 60u。
	// OBSERVED: axe_bullet2.mp4 frame 80-100, 130-150px × 0.4167u/px ≈ 54-63u。acceptance 50-75u。
	inline constexpr FAxeParams AxeTable[MaxWeaponLevel] = {
		{  20.f, 4.00f, 360.f, 120.f, 2, 3 },  // Lv1: D=20, Amount=2(baseline), ArcHeight=120→apex≈60u
		{  20.f, 4.00f, 360.f, 120.f, 3, 3 },  // Lv2: Amount+1
		{  40.f, 4.00f, 360.f, 120.f, 3, 3 },  // Lv3: D+20
		{  40.f, 4.00f, 360.f, 120.f, 3, 5 },  // Lv4: Pierce+2
		{  40.f, 4.00f, 360.f, 120.f, 4, 5 },  // Lv5: Amount+1
		{  60.f, 4.00f, 360.f, 120.f, 4, 5 },  // Lv6: D+20
		{  60.f, 4.00f, 360.f, 120.f, 4, 7 },  // Lv7: Pierce+2
		{  80.f, 4.00f, 360.f, 120.f, 4, 7 },  // Lv8: D+20
	};

	// DeathSpiral (Axe 進化): Damage=60, CD=4.0s, Speed=288u(80%), ArcHeight=144u, Pierce=1000(無限貫通)
	// 進化武器は MaxLevel=1 のため Lv1 のみ参照される
	inline constexpr FAxeParams DeathSpiralTable[MaxWeaponLevel] = {
		{  60.f, 4.00f, 288.f, 144.f, 9, 1000 },  // Lv1
		{  60.f, 4.00f, 288.f, 144.f, 9, 1000 },  // Lv2 (未使用)
		{  60.f, 4.00f, 288.f, 144.f, 9, 1000 },  // Lv3 (未使用)
		{  60.f, 4.00f, 288.f, 144.f, 9, 1000 },  // Lv4 (未使用)
		{  60.f, 4.00f, 288.f, 144.f, 9, 1000 },  // Lv5 (未使用)
		{  60.f, 4.00f, 288.f, 144.f, 9, 1000 },  // Lv6 (未使用)
		{  60.f, 4.00f, 288.f, 144.f, 9, 1000 },  // Lv7 (未使用)
		{  60.f, 4.00f, 288.f, 144.f, 9, 1000 },  // Lv8 (未使用)
	};

	struct FCrossParams
	{
		float Damage;
		float Cooldown;
		float Speed;
		float Radius;
		int32 Amount;
		float KnockbackStrength;
	};

	// Cross: CD=2.0s 全レベル固定。Speed は Lv3 で +25%、Lv6 で +25%。
	// Speed: 100%=320u → 125%=400u → 150%=480u
	// Amount: Lv1 baseline=2（1発目即時、2発目0.1s後）。Lv4=3, Lv7=4。
	inline constexpr FCrossParams CrossTable[MaxWeaponLevel] = {
		{  5.f, 2.00f, 320.f, 12.0f, 2, KnockbackSim_1 },  // Lv1: D=5, Speed=100%, Amount=2(baseline)
		{ 15.f, 2.00f, 320.f, 12.0f, 2, KnockbackSim_1 },  // Lv2: D+10
		{ 15.f, 2.00f, 400.f, 13.2f, 2, KnockbackSim_1 },  // Lv3: Speed+25%, Area+10%
		{ 15.f, 2.00f, 400.f, 13.2f, 3, KnockbackSim_1 },  // Lv4: Amount+1
		{ 25.f, 2.00f, 400.f, 13.2f, 3, KnockbackSim_1 },  // Lv5: D+10
		{ 25.f, 2.00f, 480.f, 14.4f, 3, KnockbackSim_1 },  // Lv6: Speed+25%, Area+10%
		{ 25.f, 2.00f, 480.f, 14.4f, 4, KnockbackSim_1 },  // Lv7: Amount+1
		{ 35.f, 2.00f, 480.f, 14.4f, 4, KnockbackSim_1 },  // Lv8: D+10
	};

	// HeavenSword (Cross 進化): Damage=77, CD=3.3s, Speed=640u(200%)
	// 進化武器は MaxLevel=1 のため Lv1 のみ参照される
	inline constexpr FCrossParams HeavenSwordTable[MaxWeaponLevel] = {
		{  77.f, 3.30f, 640.f, 14.4f, 1, KnockbackSim_6 },  // Lv1
		{  77.f, 3.30f, 640.f, 14.4f, 1, KnockbackSim_6 },  // Lv2 (未使用)
		{  77.f, 3.30f, 640.f, 14.4f, 1, KnockbackSim_6 },  // Lv3 (未使用)
		{  77.f, 3.30f, 640.f, 14.4f, 1, KnockbackSim_6 },  // Lv4 (未使用)
		{  77.f, 3.30f, 640.f, 14.4f, 1, KnockbackSim_6 },  // Lv5 (未使用)
		{  77.f, 3.30f, 640.f, 14.4f, 1, KnockbackSim_6 },  // Lv6 (未使用)
		{  77.f, 3.30f, 640.f, 14.4f, 1, KnockbackSim_6 },  // Lv7 (未使用)
		{  77.f, 3.30f, 640.f, 14.4f, 1, KnockbackSim_6 },  // Lv8 (未使用)
	};

	struct FKingBibleParams
	{
		float Damage;
		float Cooldown;
		float Duration;
		float OrbitRadius;
		int32 Amount;
		float RotSpeed;    // rad/sec
		float KnockbackStrength;
	};

	// KingBible: CD=3.0s。OrbitRadius: 100%=50u,125%=62.5u,150%=75u
	// RotSpeed: 100%=4.0rad/s,130%=5.2rad/s,160%=6.4rad/s
	// Amount: Lv1 baseline=2（2冊が円周上に180°等間隔配置）。Lv2=3, Lv5=4, Lv8=5。
	inline constexpr FKingBibleParams KingBibleTable[MaxWeaponLevel] = {
		{ 10.f, 3.f, 3.0f, 50.0f, 2, 4.0f, KnockbackSim_1 },  // Lv1: D=10, OR=100%, RS=100%, Amount=2(baseline)
		{ 10.f, 3.f, 3.0f, 50.0f, 3, 4.0f, KnockbackSim_1 },  // Lv2: Amount+1
		{ 10.f, 3.f, 3.0f, 62.5f, 3, 5.2f, KnockbackSim_1 },  // Lv3: Area+25%, Speed+30%
		{ 20.f, 3.f, 3.5f, 62.5f, 3, 5.2f, KnockbackSim_1 },  // Lv4: D+10, Duration+0.5s
		{ 20.f, 3.f, 3.5f, 62.5f, 4, 5.2f, KnockbackSim_1 },  // Lv5: Amount+1
		{ 20.f, 3.f, 3.5f, 75.0f, 4, 6.4f, KnockbackSim_1 },  // Lv6: Area+25%, Speed+30%
		{ 30.f, 3.f, 4.0f, 75.0f, 4, 6.4f, KnockbackSim_1 },  // Lv7: D+10, Duration+0.5s
		{ 30.f, 3.f, 4.0f, 75.0f, 5, 6.4f, KnockbackSim_1 },  // Lv8: Amount+1
	};

	// UnholyVespers (KingBible 進化): Damage=30, OrbitRadius=87.5u(175%), RotSpeed=6.0(150%), Amount=4
	// 進化武器は MaxLevel=1 のため Lv1 のみ参照される
	inline constexpr FKingBibleParams UnholyVespersTable[MaxWeaponLevel] = {
		{ 30.f, 3.f, 3.0f, 87.5f, 4, 6.0f, KnockbackSim_4 },  // Lv1
		{ 30.f, 3.f, 3.0f, 87.5f, 4, 6.0f, KnockbackSim_4 },  // Lv2 (未使用)
		{ 30.f, 3.f, 3.0f, 87.5f, 4, 6.0f, KnockbackSim_4 },  // Lv3 (未使用)
		{ 30.f, 3.f, 3.0f, 87.5f, 4, 6.0f, KnockbackSim_4 },  // Lv4 (未使用)
		{ 30.f, 3.f, 3.0f, 87.5f, 4, 6.0f, KnockbackSim_4 },  // Lv5 (未使用)
		{ 30.f, 3.f, 3.0f, 87.5f, 4, 6.0f, KnockbackSim_4 },  // Lv6 (未使用)
		{ 30.f, 3.f, 3.0f, 87.5f, 4, 6.0f, KnockbackSim_4 },  // Lv7 (未使用)
		{ 30.f, 3.f, 3.0f, 87.5f, 4, 6.0f, KnockbackSim_4 },  // Lv8 (未使用)
	};

	struct FFireWandParams
	{
		float Damage;
		float Cooldown;
		float Speed;
		float ExplosionRadius;
		int32 Amount;
	};

	// FireWand: CD=3.0s 全レベル固定。Lv1 から4発の扇状発射。
	// OBSERVED: fire_wand_bullet4.mp4 frame 290→300, 38-39px/10f × 0.4167u/px ≈ 96u/s (Lv1 75%)
	// acceptance range 80-115u/s。wiki の 75%=360u は Camera Z=2000 基準と合わないため動画値を採用。
	// 基準 100% = 96/0.75 = 128u/s。各 Lv は wiki の相対 % で比例換算。
	inline constexpr FFireWandParams FireWandTable[MaxWeaponLevel] = {
		{ 20.f, 3.00f,  96.f, 30.f, 4 },  // Lv1: D=20, Speed=75%=96u/s(OBSERVED)
		{ 30.f, 3.00f,  96.f, 30.f, 4 },  // Lv2: D+10
		{ 40.f, 3.00f, 122.f, 30.f, 4 },  // Lv3: D+10, Speed+20%(95%=122u/s)
		{ 50.f, 3.00f, 122.f, 30.f, 4 },  // Lv4: D+10
		{ 60.f, 3.00f, 147.f, 30.f, 4 },  // Lv5: D+10, Speed+20%(115%=147u/s)
		{ 70.f, 3.00f, 147.f, 30.f, 4 },  // Lv6: D+10
		{ 80.f, 3.00f, 173.f, 30.f, 4 },  // Lv7: D+10, Speed+20%(135%=173u/s)
		{ 90.f, 3.00f, 173.f, 30.f, 4 },  // Lv8: D+10
	};

	// Hellfire (FireWand 進化): Damage=100, CD=3.0s, Speed=100%=128u/s, ExpRad=30u
	// wiki の 100% を FireWand 動画由来基準 128u/s に換算。Hellfire 動画測定値は未取得のため比例維持。
	// 進化武器は MaxLevel=1 のため Lv1 のみ参照される
	inline constexpr FFireWandParams HellfireTable[MaxWeaponLevel] = {
		{ 100.f, 3.00f, 128.f, 30.f, 4 },  // Lv1: Speed=100%=128u/s
		{ 100.f, 3.00f, 128.f, 30.f, 4 },  // Lv2 (未使用)
		{ 100.f, 3.00f, 128.f, 30.f, 4 },  // Lv3 (未使用)
		{ 100.f, 3.00f, 128.f, 30.f, 4 },  // Lv4 (未使用)
		{ 100.f, 3.00f, 128.f, 30.f, 4 },  // Lv5 (未使用)
		{ 100.f, 3.00f, 128.f, 30.f, 4 },  // Lv6 (未使用)
		{ 100.f, 3.00f, 128.f, 30.f, 4 },  // Lv7 (未使用)
		{ 100.f, 3.00f, 128.f, 30.f, 4 },  // Lv8 (未使用)
	};

	struct FSantaWaterParams
	{
		float Damage;
		float Cooldown;
		float Radius;
		float Duration;
		int32 Amount;
	};

	// SantaWater: CD=4.5s 全レベル固定。Radius: 100%=30u。Duration は Lv3,5,7 で増加。
	// Amount: Lv1 baseline=2（1個目即時、2個目0.3s後）。Amount<4: 敵付近 drop。Amount>=4: 円形配置。
	// KingBible の orbit 等間隔仕様とは別物（SantaWater は敵ターゲット or 円形配置）。
	inline constexpr FSantaWaterParams SantaWaterTable[MaxWeaponLevel] = {
		{ 10.f, 4.50f, 30.f, 2.00f, 2 },  // Lv1: D=10, R=100%, Dur=2.0, Amount=2(baseline)
		{ 10.f, 4.50f, 36.f, 2.00f, 3 },  // Lv2: R+20%(120%), Amount+1
		{ 20.f, 4.50f, 36.f, 2.50f, 3 },  // Lv3: D+10, Dur+0.50
		{ 20.f, 4.50f, 42.f, 2.50f, 4 },  // Lv4: R+20%(140%), Amount+1 → 円形配置開始
		{ 30.f, 4.50f, 42.f, 2.75f, 4 },  // Lv5: D+10, Dur+0.25
		{ 30.f, 4.50f, 48.f, 2.75f, 5 },  // Lv6: R+20%(160%), Amount+1
		{ 35.f, 4.50f, 48.f, 3.00f, 5 },  // Lv7: D+5,  Dur+0.25
		{ 40.f, 4.50f, 54.f, 3.00f, 5 },  // Lv8: D+5,  R+20%(180%)
	};

	// LaBorra (SantaWater 進化): Damage=40, CD=4.0s, Radius=60u(200%), Duration=4.0s, Amount=4
	// 進化武器は MaxLevel=1 のため Lv1 のみ参照される
	inline constexpr FSantaWaterParams LaBorraTable[MaxWeaponLevel] = {
		{ 40.f, 4.00f, 60.f, 4.0f, 4 },  // Lv1
		{ 40.f, 4.00f, 60.f, 4.0f, 4 },  // Lv2 (未使用)
		{ 40.f, 4.00f, 60.f, 4.0f, 4 },  // Lv3 (未使用)
		{ 40.f, 4.00f, 60.f, 4.0f, 4 },  // Lv4 (未使用)
		{ 40.f, 4.00f, 60.f, 4.0f, 4 },  // Lv5 (未使用)
		{ 40.f, 4.00f, 60.f, 4.0f, 4 },  // Lv6 (未使用)
		{ 40.f, 4.00f, 60.f, 4.0f, 4 },  // Lv7 (未使用)
		{ 40.f, 4.00f, 60.f, 4.0f, 4 },  // Lv8 (未使用)
	};

	struct FRunetracerParams
	{
		float Damage;
		float Cooldown;
		float Speed;
		float Duration;
		int32 Amount;
		int32 MaxBounce;
	};

	// Runetracer: CD=3.0s 全レベル固定。
	// Speed: OBSERVED: 弾が画面下(225u)まで70f(60fps)で到達 → 225×60/70 ≈ 193u/s (Lv1=100%)
	// Duration: OBSERVED: 往復140f(60fps) = 140/60 ≈ 2.33s (Lv1 baseline)
	// MaxBounce は既存の壁反射寿命近似。Amount: Lv1=2, Lv4=3, Lv7=4。
	inline constexpr FRunetracerParams RunetracerTable[MaxWeaponLevel] = {
		{ 10.f, 3.00f, 193.f, 2.33f, 2, 3 },  // Lv1: D=10, Speed=100%=193u/s(OBSERVED), Duration=2.33s(OBSERVED)
		{ 15.f, 3.00f, 232.f, 2.33f, 2, 4 },  // Lv2: D+5, Speed+20%=232u/s
		{ 20.f, 3.00f, 232.f, 2.63f, 2, 4 },  // Lv3: D+5, Duration+0.3s
		{ 20.f, 3.00f, 232.f, 2.63f, 3, 5 },  // Lv4: Amount+1
		{ 25.f, 3.00f, 270.f, 2.63f, 3, 5 },  // Lv5: D+5, Speed+20%=270u/s (193×1.4)
		{ 30.f, 3.00f, 270.f, 2.93f, 3, 6 },  // Lv6: D+5, Duration+0.3s
		{ 30.f, 3.00f, 270.f, 2.93f, 4, 6 },  // Lv7: Amount+1
		{ 30.f, 3.00f, 270.f, 3.33f, 4, 7 },  // Lv8: Duration+0.4s
	};

	// NO FUTURE (Runetracer 進化): Damage=30, CD=1.0s, Speed=193×280%=540u/s, MaxBounce=7
	// 進化武器は MaxLevel=1 のため Lv1 のみ参照される
	inline constexpr FRunetracerParams NoFutureTable[MaxWeaponLevel] = {
		{ 30.f, 1.00f, 540.f, 3.00f, 1, 7 },  // Lv1: Speed=193×2.8=540u/s
		{ 30.f, 1.00f, 540.f, 3.00f, 1, 7 },  // Lv2 (未使用)
		{ 30.f, 1.00f, 540.f, 3.00f, 1, 7 },  // Lv3 (未使用)
		{ 30.f, 1.00f, 540.f, 3.00f, 1, 7 },  // Lv4 (未使用)
		{ 30.f, 1.00f, 540.f, 3.00f, 1, 7 },  // Lv5 (未使用)
		{ 30.f, 1.00f, 540.f, 3.00f, 1, 7 },  // Lv6 (未使用)
		{ 30.f, 1.00f, 540.f, 3.00f, 1, 7 },  // Lv7 (未使用)
		{ 30.f, 1.00f, 540.f, 3.00f, 1, 7 },  // Lv8 (未使用)
	};

	struct FLightningRingParams
	{
		float Damage;
		float Cooldown;
		float Radius;
		int32 Amount;
	};

	// LightningRing: CD=4.5s 全レベル固定。Radius: Area100%=30u
	inline constexpr FLightningRingParams LightningRingTable[MaxWeaponLevel] = {
		{ 15.f, 4.50f, 30.f, 2 },  // Lv1: D=15, Amount=2
		{ 15.f, 4.50f, 30.f, 3 },  // Lv2: Amount+1
		{ 25.f, 4.50f, 60.f, 3 },  // Lv3: D+10, Area+100%
		{ 25.f, 4.50f, 60.f, 4 },  // Lv4: Amount+1
		{ 45.f, 4.50f, 90.f, 4 },  // Lv5: D+20, Area+100%
		{ 45.f, 4.50f, 90.f, 5 },  // Lv6: Amount+1
		{ 65.f, 4.50f,120.f, 5 },  // Lv7: D+20, Area+100%
		{ 65.f, 4.50f,120.f, 6 },  // Lv8: Amount+1
	};

	// Thunder Loop (LightningRing 進化): Damage=65, CD=4.5s, Amount=6
	// 進化武器は MaxLevel=1 のため Lv1 のみ参照される
	inline constexpr FLightningRingParams ThunderLoopTable[MaxWeaponLevel] = {
		{ 65.f, 4.50f,120.f, 6 },  // Lv1
		{ 65.f, 4.50f,120.f, 6 },  // Lv2 (未使用)
		{ 65.f, 4.50f,120.f, 6 },  // Lv3 (未使用)
		{ 65.f, 4.50f,120.f, 6 },  // Lv4 (未使用)
		{ 65.f, 4.50f,120.f, 6 },  // Lv5 (未使用)
		{ 65.f, 4.50f,120.f, 6 },  // Lv6 (未使用)
		{ 65.f, 4.50f,120.f, 6 },  // Lv7 (未使用)
		{ 65.f, 4.50f,120.f, 6 },  // Lv8 (未使用)
	};

	struct FPentagramParams
	{
		float Damage;
		float Cooldown;
		float Radius;
	};

	// Pentagram: CD は Lv2,4 で -10s、Lv6,8 で -5s。Damage は全敵HP相当(999で近似)
	inline constexpr FPentagramParams PentagramTable[MaxWeaponLevel] = {
		{ 999.f, 90.0f, 9999.f },  // Lv1: CD=90s
		{ 999.f, 80.0f, 9999.f },  // Lv2: CD-10s
		{ 999.f, 80.0f, 9999.f },  // Lv3: Chance+25%
		{ 999.f, 70.0f, 9999.f },  // Lv4: CD-10s
		{ 999.f, 70.0f, 9999.f },  // Lv5: Chance+20%
		{ 999.f, 65.0f, 9999.f },  // Lv6: CD-5s
		{ 999.f, 65.0f, 9999.f },  // Lv7: Chance+20%
		{ 999.f, 60.0f, 9999.f },  // Lv8: CD-5s
	};

	// Gorgeous Moon (Pentagram 進化): CD=60s、Damage=999、Radius=9999
	// 進化武器は MaxLevel=1 のため Lv1 のみ参照される
	inline constexpr FPentagramParams GorgeousMoonTable[MaxWeaponLevel] = {
		{ 999.f, 60.0f, 9999.f },  // Lv1
		{ 999.f, 60.0f, 9999.f },  // Lv2 (未使用)
		{ 999.f, 60.0f, 9999.f },  // Lv3 (未使用)
		{ 999.f, 60.0f, 9999.f },  // Lv4 (未使用)
		{ 999.f, 60.0f, 9999.f },  // Lv5 (未使用)
		{ 999.f, 60.0f, 9999.f },  // Lv6 (未使用)
		{ 999.f, 60.0f, 9999.f },  // Lv7 (未使用)
		{ 999.f, 60.0f, 9999.f },  // Lv8 (未使用)
	};

	// Peachone 砲撃モデル定数（wiki由来）
	static constexpr float PeachoneProjectileInterval = 0.025f;  // OBSERVED: 砲撃の発射間隔（weapon_peachone.md）
	static constexpr int32 PeachoneSetsPerActivation   = 4;       // wiki: 1 activation あたり 4 set

	// 順次発射間隔（wiki・動画観察由来）
	static constexpr float CrossProjectileInterval      = 0.10f;  // wiki: 0.1s projectile interval (weapon_cross.md)
	static constexpr float AxeProjectileInterval        = 0.20f;  // OBSERVED: ~0.2s short volley (weapon_axe.md)
	static constexpr float RunetracerProjectileInterval = 0.20f;  // OBSERVED: ~0.2s sequence (weapon_runetracer.md)

	// Fire Wand ファン角度（動画測定値）
	// OBSERVED: fire_wand_bullet4.mp4 frame 290/300, 4発全体で約16°, 1 step ≈ 5.3°
	// acceptance range total 14-18°。
	static constexpr float FireWandAngleStepDeg = 5.3f;

	// Cross 折り返し往路距離（動画測定値）
	// OBSERVED: cross_bullet2.mp4 frame 80-110, 170-180px × 0.4167u/px ≈ 75u
	// acceptance range 60-90u。speed によらず距離を固定する。
	static constexpr float CrossReverseDistance = 75.f;

	// Runetracer Hitbox Delay（wiki: "same enemy cannot be hit more often than every 0.5s by the same rune"）
	static constexpr float RunetracerHitboxDelay = 0.5f;

	// Axe の Area スケール係数（wiki: "Axe scales with Area × 1.3"）
	static constexpr float AxeAreaScaleFactor = 1.3f;

	// Axe 上方向発射コーン: 真上（+Y）から左右 30° 以内のランダム方向（45°の2/3）
	static constexpr float AxeRandomConeHalfAngle = UE_PI / 6.f;  // 30° in radians

	// 画面半幅・半高（Camera Z=2000 基準、横幅 1920px=800u）
	// on-screen 判定に使用: |enemy.x - player.x| <= ScreenHalfWidthU かつ |y| <= ScreenHalfHeightU
	static constexpr float ScreenHalfWidthU  = 400.f;  // 800u/2
	static constexpr float ScreenHalfHeightU = 225.f;  // 450u/2 (16:9)

	// King Bible per-orb hit cooldown（wiki: "same bible hits same enemy no more than every 1.7s"）
	static constexpr float KingBibleOrbHitInterval = 1.7f;

	// Lightning Ring strike marker の寿命（visual/obs 用短寿命 GroundZone）
	static constexpr float LightningRingStrikeLifeTime = 0.15f;

	struct FPeachoneParams
	{
		float Damage;
		float Cooldown;
		float OrbitRadius;       // 軌道半径（固定、Area不使用）
		float OrbitRotSpeed;     // 軌道回転速度 rad/s（固定）
		float TargetZoneRadius;  // target zone 半径（固定、Area不使用）
		float ImpactRadius;      // 弾当たり半径（Area×Lv でスケール）
		int32 Amount;            // projectiles per set (wiki: Lv1=4, +1/level)
	};

	// Peachone: OBSERVED: peachone_bullet25.mp4
	//   OrbitRadius≈168u, OrbitRotSpeed≈0.8rad/s, TargetZoneRadius≈49u, ImpactRadius≈4.5u
	// CD: Lv4,7 で -0.3s。Amount: Lv1=4, +1/Lv。BaseArea(Lv2): ImpactRadius のみ +40%。
	// TargetZoneRadius と OrbitRadius は Area/Passive 不使用（fixed）。
	inline constexpr FPeachoneParams PeachoneTable[MaxWeaponLevel] = {
		{ 10.f, 1.0f, 168.f, 0.8f, 49.f, 4.5f,  4 },  // Lv1: OBSERVED values, Amount=4
		{ 10.f, 1.0f, 168.f, 0.8f, 49.f, 6.3f,  5 },  // Lv2: ImpactRadius×1.4(BaseArea+40%), Amount+1
		{ 20.f, 1.0f, 168.f, 0.8f, 49.f, 6.3f,  6 },  // Lv3: D+10, Amount+1
		{ 20.f, 0.7f, 168.f, 0.8f, 49.f, 6.3f,  7 },  // Lv4: CD-0.3, Amount+1
		{ 20.f, 0.7f, 168.f, 0.8f, 49.f, 8.8f,  8 },  // Lv5: ImpactRadius×1.4(BaseArea+40%), Amount+1
		{ 30.f, 0.7f, 168.f, 0.8f, 49.f, 8.8f,  9 },  // Lv6: D+10, Amount+1
		{ 30.f, 0.4f, 168.f, 0.8f, 49.f, 8.8f, 10 },  // Lv7: CD-0.3, Amount+1
		{ 30.f, 0.4f, 168.f, 0.8f, 49.f,12.3f, 11 },  // Lv8: ImpactRadius×1.4(BaseArea+40%), Amount+1
	};

	// EbonyWings: Peachone と同じパラメータ・逆回転・π初期位相
	// Vandalier: Peachone+EbonyWings 統合（同パラメータ使用）

	struct FLaurelParams
	{
		float ShieldDuration;  // 秒
		float Cooldown;        // 秒
	};

	// Laurel: 最大レベルは 7 (LaurelMaxLevel)。CD は Lv2,3,5,6 で -0.5s。
	// ShieldDuration は Lv2,3,5,6 で +0.2s (+0.8s 合計)。Lv4,7 で Charge+1
	inline constexpr FLaurelParams LaurelTable[MaxWeaponLevel] = {
		{ 1.0f, 10.0f },  // Lv1: CD=10.0, ShieldDur=1.0
		{ 1.2f,  9.5f },  // Lv2: CD-0.5, ShieldDur+0.2
		{ 1.4f,  9.0f },  // Lv3: CD-0.5, ShieldDur+0.2
		{ 1.4f,  9.0f },  // Lv4: Charge+1
		{ 1.6f,  8.5f },  // Lv5: CD-0.5, ShieldDur+0.2
		{ 1.8f,  8.0f },  // Lv6: CD-0.5, ShieldDur+0.2
		{ 1.8f,  8.0f },  // Lv7: Charge+1
		{ 1.8f,  8.0f },  // Lv8: (未使用 - MaxLevel=7)
	};

	// ---- パッシブアイテム定数 ------------------------------------------------

	// アイテム別最大レベルテーブル（EPassiveItemType のインデックスと対応）
	// None=0, Spinach=5, Armor=5, HollowHeart=5, Pummarola=5, EmptyTome=5,
	// Candelabrador=5, Bracer=5, Spellbinder=5, Duplicator=2, Wings=5,
	// Attractorb=5, Clover=5, Crown=5, StoneMask=0(coin対象外), SkullOManiac=5, Tirajisu=2, TorronasBox=9
	inline constexpr const int32 (&PassiveMaxLevel)[18] = SurvivorsWikiSpec::PassiveMaxLevel;

	inline constexpr const float (&AttractorbPickupRadiusMult)[5] = SurvivorsWikiSpec::AttractorbPickupRadiusMult;

	inline constexpr int32 GetWeaponMaxLevel(EWeaponType Type)
	{
		switch (Type)
		{
		case EWeaponType::None:
			return 0;
		case EWeaponType::Laurel:
			return SurvivorsWikiSpec::LaurelMaxLevel;
		case EWeaponType::SoulEater:
		case EWeaponType::BloodyTear:
		case EWeaponType::HolyWand:
		case EWeaponType::ThousandEdge:
		case EWeaponType::DeathSpiral:
		case EWeaponType::HeavenSword:
		case EWeaponType::UnholyVespers:
		case EWeaponType::Hellfire:
		case EWeaponType::LaBorra:
		case EWeaponType::NoFuture:
		case EWeaponType::ThunderLoop:
		case EWeaponType::GorgeousMoon:
		case EWeaponType::Vandalier:
			return SurvivorsWikiSpec::EvolvedWeaponMaxLevel;
		default:
			return MaxWeaponLevel;
		}
	}

	// ---- ジェム・基本定数 ---------------------------------------------------

	inline constexpr const float (&GemXPValues)[3] = SurvivorsWikiSpec::GemXPValues;
	static constexpr float BlueGemMaxXP                  = SurvivorsWikiSpec::BlueGemMaxXP;
	static constexpr float GreenGemMaxXP                 = SurvivorsWikiSpec::GreenGemMaxXP;
	static constexpr int32 RedGemMinMultiplier           = SurvivorsWikiSpec::RedGemMinMultiplier;
	static constexpr int32 RedGemMaxMultiplier           = SurvivorsWikiSpec::RedGemMaxMultiplier;

	inline constexpr EGemType GemTypeForExperience(float XP)
	{
		switch (SurvivorsWikiSpec::GemColorForExperience(XP))
		{
		case SurvivorsWikiSpec::EGemColor::Blue:
			return EGemType::Blue;
		case SurvivorsWikiSpec::EGemColor::Green:
			return EGemType::Green;
		case SurvivorsWikiSpec::EGemColor::Red:
		default:
			return EGemType::Red;
		}
	}

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

	// ---- 武器有効射程・カテゴリ関数 -----------------------------------------

	// 有効射程の正規化値（0.0〜1.0）。obs の weapon_attack_range_norm に使用する。
	// 値はゲームフィールド半径を基準とした相対値（0=接触系, 1=画面外まで届く）。
	inline constexpr float GetWeaponEffectiveRange(EWeaponType T)
	{
		switch (T)
		{
			case EWeaponType::Garlic:        return 0.0f;
			case EWeaponType::KingBible:     return 0.1f;
			case EWeaponType::Whip:          return 0.15f;
			case EWeaponType::MagicWand:     return 0.5f;
			case EWeaponType::FireWand:      return 0.5f;
			case EWeaponType::LightningRing: return 0.4f;
			case EWeaponType::Knife:         return 0.8f;
			case EWeaponType::Cross:         return 0.7f;
			case EWeaponType::Axe:           return 0.6f;
			case EWeaponType::SantaWater:    return 0.3f;
			case EWeaponType::Runetracer:    return 0.6f;
			default:                         return 0.5f;
		}
	}

	// 武器カテゴリ（整数）。obs の weapon_category_onehot に使用する。
	// 0=garlic_auto, 1=orbital, 2=melee_line, 3=ranged_targeted,
	// 4=ranged_directional, 5=area_drop, 6=defensive
	inline constexpr int32 GetWeaponCategory(EWeaponType T)
	{
		switch (T)
		{
			case EWeaponType::Garlic:
			case EWeaponType::SoulEater:      return 0; // garlic_auto
			case EWeaponType::KingBible:
			case EWeaponType::UnholyVespers:  return 1; // orbital
			case EWeaponType::Whip:
			case EWeaponType::BloodyTear:     return 2; // melee_line
			case EWeaponType::MagicWand:
			case EWeaponType::FireWand:
			case EWeaponType::LightningRing:
			case EWeaponType::HolyWand:
			case EWeaponType::ThunderLoop:    return 3; // ranged_targeted
			case EWeaponType::Knife:
			case EWeaponType::Axe:
			case EWeaponType::Cross:
			case EWeaponType::Peachone:
			case EWeaponType::EbonyWings:     return 4; // ranged_directional
			case EWeaponType::SantaWater:
			case EWeaponType::Runetracer:
			case EWeaponType::Hellfire:
			case EWeaponType::LaBorra:
			case EWeaponType::GorgeousMoon:   return 5; // area_drop
			default:                          return 6; // defensive
		}
	}

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
