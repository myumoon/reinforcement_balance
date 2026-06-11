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
	static constexpr float SantaWaterWarningTime        = 1.00f;

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

	// Whip: Area100%=50u。CD は全レベル固定 1.35s。Lv2 で Amount+1
	inline constexpr FWhipParams WhipTable[MaxWeaponLevel] = {
		{ 10.f, 1.35f, 50.f, 15.f, 1 },  // Lv1: D=10, Area=100%, Amount=1
		{ 10.f, 1.35f, 50.f, 15.f, 2 },  // Lv2: Amount+1
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

	// MagicWand: Speed は全レベル固定 600u(100%)。Lv3 で CD-0.2、各 Lv で Amount/Pierce 増加
	inline constexpr FMagicWandParams MagicWandTable[MaxWeaponLevel] = {
		{ 10.f, 1.20f, 600.f, 1, 1 },  // Lv1: D=10, CD=1.20, Amount=1, Pierce=1
		{ 10.f, 1.20f, 600.f, 2, 1 },  // Lv2: Amount+1
		{ 10.f, 1.00f, 600.f, 2, 1 },  // Lv3: CD-0.20
		{ 10.f, 1.00f, 600.f, 3, 1 },  // Lv4: Amount+1
		{ 20.f, 1.00f, 600.f, 3, 1 },  // Lv5: D+10
		{ 20.f, 1.00f, 600.f, 4, 1 },  // Lv6: Amount+1
		{ 20.f, 1.00f, 600.f, 4, 2 },  // Lv7: Pierce+1
		{ 30.f, 1.00f, 600.f, 4, 2 },  // Lv8: D+10
	};

	// HolyWand (MagicWand 進化): Damage=30, CD=0.5s, Speed=1200u(200%), Amount=4, Pierce=2
	// 進化武器は MaxLevel=1 のため Lv1 のみ参照される
	inline constexpr FMagicWandParams HolyWandTable[MaxWeaponLevel] = {
		{ 30.f, 0.50f, 1200.f, 4, 2 },  // Lv1
		{ 30.f, 0.50f, 1200.f, 4, 2 },  // Lv2 (未使用)
		{ 30.f, 0.50f, 1200.f, 4, 2 },  // Lv3 (未使用)
		{ 30.f, 0.50f, 1200.f, 4, 2 },  // Lv4 (未使用)
		{ 30.f, 0.50f, 1200.f, 4, 2 },  // Lv5 (未使用)
		{ 30.f, 0.50f, 1200.f, 4, 2 },  // Lv6 (未使用)
		{ 30.f, 0.50f, 1200.f, 4, 2 },  // Lv7 (未使用)
		{ 30.f, 0.50f, 1200.f, 4, 2 },  // Lv8 (未使用)
	};

	struct FKnifeParams
	{
		float Damage;
		float Cooldown;
		float Speed;
		int32 Amount;
		int32 Pierce;  // 貫通数: 1=単体命中, 2=2体貫通, 3=3体貫通
	};

	// Knife: CD=1.0s 全レベル固定、Speed=800u(100%) 全レベル固定
	// Pierce: Lv1=1体, Lv5=2体, Lv8=3体
	inline constexpr FKnifeParams KnifeTable[MaxWeaponLevel] = {
		{  6.5f, 1.00f, 800.f, 1, 1 },  // Lv1: D=6.5, Amount=1, Pierce=1
		{  6.5f, 1.00f, 800.f, 2, 1 },  // Lv2: Amount+1
		{ 11.5f, 1.00f, 800.f, 3, 1 },  // Lv3: D+5, Amount+1
		{ 11.5f, 1.00f, 800.f, 4, 1 },  // Lv4: Amount+1
		{ 11.5f, 1.00f, 800.f, 4, 2 },  // Lv5: Pierce+1
		{ 11.5f, 1.00f, 800.f, 5, 2 },  // Lv6: Amount+1
		{ 16.5f, 1.00f, 800.f, 6, 2 },  // Lv7: D+5, Amount+1
		{ 16.5f, 1.00f, 800.f, 6, 3 },  // Lv8: Pierce+1
	};

	// ThousandEdge (Knife 進化): Damage=16.5, CD=0.35s, Speed=1200u(150%), Amount=6, Pierce=3
	// 進化武器は MaxLevel=1 のため Lv1 のみ参照される
	inline constexpr FKnifeParams ThousandEdgeTable[MaxWeaponLevel] = {
		{ 16.5f, 0.35f, 1200.f, 6, 3 },  // Lv1
		{ 16.5f, 0.35f, 1200.f, 6, 3 },  // Lv2 (未使用)
		{ 16.5f, 0.35f, 1200.f, 6, 3 },  // Lv3 (未使用)
		{ 16.5f, 0.35f, 1200.f, 6, 3 },  // Lv4 (未使用)
		{ 16.5f, 0.35f, 1200.f, 6, 3 },  // Lv5 (未使用)
		{ 16.5f, 0.35f, 1200.f, 6, 3 },  // Lv6 (未使用)
		{ 16.5f, 0.35f, 1200.f, 6, 3 },  // Lv7 (未使用)
		{ 16.5f, 0.35f, 1200.f, 6, 3 },  // Lv8 (未使用)
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
	// Amount: Lv1=1,Lv2=2,Lv5=3。Pierce: Lv1=3体,Lv4=5体,Lv7=7体
	inline constexpr FAxeParams AxeTable[MaxWeaponLevel] = {
		{  20.f, 4.00f, 360.f, 120.f, 1, 3 },  // Lv1: D=20, Pierce=3
		{  20.f, 4.00f, 360.f, 120.f, 2, 3 },  // Lv2: Amount+1
		{  40.f, 4.00f, 360.f, 120.f, 2, 3 },  // Lv3: D+20
		{  40.f, 4.00f, 360.f, 120.f, 2, 5 },  // Lv4: Pierce+2
		{  40.f, 4.00f, 360.f, 120.f, 3, 5 },  // Lv5: Amount+1
		{  60.f, 4.00f, 360.f, 120.f, 3, 5 },  // Lv6: D+20
		{  60.f, 4.00f, 360.f, 120.f, 3, 7 },  // Lv7: Pierce+2
		{  80.f, 4.00f, 360.f, 120.f, 3, 7 },  // Lv8: D+20
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
	inline constexpr FCrossParams CrossTable[MaxWeaponLevel] = {
		{  5.f, 2.00f, 320.f, 12.0f, 1, KnockbackSim_1 },  // Lv1: D=5,  Speed=100%
		{ 15.f, 2.00f, 320.f, 12.0f, 1, KnockbackSim_1 },  // Lv2: D+10
		{ 15.f, 2.00f, 400.f, 13.2f, 1, KnockbackSim_1 },  // Lv3: Speed+25%, Area+10%
		{ 15.f, 2.00f, 400.f, 13.2f, 2, KnockbackSim_1 },  // Lv4: Amount+1
		{ 25.f, 2.00f, 400.f, 13.2f, 2, KnockbackSim_1 },  // Lv5: D+10
		{ 25.f, 2.00f, 480.f, 14.4f, 2, KnockbackSim_1 },  // Lv6: Speed+25%, Area+10%
		{ 25.f, 2.00f, 480.f, 14.4f, 3, KnockbackSim_1 },  // Lv7: Amount+1
		{ 35.f, 2.00f, 480.f, 14.4f, 3, KnockbackSim_1 },  // Lv8: D+10
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
	inline constexpr FKingBibleParams KingBibleTable[MaxWeaponLevel] = {
		{ 10.f, 3.f, 3.0f, 50.0f, 1, 4.0f, KnockbackSim_1 },  // Lv1: D=10, OR=100%, RS=100%, Amount=1
		{ 10.f, 3.f, 3.0f, 50.0f, 2, 4.0f, KnockbackSim_1 },  // Lv2: Amount+1
		{ 10.f, 3.f, 3.0f, 62.5f, 2, 5.2f, KnockbackSim_1 },  // Lv3: Area+25%, Speed+30%
		{ 20.f, 3.f, 3.5f, 62.5f, 2, 5.2f, KnockbackSim_1 },  // Lv4: D+10, Duration+0.5s
		{ 20.f, 3.f, 3.5f, 62.5f, 3, 5.2f, KnockbackSim_1 },  // Lv5: Amount+1
		{ 20.f, 3.f, 3.5f, 75.0f, 3, 6.4f, KnockbackSim_1 },  // Lv6: Area+25%, Speed+30%
		{ 30.f, 3.f, 4.0f, 75.0f, 3, 6.4f, KnockbackSim_1 },  // Lv7: D+10, Duration+0.5s
		{ 30.f, 3.f, 4.0f, 75.0f, 4, 6.4f, KnockbackSim_1 },  // Lv8: Amount+1
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

	// FireWand: CD=3.0s 全レベル固定。Speed: 75%=360u(基準100%=480u)。Lv1 から4発の扇状発射。
	// Speed: Lv1=75%,Lv3=95%,Lv5=115%,Lv7=135%。ExplosionRadius は全レベル固定
	inline constexpr FFireWandParams FireWandTable[MaxWeaponLevel] = {
		{ 20.f, 3.00f, 360.f, 30.f, 4 },  // Lv1: D=20, Speed=75%
		{ 30.f, 3.00f, 360.f, 30.f, 4 },  // Lv2: D+10
		{ 40.f, 3.00f, 456.f, 30.f, 4 },  // Lv3: D+10, Speed+20%(95%)
		{ 50.f, 3.00f, 456.f, 30.f, 4 },  // Lv4: D+10
		{ 60.f, 3.00f, 552.f, 30.f, 4 },  // Lv5: D+10, Speed+20%(115%)
		{ 70.f, 3.00f, 552.f, 30.f, 4 },  // Lv6: D+10
		{ 80.f, 3.00f, 648.f, 30.f, 4 },  // Lv7: D+10, Speed+20%(135%)
		{ 90.f, 3.00f, 648.f, 30.f, 4 },  // Lv8: D+10
	};

	// Hellfire (FireWand 進化): Damage=100, CD=3.0s, Speed=480u(100%), ExpRad=30u
	// 進化武器は MaxLevel=1 のため Lv1 のみ参照される
	inline constexpr FFireWandParams HellfireTable[MaxWeaponLevel] = {
		{ 100.f, 3.00f, 480.f, 30.f, 4 },  // Lv1
		{ 100.f, 3.00f, 480.f, 30.f, 4 },  // Lv2 (未使用)
		{ 100.f, 3.00f, 480.f, 30.f, 4 },  // Lv3 (未使用)
		{ 100.f, 3.00f, 480.f, 30.f, 4 },  // Lv4 (未使用)
		{ 100.f, 3.00f, 480.f, 30.f, 4 },  // Lv5 (未使用)
		{ 100.f, 3.00f, 480.f, 30.f, 4 },  // Lv6 (未使用)
		{ 100.f, 3.00f, 480.f, 30.f, 4 },  // Lv7 (未使用)
		{ 100.f, 3.00f, 480.f, 30.f, 4 },  // Lv8 (未使用)
	};

	struct FSantaWaterParams
	{
		float Damage;
		float Cooldown;
		float Radius;
		float Duration;
		int32 Amount;
	};

	// SantaWater: CD=4.5s 全レベル固定。Radius: 100%=30u。Duration は Lv3,5,7 で増加
	inline constexpr FSantaWaterParams SantaWaterTable[MaxWeaponLevel] = {
		{ 10.f, 4.50f, 30.f, 2.00f, 1 },  // Lv1: D=10, R=100%, Dur=2.0, Amount=1
		{ 10.f, 4.50f, 36.f, 2.00f, 2 },  // Lv2: R+20%(120%), Amount+1
		{ 20.f, 4.50f, 36.f, 2.50f, 2 },  // Lv3: D+10, Dur+0.50
		{ 20.f, 4.50f, 42.f, 2.50f, 3 },  // Lv4: R+20%(140%), Amount+1
		{ 30.f, 4.50f, 42.f, 2.75f, 3 },  // Lv5: D+10, Dur+0.25
		{ 30.f, 4.50f, 48.f, 2.75f, 4 },  // Lv6: R+20%(160%), Amount+1
		{ 35.f, 4.50f, 48.f, 3.00f, 4 },  // Lv7: D+5,  Dur+0.25
		{ 40.f, 4.50f, 54.f, 3.00f, 4 },  // Lv8: D+5,  R+20%(180%)
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

	// Runetracer: CD=3.0s 全レベル固定。Speed: 100%=440u,120%=528u,140%=616u
	// Duration/Amount は仕様値を使用。MaxBounce は既存の壁反射寿命近似。
	inline constexpr FRunetracerParams RunetracerTable[MaxWeaponLevel] = {
		{ 10.f, 3.00f, 440.f, 2.25f, 1, 3 },  // Lv1: D=10, Speed=100%
		{ 15.f, 3.00f, 528.f, 2.25f, 1, 4 },  // Lv2: D+5, Speed+20%
		{ 20.f, 3.00f, 528.f, 2.55f, 1, 4 },  // Lv3: D+5, Duration+0.3s
		{ 20.f, 3.00f, 528.f, 2.55f, 2, 5 },  // Lv4: Amount+1
		{ 25.f, 3.00f, 616.f, 2.55f, 2, 5 },  // Lv5: D+5, Speed+20%
		{ 30.f, 3.00f, 616.f, 2.85f, 2, 6 },  // Lv6: D+5, Duration+0.3s
		{ 30.f, 3.00f, 616.f, 2.85f, 3, 6 },  // Lv7: Amount+1
		{ 30.f, 3.00f, 616.f, 3.25f, 3, 7 },  // Lv8: Duration+0.4s
	};

	// NO FUTURE (Runetracer 進化): Damage=30, CD=1.0s, Speed=1232u(280%), MaxBounce=7
	// 進化武器は MaxLevel=1 のため Lv1 のみ参照される
	inline constexpr FRunetracerParams NoFutureTable[MaxWeaponLevel] = {
		{ 30.f, 1.00f, 1232.f, 3.00f, 1, 7 },  // Lv1
		{ 30.f, 1.00f, 1232.f, 3.00f, 1, 7 },  // Lv2 (未使用)
		{ 30.f, 1.00f, 1232.f, 3.00f, 1, 7 },  // Lv3 (未使用)
		{ 30.f, 1.00f, 1232.f, 3.00f, 1, 7 },  // Lv4 (未使用)
		{ 30.f, 1.00f, 1232.f, 3.00f, 1, 7 },  // Lv5 (未使用)
		{ 30.f, 1.00f, 1232.f, 3.00f, 1, 7 },  // Lv6 (未使用)
		{ 30.f, 1.00f, 1232.f, 3.00f, 1, 7 },  // Lv7 (未使用)
		{ 30.f, 1.00f, 1232.f, 3.00f, 1, 7 },  // Lv8 (未使用)
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

	struct FPeachoneParams
	{
		float Damage;
		float Cooldown;
		float OrbitRadius;
		float BombRadius;
	};

	// Peachone: OrbitRadius=60u 全レベル固定。BombRadius: Area100%=30u,140%=42u,180%=54u,220%=66u
	// CD: Lv4,7 で -0.3s。Amount はコード側で管理
	inline constexpr FPeachoneParams PeachoneTable[MaxWeaponLevel] = {
		{ 10.f, 1.0f, 60.f, 30.f },  // Lv1: D=10, CD=1.0, Area=100%
		{ 10.f, 1.0f, 60.f, 42.f },  // Lv2: Area+40%
		{ 20.f, 1.0f, 60.f, 42.f },  // Lv3: D+10
		{ 20.f, 0.7f, 60.f, 42.f },  // Lv4: CD-0.3
		{ 20.f, 0.7f, 60.f, 54.f },  // Lv5: Area+40%
		{ 30.f, 0.7f, 60.f, 54.f },  // Lv6: D+10
		{ 30.f, 0.4f, 60.f, 54.f },  // Lv7: CD-0.3
		{ 30.f, 0.4f, 60.f, 66.f },  // Lv8: Area+40%
	};

	// EbonyWings: Peachone と同じパラメータ・逆回転
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
