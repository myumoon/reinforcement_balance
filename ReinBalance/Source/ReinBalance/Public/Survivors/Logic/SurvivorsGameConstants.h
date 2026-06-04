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

	// ---- 残り 14 武器パラメータ構造体・テーブル -----------------------------------

	struct FWhipParams
	{
		float Damage;
		float Cooldown;   // 秒
		float Width;      // シム座標（横幅半径）
		float Height;     // シム座標（縦幅半径）
	};

	inline constexpr FWhipParams WhipTable[MaxWeaponLevel] = {
		{ 10.f, 1.50f, 50.f, 15.f },  // Lv1
		{ 15.f, 1.50f, 50.f, 15.f },  // Lv2
		{ 15.f, 1.20f, 60.f, 15.f },  // Lv3
		{ 20.f, 1.20f, 60.f, 15.f },  // Lv4
		{ 20.f, 1.00f, 70.f, 15.f },  // Lv5
		{ 25.f, 1.00f, 70.f, 20.f },  // Lv6
		{ 25.f, 0.85f, 80.f, 20.f },  // Lv7
		{ 30.f, 0.85f, 80.f, 25.f },  // Lv8
	};

	// BloodyTear (Whip 進化): Whip Lv4 相当から開始、約 1.5〜2 倍強化
	inline constexpr FWhipParams BloodyTearTable[MaxWeaponLevel] = {
		{ 30.f, 1.00f, 80.f, 20.f },  // Lv1
		{ 35.f, 0.95f, 85.f, 22.f },  // Lv2
		{ 40.f, 0.90f, 90.f, 24.f },  // Lv3
		{ 45.f, 0.85f, 95.f, 26.f },  // Lv4
		{ 50.f, 0.80f,100.f, 28.f },  // Lv5
		{ 55.f, 0.75f,105.f, 30.f },  // Lv6
		{ 60.f, 0.70f,110.f, 32.f },  // Lv7
		{ 65.f, 0.65f,120.f, 35.f },  // Lv8
	};

	struct FMagicWandParams
	{
		float Damage;
		float Cooldown;
		float Speed;
		int32 Amount;
	};

	inline constexpr FMagicWandParams MagicWandTable[MaxWeaponLevel] = {
		{ 20.f, 0.50f, 300.f, 1 },  // Lv1
		{ 25.f, 0.50f, 300.f, 1 },  // Lv2
		{ 25.f, 0.40f, 320.f, 1 },  // Lv3
		{ 30.f, 0.40f, 320.f, 2 },  // Lv4
		{ 35.f, 0.35f, 340.f, 2 },  // Lv5
		{ 35.f, 0.30f, 360.f, 2 },  // Lv6
		{ 40.f, 0.30f, 360.f, 3 },  // Lv7
		{ 45.f, 0.25f, 380.f, 3 },  // Lv8
	};

	inline constexpr FMagicWandParams HolyWandTable[MaxWeaponLevel] = {
		{ 40.f, 0.40f, 360.f, 2 },  // Lv1
		{ 45.f, 0.38f, 370.f, 2 },  // Lv2
		{ 50.f, 0.35f, 380.f, 3 },  // Lv3
		{ 55.f, 0.32f, 390.f, 3 },  // Lv4
		{ 60.f, 0.30f, 400.f, 3 },  // Lv5
		{ 65.f, 0.28f, 410.f, 4 },  // Lv6
		{ 70.f, 0.25f, 420.f, 4 },  // Lv7
		{ 80.f, 0.22f, 440.f, 5 },  // Lv8
	};

	struct FKnifeParams
	{
		float Damage;
		float Cooldown;
		float Speed;
		int32 Amount;
	};

	inline constexpr FKnifeParams KnifeTable[MaxWeaponLevel] = {
		{ 15.f, 0.35f, 400.f, 1 },  // Lv1
		{ 15.f, 0.30f, 420.f, 1 },  // Lv2
		{ 20.f, 0.30f, 420.f, 2 },  // Lv3
		{ 20.f, 0.25f, 440.f, 2 },  // Lv4
		{ 25.f, 0.25f, 440.f, 3 },  // Lv5
		{ 25.f, 0.20f, 460.f, 3 },  // Lv6
		{ 30.f, 0.20f, 460.f, 4 },  // Lv7
		{ 35.f, 0.18f, 480.f, 4 },  // Lv8
	};

	inline constexpr FKnifeParams ThousandEdgeTable[MaxWeaponLevel] = {
		{ 30.f, 0.18f, 480.f, 3 },  // Lv1
		{ 35.f, 0.17f, 500.f, 4 },  // Lv2
		{ 40.f, 0.16f, 510.f, 4 },  // Lv3
		{ 45.f, 0.15f, 520.f, 5 },  // Lv4
		{ 50.f, 0.14f, 530.f, 5 },  // Lv5
		{ 55.f, 0.13f, 540.f, 6 },  // Lv6
		{ 60.f, 0.12f, 550.f, 6 },  // Lv7
		{ 70.f, 0.10f, 580.f, 7 },  // Lv8
	};

	struct FAxeParams
	{
		float Damage;
		float Cooldown;
		float Speed;
		float ArcHeight;  // 最大弧高さ（シム座標）
	};

	inline constexpr FAxeParams AxeTable[MaxWeaponLevel] = {
		{  50.f, 1.20f, 180.f, 120.f },  // Lv1
		{  60.f, 1.20f, 190.f, 130.f },  // Lv2
		{  60.f, 1.00f, 200.f, 140.f },  // Lv3
		{  70.f, 1.00f, 200.f, 140.f },  // Lv4
		{  80.f, 0.90f, 210.f, 150.f },  // Lv5
		{  90.f, 0.85f, 220.f, 160.f },  // Lv6
		{ 100.f, 0.85f, 230.f, 170.f },  // Lv7
		{ 110.f, 0.75f, 240.f, 180.f },  // Lv8
	};

	inline constexpr FAxeParams DeathSpiralTable[MaxWeaponLevel] = {
		{ 100.f, 0.70f, 260.f, 180.f },  // Lv1
		{ 110.f, 0.65f, 270.f, 190.f },  // Lv2
		{ 120.f, 0.62f, 280.f, 200.f },  // Lv3
		{ 130.f, 0.60f, 290.f, 210.f },  // Lv4
		{ 140.f, 0.58f, 300.f, 220.f },  // Lv5
		{ 150.f, 0.55f, 310.f, 230.f },  // Lv6
		{ 165.f, 0.52f, 320.f, 240.f },  // Lv7
		{ 180.f, 0.50f, 340.f, 260.f },  // Lv8
	};

	struct FCrossParams
	{
		float Damage;
		float Cooldown;
		float Speed;
	};

	inline constexpr FCrossParams CrossTable[MaxWeaponLevel] = {
		{  50.f, 1.50f, 160.f },  // Lv1
		{  60.f, 1.40f, 170.f },  // Lv2
		{  70.f, 1.30f, 180.f },  // Lv3
		{  80.f, 1.20f, 190.f },  // Lv4
		{  90.f, 1.10f, 200.f },  // Lv5
		{ 100.f, 1.00f, 210.f },  // Lv6
		{ 110.f, 0.90f, 220.f },  // Lv7
		{ 120.f, 0.80f, 230.f },  // Lv8
	};

	inline constexpr FCrossParams HeavenSwordTable[MaxWeaponLevel] = {
		{ 120.f, 0.80f, 230.f },  // Lv1
		{ 135.f, 0.75f, 240.f },  // Lv2
		{ 150.f, 0.72f, 250.f },  // Lv3
		{ 165.f, 0.68f, 260.f },  // Lv4
		{ 180.f, 0.65f, 270.f },  // Lv5
		{ 195.f, 0.62f, 280.f },  // Lv6
		{ 210.f, 0.58f, 290.f },  // Lv7
		{ 240.f, 0.55f, 310.f },  // Lv8
	};

	struct FKingBibleParams
	{
		float Damage;
		float Cooldown;    // 0 = 常時ヒット（HitInterval で制御）
		float OrbitRadius;
		int32 Amount;
		float RotSpeed;    // rad/sec
	};

	inline constexpr FKingBibleParams KingBibleTable[MaxWeaponLevel] = {
		{ 10.f, 0.f, 50.f, 1, 2.0f },  // Lv1
		{ 12.f, 0.f, 55.f, 1, 2.0f },  // Lv2
		{ 12.f, 0.f, 55.f, 2, 2.2f },  // Lv3
		{ 14.f, 0.f, 60.f, 2, 2.2f },  // Lv4
		{ 14.f, 0.f, 60.f, 3, 2.4f },  // Lv5
		{ 16.f, 0.f, 65.f, 3, 2.4f },  // Lv6
		{ 16.f, 0.f, 65.f, 4, 2.6f },  // Lv7
		{ 20.f, 0.f, 70.f, 4, 3.0f },  // Lv8
	};

	inline constexpr FKingBibleParams UnholyVespersTable[MaxWeaponLevel] = {
		{ 20.f, 0.f, 70.f, 3, 3.0f },  // Lv1
		{ 22.f, 0.f, 75.f, 4, 3.0f },  // Lv2
		{ 24.f, 0.f, 75.f, 4, 3.2f },  // Lv3
		{ 26.f, 0.f, 80.f, 5, 3.2f },  // Lv4
		{ 28.f, 0.f, 80.f, 5, 3.4f },  // Lv5
		{ 30.f, 0.f, 85.f, 6, 3.4f },  // Lv6
		{ 32.f, 0.f, 85.f, 6, 3.6f },  // Lv7
		{ 36.f, 0.f, 90.f, 7, 4.0f },  // Lv8
	};

	struct FFireWandParams
	{
		float Damage;
		float Cooldown;
		float Speed;
		float ExplosionRadius;
	};

	inline constexpr FFireWandParams FireWandTable[MaxWeaponLevel] = {
		{ 40.f, 1.40f, 180.f, 30.f },  // Lv1
		{ 50.f, 1.30f, 190.f, 35.f },  // Lv2
		{ 50.f, 1.20f, 200.f, 35.f },  // Lv3
		{ 60.f, 1.10f, 210.f, 40.f },  // Lv4
		{ 60.f, 1.00f, 220.f, 45.f },  // Lv5
		{ 70.f, 0.90f, 230.f, 50.f },  // Lv6
		{ 80.f, 0.85f, 240.f, 55.f },  // Lv7
		{ 90.f, 0.75f, 250.f, 60.f },  // Lv8
	};

	inline constexpr FFireWandParams HellfireTable[MaxWeaponLevel] = {
		{  90.f, 0.70f, 260.f,  60.f },  // Lv1
		{ 100.f, 0.65f, 270.f,  65.f },  // Lv2
		{ 110.f, 0.62f, 280.f,  70.f },  // Lv3
		{ 120.f, 0.58f, 290.f,  75.f },  // Lv4
		{ 130.f, 0.55f, 300.f,  80.f },  // Lv5
		{ 140.f, 0.52f, 310.f,  85.f },  // Lv6
		{ 155.f, 0.50f, 320.f,  90.f },  // Lv7
		{ 170.f, 0.45f, 340.f, 100.f },  // Lv8
	};

	struct FSantaWaterParams
	{
		float Damage;
		float Cooldown;
		float Radius;
		float Duration;
		int32 Amount;
	};

	inline constexpr FSantaWaterParams SantaWaterTable[MaxWeaponLevel] = {
		{ 10.f, 2.00f, 30.f, 3.0f, 1 },  // Lv1
		{ 12.f, 1.80f, 35.f, 3.5f, 1 },  // Lv2
		{ 12.f, 1.80f, 35.f, 3.5f, 2 },  // Lv3
		{ 15.f, 1.60f, 40.f, 4.0f, 2 },  // Lv4
		{ 15.f, 1.50f, 45.f, 4.0f, 3 },  // Lv5
		{ 18.f, 1.30f, 45.f, 4.5f, 3 },  // Lv6
		{ 18.f, 1.20f, 50.f, 4.5f, 4 },  // Lv7
		{ 20.f, 1.00f, 55.f, 5.0f, 4 },  // Lv8
	};

	inline constexpr FSantaWaterParams LaBorraTable[MaxWeaponLevel] = {
		{ 20.f, 0.90f, 55.f, 5.0f, 3 },  // Lv1
		{ 22.f, 0.85f, 60.f, 5.5f, 3 },  // Lv2
		{ 24.f, 0.82f, 60.f, 5.5f, 4 },  // Lv3
		{ 26.f, 0.78f, 65.f, 6.0f, 4 },  // Lv4
		{ 28.f, 0.75f, 70.f, 6.5f, 4 },  // Lv5
		{ 30.f, 0.70f, 75.f, 7.0f, 5 },  // Lv6
		{ 32.f, 0.65f, 75.f, 7.0f, 5 },  // Lv7
		{ 36.f, 0.60f, 80.f, 8.0f, 6 },  // Lv8
	};

	struct FRunetracerParams
	{
		float Damage;
		float Cooldown;
		float Speed;
		int32 MaxBounce;
	};

	inline constexpr FRunetracerParams RunetracerTable[MaxWeaponLevel] = {
		{ 15.f, 0.60f, 220.f, 3 },  // Lv1
		{ 18.f, 0.55f, 230.f, 4 },  // Lv2
		{ 20.f, 0.50f, 240.f, 4 },  // Lv3
		{ 22.f, 0.45f, 250.f, 5 },  // Lv4
		{ 25.f, 0.40f, 260.f, 5 },  // Lv5
		{ 28.f, 0.38f, 270.f, 6 },  // Lv6
		{ 30.f, 0.35f, 280.f, 6 },  // Lv7
		{ 35.f, 0.30f, 290.f, 7 },  // Lv8
	};

	inline constexpr FRunetracerParams NoFutureTable[MaxWeaponLevel] = {
		{ 35.f, 0.28f, 300.f,  7 },  // Lv1
		{ 40.f, 0.26f, 310.f,  8 },  // Lv2
		{ 45.f, 0.24f, 320.f,  8 },  // Lv3
		{ 50.f, 0.22f, 330.f,  9 },  // Lv4
		{ 55.f, 0.20f, 340.f,  9 },  // Lv5
		{ 60.f, 0.18f, 350.f, 10 },  // Lv6
		{ 65.f, 0.16f, 360.f, 10 },  // Lv7
		{ 72.f, 0.14f, 380.f, 12 },  // Lv8
	};

	struct FLightningRingParams
	{
		float Damage;
		float Cooldown;
		int32 Amount;
	};

	inline constexpr FLightningRingParams LightningRingTable[MaxWeaponLevel] = {
		{  40.f, 1.00f, 1 },  // Lv1
		{  50.f, 0.90f, 1 },  // Lv2
		{  50.f, 0.80f, 2 },  // Lv3
		{  60.f, 0.75f, 2 },  // Lv4
		{  70.f, 0.70f, 3 },  // Lv5
		{  80.f, 0.65f, 3 },  // Lv6
		{  90.f, 0.60f, 4 },  // Lv7
		{ 100.f, 0.50f, 4 },  // Lv8
	};

	inline constexpr FLightningRingParams ThunderLoopTable[MaxWeaponLevel] = {
		{ 100.f, 0.45f, 4 },  // Lv1
		{ 110.f, 0.42f, 5 },  // Lv2
		{ 120.f, 0.40f, 5 },  // Lv3
		{ 130.f, 0.38f, 6 },  // Lv4
		{ 140.f, 0.35f, 6 },  // Lv5
		{ 150.f, 0.32f, 7 },  // Lv6
		{ 165.f, 0.30f, 7 },  // Lv7
		{ 180.f, 0.25f, 8 },  // Lv8
	};

	struct FPentagramParams
	{
		float Damage;
		float Cooldown;
		float Radius;
	};

	inline constexpr FPentagramParams PentagramTable[MaxWeaponLevel] = {
		{ 999.f, 15.0f, 9999.f },  // Lv1
		{ 999.f, 14.0f, 9999.f },  // Lv2
		{ 999.f, 13.0f, 9999.f },  // Lv3
		{ 999.f, 12.0f, 9999.f },  // Lv4
		{ 999.f, 11.0f, 9999.f },  // Lv5
		{ 999.f, 10.0f, 9999.f },  // Lv6
		{ 999.f,  9.0f, 9999.f },  // Lv7
		{ 999.f,  8.0f, 9999.f },  // Lv8
	};

	inline constexpr FPentagramParams GorgeousMoonTable[MaxWeaponLevel] = {
		{ 999.f, 12.0f, 9999.f },  // Lv1
		{ 999.f, 11.0f, 9999.f },  // Lv2
		{ 999.f, 10.0f, 9999.f },  // Lv3
		{ 999.f,  9.5f, 9999.f },  // Lv4
		{ 999.f,  9.0f, 9999.f },  // Lv5
		{ 999.f,  8.5f, 9999.f },  // Lv6
		{ 999.f,  8.0f, 9999.f },  // Lv7
		{ 999.f,  7.0f, 9999.f },  // Lv8
	};

	struct FPeachoneParams
	{
		float Damage;
		float Cooldown;
		float OrbitRadius;
		float BombRadius;
	};

	inline constexpr FPeachoneParams PeachoneTable[MaxWeaponLevel] = {
		{ 25.f, 3.0f, 60.f, 30.f },  // Lv1
		{ 30.f, 2.8f, 65.f, 32.f },  // Lv2
		{ 30.f, 2.6f, 65.f, 35.f },  // Lv3
		{ 35.f, 2.4f, 70.f, 38.f },  // Lv4
		{ 40.f, 2.2f, 75.f, 40.f },  // Lv5
		{ 45.f, 2.0f, 75.f, 42.f },  // Lv6
		{ 50.f, 1.8f, 80.f, 45.f },  // Lv7
		{ 55.f, 1.6f, 80.f, 50.f },  // Lv8
	};

	// EbonyWings: Peachone と同じパラメータ・逆回転
	// Vandalier: Peachone+EbonyWings 統合（同パラメータ使用）

	struct FLaurelParams
	{
		float ShieldDuration;  // 秒
		float Cooldown;        // 秒
	};

	inline constexpr FLaurelParams LaurelTable[MaxWeaponLevel] = {
		{ 1.0f, 8.0f },  // Lv1
		{ 1.5f, 7.5f },  // Lv2
		{ 2.0f, 7.0f },  // Lv3
		{ 2.5f, 6.5f },  // Lv4
		{ 3.0f, 6.0f },  // Lv5
		{ 3.5f, 5.5f },  // Lv6
		{ 4.0f, 5.0f },  // Lv7
		{ 5.0f, 4.0f },  // Lv8
	};

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
