#pragma once

#include "CoreMinimal.h"
#include "SurvivorsTypes.generated.h"

struct FSurvivorsObsSegment
{
	FString Name;
	int32 Dim;
};

// ---- 前方宣言が必要な既存強型（後続の構造体で参照する） ---------------------

struct FDamage
{
	float Value = 0.f;
	explicit FDamage(float InValue = 0.f) : Value(InValue) {}
};

struct FSurvivorsElapsedTime
{
	float Seconds = 0.f;
	explicit FSurvivorsElapsedTime(float InSeconds = 0.f) : Seconds(InSeconds) {}
};

// ---- 新規強型定義 -------------------------------------------------------------

// 武器レベル（1〜8）── FPlayerLevel と混同しないために分離
struct FWeaponLevel
{
	int32 Value = 1;
	explicit FWeaponLevel(int32 InValue = 1) : Value(InValue) {}
	// IsMax は SurvivorsGameConstants::MaxWeaponLevel(8) を参照するため
	// inline で実装。循環参照を避けるため直値を使用。
	bool IsMax() const { return Value >= 8; }
};

// プレイヤーレベル（1〜MaxPlayerLevel）
struct FPlayerLevel
{
	int32 Value = 1;
	explicit FPlayerLevel(int32 InValue = 1) : Value(InValue) {}
};

// シム空間の半径（u）── UE ワールド半径（cm）と混同しないために分離
struct FSimRadius
{
	float Value = 0.f;
	explicit FSimRadius(float InValue = 0.f) : Value(InValue) {}
};

// クールダウン残時間（秒）── FSurvivorsElapsedTime と区別
struct FCooldownSeconds
{
	float Value = 0.f;
	explicit FCooldownSeconds(float InValue = 0.f) : Value(InValue) {}
	bool IsReady() const { return Value <= 0.f; }
};

// プロジェクタイル残寿命（秒）
struct FProjectileLifeTime
{
	float Seconds = 0.f;
	explicit FProjectileLifeTime(float InSeconds = 0.f) : Seconds(InSeconds) {}
	bool IsExpired() const { return Seconds <= 0.f; }
	void Tick(float Dt) { Seconds -= Dt; }
};

// 周回角度（ラジアン）── King Bible / Peachone の軌道位置
struct FOrbitAngleRad
{
	float Value = 0.f;
	explicit FOrbitAngleRad(float InValue = 0.f) : Value(InValue) {}
	void Advance(float RadPerSec, float Dt) { Value += RadPerSec * Dt; }
};

// バウンス残回数 ── Runetracer 用
struct FBounceCount
{
	int32 Value = 0;
	explicit FBounceCount(int32 InValue = 0) : Value(InValue) {}
	bool HasBounces() const { return Value > 0; }
	void Consume() { --Value; }
};

// ---- Enum 定義 ---------------------------------------------------------------

UENUM(BlueprintType)
enum class EWeaponType : uint8
{
	None          = 0,
	Garlic        = 1,   // 旧 Aura → Garlic にリネーム（CoreRedirect で互換維持）
	Whip          = 2,
	MagicWand     = 3,
	Knife         = 4,
	Axe           = 5,
	Cross         = 6,
	KingBible     = 7,
	FireWand      = 8,
	SantaWater    = 9,
	Runetracer    = 10,
	LightningRing = 11,
	Pentagram     = 12,
	Peachone      = 13,
	EbonyWings    = 14,
	Laurel        = 15,
	// 進化後（Lv8 + パッシブで置き換わる）
	SoulEater     = 16,  // Garlic 進化
	BloodyTear    = 17,  // Whip 進化
	HolyWand      = 18,  // MagicWand 進化
	ThousandEdge  = 19,  // Knife 進化
	DeathSpiral   = 20,  // Axe 進化
	HeavenSword   = 21,  // Cross 進化
	UnholyVespers = 22,  // KingBible 進化
	Hellfire      = 23,  // FireWand 進化
	LaBorra       = 24,  // SantaWater 進化
	NoFuture      = 25,  // Runetracer 進化
	ThunderLoop   = 26,  // LightningRing 進化
	GorgeousMoon  = 27,  // Pentagram 進化
	Vandalier     = 28,  // Peachone + EbonyWings Union
	// ID 29〜63: 予約枠（将来の武器追加用）
	// obs 正規化: type_norm = id / MaxWeaponTypeCountReserved(64)
	// ID 64 以上: 非対応（type_norm >= 1.0 になるため）
};

UENUM(BlueprintType)
enum class EPassiveItemType : uint8
{
	None         = 0,
	Spinach      = 1,
	Armor        = 2,
	HollowHeart  = 3,
	Pummarola    = 4,
	EmptyTome    = 5,
	Candelabrador= 6,
	Bracer       = 7,
	Spellbinder  = 8,
	Duplicator   = 9,
	Wings        = 10,
	Attractorb   = 11,
	Clover       = 12,
	Crown        = 13,
	StoneMask    = 14,
	SkullOManiac = 15,
	Tirajisu     = 16,
	TorronasBox  = 17,
	// ID 18〜31: 予約枠（将来のパッシブ追加用）
	// obs 正規化: type_norm = id / MaxPassiveTypeCountReserved(32)
};

UENUM(BlueprintType)
enum class EGemType : uint8
{
	Blue  = 0,
	Green = 1,
	Red   = 2,
};

UENUM(BlueprintType)
enum class EFloorPickupType : uint8
{
	FloorChicken = 0,   // HP +30
	LittleHeart  = 1,   // HP +1
};

UENUM(BlueprintType)
enum class ESpecialPickupType : uint8
{
	Rosary    = 0,  // 画面内全敵を即撃破
	Orologion = 1,  // 全敵10秒フリーズ
	Vacuum    = 2,  // 全ジェムを引き寄せ
};

// ---- 構造体定義 --------------------------------------------------------------

struct FWeaponSlot
{
	EWeaponType      Type     = EWeaponType::None;
	FWeaponLevel     Level;                    // int32 → FWeaponLevel
	FCooldownSeconds Cooldown;                 // float → FCooldownSeconds
};

struct FPassiveSlot
{
	EPassiveItemType Type  = EPassiveItemType::None;
	int32            Level = 0;
};

// プロジェクタイル状態（強型使用）
struct FProjectileState
{
	FVector2D            Pos;
	FVector2D            Vel;
	FSimRadius           Radius;                                           // シム空間の当たり半径
	FDamage              Damage = FDamage(0.f);                            // 既存強型を流用
	EWeaponType          WeaponType    = EWeaponType::None;
	int32                WeaponSlotIdx = 0;
	FProjectileLifeTime  LifeTime;                                         // 残寿命（秒）
	FBounceCount         BounceCount;                                      // Runetracer バウンス残回数
	bool                 bPiercing     = false;
	FOrbitAngleRad       AngleRad;                                         // King Bible 等の軌道角度
	FSurvivorsElapsedTime LastHitTime  = FSurvivorsElapsedTime(-1000.f);   // 範囲武器クールダウン
	TArray<int32>        HitEnemyIds;  // 非 piercing 弾のヒット済み UniqueId（TSet より線形探索が速い）
};

// グラウンドゾーン（Santa Water / La Borra 用）
struct FGroundZoneState
{
	FVector2D          Pos;
	float              Radius       = 30.f;
	float              Damage       = 10.f;
	float              LifeTime     = 5.f;
	float              HitCooldown  = 0.5f;
	int32              WeaponSlotIdx = 0;
	EWeaponType        WeaponType   = EWeaponType::None;
	TMap<int32, float> EnemyLastHitTime; // Key: FEnemyState::UniqueId, Value: 最終ヒット時刻
};

// フロアヒールアイテム
struct FFloorPickupState
{
	FVector2D        Pos;
	EFloorPickupType Type    = EFloorPickupType::FloorChicken;
	bool             bActive = true;
};

// 特殊アイテム（Rosary / Orologion / Vacuum）
struct FSpecialPickupState
{
	FVector2D          Pos;
	ESpecialPickupType Type    = ESpecialPickupType::Rosary;
	bool               bActive = true;
};

// 破壊可能オブジェクト（Mad Forest のたき火）
struct FDestructibleState
{
	FVector2D Pos;
	bool      bActive = true;  // false = 破壊済み
};

// パッシブ効果合成キャッシュ
struct FPassiveEffects
{
	float DamageMult        = 1.f;
	float CooldownMult      = 1.f;
	float AreaMult          = 1.f;
	float SpeedMult         = 1.f;
	float DurationMult      = 1.f;
	float ExtraAmount       = 0.f;
	float MoveSpeedMult     = 1.f;
	float PickupRadiusMult  = 1.f;
	float HpMult            = 1.f;
	float RegenPerSec       = 0.f;
	float ArmorFlat         = 0.f;
	int32 MaxRevivalCount   = 0;   // Tirajisú: 最大リバイバル数
};

// エネミー状態（GarlicLastHitTime → UniqueId + WeaponLastHitTime[6] に更新）
struct FEnemyState
{
	FVector2D Pos;
	FVector2D Vel;
	int32     TypeId         = 0;
	float     HP             = 0.f;
	float     MaxHP          = 1.f;
	float     CollisionRadius= 0.f;
	float     ContactDamage  = 0.f;
	float     PlayerLastHitTime = -1000.f;
	int32     UniqueId       = 0;  // スポーン時にゲームグローバルカウンタで割り当て
	bool      bFrozen        = false;  // フリーズ状態（将来 freeze 武器用、現在は常に false）
	// 武器スロット別最終ヒット時刻（MaxWeaponSlots=6 に対応）
	FSurvivorsElapsedTime WeaponLastHitTime[6] = {
		FSurvivorsElapsedTime(-1000.f), FSurvivorsElapsedTime(-1000.f),
		FSurvivorsElapsedTime(-1000.f), FSurvivorsElapsedTime(-1000.f),
		FSurvivorsElapsedTime(-1000.f), FSurvivorsElapsedTime(-1000.f),
	};
};

struct FGemState
{
	FVector2D Pos;
	EGemType Type = EGemType::Blue;
};

// ---- 既存強型（後方互換維持） ------------------------------------------------

struct FEnemyTypeId
{
	int32 Value = 0;

	explicit FEnemyTypeId(int32 InValue = 0)
		: Value(InValue)
	{
	}

	int32 ToIndex() const { return Value; }
};

struct FHp
{
	float Value = 0.f;

	explicit FHp(float InValue = 0.f)
		: Value(InValue)
	{
	}
};

struct FSurvivorsGameTime
{
	float Seconds = 0.f;

	explicit FSurvivorsGameTime(float InSeconds = 0.f)
		: Seconds(InSeconds)
	{
	}
};

USTRUCT(BlueprintType)
struct FEnemyTypeParams
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadOnly)
	FString Name;

	UPROPERTY(EditAnywhere, BlueprintReadOnly)
	float BaseHP = 1.f;

	UPROPERTY(EditAnywhere, BlueprintReadOnly)
	float Speed = 50.f;

	UPROPERTY(EditAnywhere, BlueprintReadOnly)
	float ContactDamage = 5.f;

	UPROPERTY(EditAnywhere, BlueprintReadOnly)
	float CollisionRadius = 10.f;

	UPROPERTY(EditAnywhere, BlueprintReadOnly)
	float KnockbackResistance = 0.f;

	UPROPERTY(EditAnywhere, BlueprintReadOnly)
	bool bIsBoss = false;
};

USTRUCT(BlueprintType)
struct FEnemySpawnWeight
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadOnly)
	int32 TypeId = 0;

	UPROPERTY(EditAnywhere, BlueprintReadOnly)
	float Weight = 1.f;
};

USTRUCT(BlueprintType)
struct FSpawnWave
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadOnly)
	float TimeStart = 0.f;

	UPROPERTY(EditAnywhere, BlueprintReadOnly)
	float TimeEnd = 30.f;

	UPROPERTY(EditAnywhere, BlueprintReadOnly)
	float SpawnRate = 1.f;

	UPROPERTY(EditAnywhere, BlueprintReadOnly)
	int32 MaxEnemies = 50;

	UPROPERTY(EditAnywhere, BlueprintReadOnly)
	TArray<FEnemySpawnWeight> EnemyWeights;
};

struct FSurvivorsSpawnDebug
{
	float ElapsedTime = 0.f;
	float MaxEpisodeTime = 0.f;
	int32 EnemyCount = 0;
	int32 CurrentWaveIndex = INDEX_NONE;
	int32 MinActiveEnemies = 0;
	int32 MaxActiveEnemies = 0;
	int32 EffectiveMinEnemies = 0;
	int32 EffectiveMaxEnemies = 0;
	int32 MaxEnemyTypeId = 0;
	int32 AllowedSpawnTypeCount = 0;
	float SpawnAccumulator = 0.f;
	bool bHasCurrentWave = false;
	bool bUsedCurriculumEnemyPool = false;
	bool bSpawnBlocked = false;
	bool bTruncated = false;
};
