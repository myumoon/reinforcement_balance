#pragma once

#include "CoreMinimal.h"
#include "Survivors/Logic/SurvivorsCollisionTypes.h"
#include "Survivors/Logic/SurvivorsValueTypes.h"
#include "SurvivorsTypes.generated.h"

struct FSurvivorsObsSegment
{
	FString Name;
	int32 Dim;
};

// ---- HitFrame 型定義 ---------------------------------------------------------

UENUM()
enum class ESurvivorsHitType : uint8
{
	ContactDamage    = 0,
	WeaponAreaDamage = 1,
	ProjectileDamage = 2,
	GroundZoneDamage = 3,
	PickupCollect    = 4,
};

struct FSurvivorsHitEvent
{
	ESurvivorsHitType    Type               = ESurvivorsHitType::ContactDamage;
	FSurvivorsCollisionRef Target;
	float                Damage             = 0.f;
	int32                WeaponSlot         = 0;
	FVector2D            KnockbackDir       = FVector2D::ZeroVector;
	float                KnockbackStrength  = 0.f;
	float                KnockbackResistance= 0.f;
};

struct FSurvivorsHitFrame
{
	TArray<FSurvivorsHitEvent> Events;
	void Reset() { Events.Reset(); }
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
	Rosary        = 0,  // 画面内全敵を即撃破
	Orologion     = 1,  // 全敵10秒フリーズ
	Vacuum        = 2,  // 全ジェムを引き寄せ
	TreasureChest = 3,  // 進化判定
};

// ---- レベルアップ選択肢 -------------------------------------------------------

struct FLevelUpChoice
{
	enum class EChoiceType : uint8
	{
		WeaponNew,
		WeaponUpgrade,
		WeaponEvolve,
		PassiveNew,
		PassiveUpgrade,
	};

	EChoiceType      ChoiceType  = EChoiceType::WeaponNew;
	EWeaponType      WeaponType  = EWeaponType::None;
	EPassiveItemType PassiveType = EPassiveItemType::None;
	int32            SlotIdx     = -1;
	int32            NewLevel    = 1;
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
	bool                 bHasReversed  = false;       // Cross ブーメランの折り返し済みフラグ
	bool                 bPendingExplosion = false;  // FireWand 爆発待機フラグ（LifeTime 切れで爆発予約、次 Tick で処理）
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
	float GrowthMult        = 1.f;
	float CurseMult         = 1.f;
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
	bool      bPendingRemove = false;  // Apply フェーズで死亡マーク済み（Finalize で削除）
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
	EGemType  Type       = EGemType::Blue;
	float     BaseExperienceValue = 2.f;
	int32     UniqueId   = 0;
	bool      bPendingRemove = false;
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
	float XPDrop = 2.f;

	UPROPERTY(EditAnywhere, BlueprintReadOnly)
	float CollisionRadius = 10.f;

	UPROPERTY(EditAnywhere, BlueprintReadOnly)
	float KnockbackResistance = 0.f;

	UPROPERTY(EditAnywhere, BlueprintReadOnly)
	bool bResistsFreeze = false;

	UPROPERTY(EditAnywhere, BlueprintReadOnly)
	bool bResistsInstantKill = false;

	UPROPERTY(EditAnywhere, BlueprintReadOnly)
	bool bResistsDebuff = false;

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
	int32 MinEnemies = 0;

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
	int32 TotalWaveCount = 0;
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
