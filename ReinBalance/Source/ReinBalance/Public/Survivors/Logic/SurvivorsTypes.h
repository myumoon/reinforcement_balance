#pragma once

#include "CoreMinimal.h"
#include "SurvivorsTypes.generated.h"

struct FSurvivorsObsSegment
{
	FString Name;
	int32 Dim;
};

UENUM(BlueprintType)
enum class EWeaponType : uint8
{
	None     = 0,
	Aura     = 1,
	Whip     = 2,
	Fireball = 3,
};

struct FWeaponSlot
{
	EWeaponType Type = EWeaponType::None;
	int32 Level = 0;
};

UENUM(BlueprintType)
enum class EGemType : uint8
{
	Blue  = 0,
	Green = 1,
	Red   = 2,
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

struct FEnemyState
{
	FVector2D Pos;
	FVector2D Vel;
	int32 TypeId = 0;
	float HP = 0.f;
	float MaxHP = 1.f;
	float CollisionRadius = 0.f;
	float ContactDamage = 0.f;
	float GarlicLastHitTime = -1000.f;
	float PlayerLastHitTime = -1000.f;
};

struct FGemState
{
	FVector2D Pos;
	EGemType Type = EGemType::Blue;
};

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

struct FDamage
{
	float Value = 0.f;

	explicit FDamage(float InValue = 0.f)
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

struct FSurvivorsElapsedTime
{
	float Seconds = 0.f;

	explicit FSurvivorsElapsedTime(float InSeconds = 0.f)
		: Seconds(InSeconds)
	{
	}
};
