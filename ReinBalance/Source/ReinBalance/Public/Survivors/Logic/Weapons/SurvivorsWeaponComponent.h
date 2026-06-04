#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Survivors/Logic/SurvivorsTypes.h"
#include "SurvivorsWeaponComponent.generated.h"

class ASurvivorsGame;
class USurvivorsWeaponBase;

/**
 * 全武器スロットを管理するコンポーネント兼ファクトリー。
 * ASurvivorsGame の PhysicsStep から TickAllWeapons() を呼ぶことで全武器を更新する。
 */
UCLASS(ClassGroup=(Custom), meta=(BlueprintSpawnableComponent))
class REINBALANCE_API USurvivorsWeaponComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	USurvivorsWeaponComponent();

	void Initialize(ASurvivorsGame* InGame);
	void Reset();

	// ---- 武器スロット着脱 ----

	/** スロット idx に武器を装備する。既存インスタンスは破棄される */
	void EquipWeapon(int32 SlotIdx, EWeaponType Type, int32 Level = 1);

	/** スロット idx の武器を外す */
	void UnequipWeapon(int32 SlotIdx);

	// ---- 毎ステップ処理 ----

	/** 全武器 Tick() を呼ぶ（PhysicsStep から呼ぶ） */
	void TickAllWeapons(float Dt);

	/** 全プロジェクタイル移動・寿命管理 */
	void TickProjectiles(float Dt);

	/** Santa Water ゾーン継続ダメージ */
	void TickGroundZones(float Dt);

	/** プロジェクタイル vs 敵 当たり判定 */
	void ApplyProjectileHits();

	// ---- obs / view アクセサ ----

	/** obs 生成用: プロジェクタイル + GroundZone を統一ビューで返す */
	TArray<FProjectileState> GetProjectileObsView() const;

	int32     GetProjectileCount()              const;
	FVector2D GetProjectilePos(int32 i)         const;
	FSimRadius GetProjectileRadius(int32 i)     const;
	EWeaponType GetProjectileWeaponType(int32 i)const;

	int32     GetGroundZoneCount()              const;
	FVector2D GetGroundZonePos(int32 i)         const;
	float     GetGroundZoneRadius(int32 i)      const;
	EWeaponType GetGroundZoneWeaponType(int32 i)const;

	/** 武器インスタンスへのアクセス（obs の cooldown 取得用） */
	USurvivorsWeaponBase* GetWeaponInstance(int32 SlotIdx) const;

	/** 武器実装がプロジェクタイル・ゾーンを追加するための API */
	void SpawnProjectile(const FProjectileState& P) { Projectiles.Add(P); }
	void SpawnGroundZone(const FGroundZoneState& Z) { GroundZones.Add(Z); }

private:
	// ---- プロジェクタイル・ゾーン（全武器共有プール） ----
	TArray<FProjectileState> Projectiles;
	TArray<FGroundZoneState>  GroundZones;

	UPROPERTY()
	TArray<TObjectPtr<USurvivorsWeaponBase>> WeaponInstances;  // [MaxWeaponSlots]

	ASurvivorsGame* Game = nullptr;

	/** ファクトリー: EWeaponType → 対応クラスのインスタンスを生成 */
	USurvivorsWeaponBase* CreateWeaponInstance(EWeaponType Type);
};
