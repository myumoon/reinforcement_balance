#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Survivors/Game/SurvivorsTypes.h"
#include "SurvivorsWeaponComponent.generated.h"

class ASurvivorsGame;
class USurvivorsWeaponBase;
class USurvivorsCollisionComponent;

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
	
public:
	// ---- obs / view アクセサ ----

	/** obs 生成用: プロジェクタイル + GroundZone を統一ビューで返す */
	TArray<FProjectileState> GetProjectileObsView() const;

	int32     GetProjectileCount()              const;
	FVector2D GetProjectilePos(int32 i)         const;
	FSimRadius GetProjectileRadius(int32 i)     const;
	EWeaponType GetProjectileWeaponType(int32 i)const;
	float     GetProjectileBoxHalfWidth(int32 i) const;

	int32     GetGroundZoneCount()              const;
	FVector2D GetGroundZonePos(int32 i)         const;
	float     GetGroundZoneRadius(int32 i)      const;
	EWeaponType GetGroundZoneWeaponType(int32 i)const;
	bool      IsGroundZoneWarning(int32 i)      const;

	/** 全スロットの軌道オーブ総数（KingBible / Peachone / EbonyWings / Vandalier） */
	int32       GetOrbitOrbCount()               const;
	FVector2D   GetOrbitOrbPos(int32 GlobalIdx)  const;
	EWeaponType GetOrbitOrbWeaponType(int32 GlobalIdx) const;
	float       GetOrbitOrbVisualRadius(int32 GlobalIdx) const;

	/** 武器インスタンスへのアクセス（obs の cooldown 取得用） */
	USurvivorsWeaponBase* GetWeaponInstance(int32 SlotIdx) const;

	// todo: 弾の管理は専用のクラスを作成し、武器生成時にコンストラクタで渡すようにしたい
	/** 武器実装がプロジェクタイル・ゾーンを追加するための API */
	void SpawnProjectile(const FProjectileState& P) { Projectiles.Add(P); }
	void SpawnGroundZone(const FGroundZoneState& Z) { GroundZones.Add(Z); }

	/** 武器実装がプロジェクタイルを直接操作するための API（ComputeHits 内で使用） */
	TArray<FProjectileState>& GetProjectiles() { return Projectiles; }

	/**
	 * 指定スロットのプロジェクタイルを全てコールバックで更新する。
	 * Axe の重力・Cross の折り返し・KingBible の軌道更新などに使用。
	 * Callback は (FProjectileState&, float Dt) → bool (false で削除) を受け取る。
	 */
	void UpdateProjectilesBySlot(int32 InSlotIdx, float Dt,
		TFunctionRef<bool(FProjectileState&, float)> Callback);


	// ---- 毎ステップ処理 ----

	/** 全武器 Tick() を呼ぶ（ダメージなし、クールダウン管理のみ） */
	void TickWeapons(float Dt);

	/** 後方互換のために残している。HitFrame 経由の当たり判定（Garlic/GroundZone）は実行しない。
	 *  正確な当たり判定には PhysicsStep 内の BuildEnemyGrid/ComputeAllWeaponHits/ApplyWeaponHits を使うこと。 */
	void TickAllWeapons(float Dt);
	
	/** 全武器 ComputeHits を呼ぶ（HitFrame に当たり判定結果を収集） */
	void ComputeAllWeaponHits(USurvivorsCollisionComponent* CollComp, FSurvivorsHitFrame& HitFrame);

	/** HitFrame のイベントを適用する（HP 更新・ノックバック・ドロップ） */
	void ApplyWeaponHits(FSurvivorsHitFrame& HitFrame);
	
private:

	/** 全プロジェクタイル移動・寿命管理 */
	void TickProjectiles(float Dt);

	/** Santa Water ゾーン寿命管理（ダメージ判定は ComputeGroundZoneHits に移管） */
	void TickGroundZones(float Dt);

	/** プロジェクタイル vs 敵 当たり判定（後方互換: 直接呼ぶ場合） */
	void ApplyProjectileHits();

private:
	// ---- プロジェクタイル・ゾーン（全武器共有プール） ----
	TArray<FProjectileState> Projectiles;
	TArray<FGroundZoneState>  GroundZones;

	UPROPERTY()
	TArray<TObjectPtr<USurvivorsWeaponBase>> WeaponInstances;  // [MaxWeaponSlots]

	ASurvivorsGame* Game = nullptr;

	/** ファクトリー: EWeaponType → 対応クラスのインスタンスを生成 */
	USurvivorsWeaponBase* CreateWeaponInstance(EWeaponType Type);

	void ComputeGroundZoneHits(USurvivorsCollisionComponent* CollComp, FSurvivorsHitFrame& HitFrame);
	void ComputeProjectileHits(USurvivorsCollisionComponent* CollComp, FSurvivorsHitFrame& HitFrame);
};
