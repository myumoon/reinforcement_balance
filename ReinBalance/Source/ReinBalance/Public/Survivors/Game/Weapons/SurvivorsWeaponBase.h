#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "Survivors/Game/SurvivorsTypes.h"
#include "SurvivorsWeaponBase.generated.h"

class ASurvivorsGame;
class USurvivorsWeaponComponent;
class USurvivorsCollisionComponent;

/**
 * 全武器クラスの抽象基底クラス。
 * 各武器は Tick(float Dt) をオーバーライドして個別挙動を実装する。
 */
UCLASS(Abstract)
class REINBALANCE_API USurvivorsWeaponBase : public UObject
{
	GENERATED_BODY()

public:
	/** 初期化・スロット割り当て */
	virtual void Initialize(ASurvivorsGame* InGame, USurvivorsWeaponComponent* InComp, int32 InSlotIdx);

	/** エピソードリセット時に呼ぶ */
	virtual void Reset();

	/** 毎物理ステップ呼ばれる（クールダウン管理・発射判定・Tick処理） */
	virtual void Tick(float Dt) PURE_VIRTUAL(USurvivorsWeaponBase::Tick, );

	/** HitFrame フェーズ: 当たり判定を計算しイベントを HitFrame に追加する */
	virtual void ComputeHits(USurvivorsCollisionComponent* CollComp, FSurvivorsHitFrame& HitFrame) {}

	/** 武器レベル変更時（パラメータ再キャッシュ用） */
	virtual void OnLevelChanged(FWeaponLevel NewLevel) {}

	// ---- 軌道オーブ View API（KingBible / Peachone / EbonyWings / Vandalier） ----
	/** この武器が持つ軌道オーブ数（0 = オーブなし） */
	virtual int32     GetOrbitOrbCount()                const { return 0; }
	/** i 番目のオーブのシム座標位置 */
	virtual FVector2D GetOrbitOrbPos(int32 OrbIdx)      const { return FVector2D::ZeroVector; }
	/** オーブの表示半径（シム単位） */
	virtual float     GetOrbitOrbVisualRadius()         const { return 0.f; }

	// ---- アクセサ ----
	EWeaponType  GetWeaponType() const { return WeaponType; }
	void         SetWeaponType(EWeaponType InType);

	FWeaponLevel GetLevel()      const { return Level; }
	void         SetLevel(FWeaponLevel InLevel);

	/** クールダウン残秒（obs 用） */
	FCooldownSeconds GetCooldownRemaining() const { return CooldownTimer; }

protected:
	ASurvivorsGame*            Game       = nullptr;
	USurvivorsWeaponComponent* WeaponComp = nullptr;
	int32                      SlotIdx    = 0;
	EWeaponType                WeaponType = EWeaponType::None;
	FWeaponLevel               Level;
	FCooldownSeconds           CooldownTimer;

	/** パッシブ効果を参照するヘルパー（Game->CachedPassiveEffects を返す） */
	const FPassiveEffects& GetPassiveEffects() const;
};
