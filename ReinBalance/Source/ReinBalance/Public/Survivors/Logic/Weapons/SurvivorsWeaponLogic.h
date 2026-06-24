#pragma once
// このファイルは UObject 系ヘッダーをインクルードしてはならない。
// CoreUObject.h, UObject/Object.h, Components/ActorComponent.h,
// .generated.h 等の追加は禁止。

#include "CoreMinimal.h"
#include "Survivors/Logic/SurvivorsTypes.h"

class FSurvivorsGameLogic;

/**
 * FSurvivorsWeaponLogic — 純粋 C++ 武器基底クラス。
 *
 * UObject 非依存。FSurvivorsGameLogic が TUniquePtr<FSurvivorsWeaponLogic> で管理する。
 */
class REINBALANCE_API FSurvivorsWeaponLogic
{
public:
	virtual ~FSurvivorsWeaponLogic() = default;

	/** 初期化・スロット割り当て */
	virtual void Initialize(FSurvivorsGameLogic* InLogic, int32 InSlotIdx);

	/** エピソードリセット時に呼ぶ */
	virtual void Reset();

	/** 毎物理ステップ呼ばれる（クールダウン管理・発射判定・Tick 処理） */
	virtual void Tick(float Dt) = 0;

	/** HitFrame フェーズ: 当たり判定を計算しイベントを HitFrame に追加する */
	virtual void ComputeHits(FSurvivorsHitFrame& HitFrame) {}

	/** 武器レベル変更時（パラメータ再キャッシュ用） */
	virtual void OnLevelChanged(FWeaponLevel NewLevel) {}

	// ---- 軌道オーブ View API（KingBible / Peachone / EbonyWings / Vandalier） ----
	virtual int32     GetOrbitOrbCount()           const { return 0; }
	virtual FVector2D GetOrbitOrbPos(int32 OrbIdx) const { return FVector2D::ZeroVector; }
	virtual float     GetOrbitOrbVisualRadius()    const { return 0.f; }

	// ---- アクセサ ----
	EWeaponType      GetWeaponType()          const { return WeaponType; }
	void             SetWeaponType(EWeaponType InType);

	FWeaponLevel     GetLevel()               const { return Level; }
	void             SetLevel(FWeaponLevel InLevel);

	/** クールダウン残秒（obs 用） */
	FCooldownSeconds GetCooldownRemaining()   const { return CooldownTimer; }

protected:
	FSurvivorsGameLogic* Logic     = nullptr;
	int32                SlotIdx   = 0;
	EWeaponType          WeaponType = EWeaponType::None;
	FWeaponLevel         Level;
	FCooldownSeconds     CooldownTimer;

	/** パッシブ効果を参照するヘルパー（Logic->CachedPassiveEffects を返す） */
	const FPassiveEffects& GetPassiveEffects() const;
};
