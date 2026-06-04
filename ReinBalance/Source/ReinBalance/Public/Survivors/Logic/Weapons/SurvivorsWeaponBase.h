#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "Survivors/Logic/SurvivorsTypes.h"
#include "SurvivorsWeaponBase.generated.h"

class ASurvivorsGame;
class USurvivorsWeaponComponent;

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

	/** 武器レベル変更時（パラメータ再キャッシュ用） */
	virtual void OnLevelChanged(FWeaponLevel NewLevel) {}

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
