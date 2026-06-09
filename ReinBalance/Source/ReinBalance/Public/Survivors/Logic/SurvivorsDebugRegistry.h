// Fill out your copyright notice in the Description page of Project Settings.

#pragma once
#include "CoreMinimal.h"

#if WITH_EDITOR

/** レベルアップ挙動のデバッグオーバーライドを提供するインターフェース */
class ISurvivorsDebugSlot
{
public:
	virtual ~ISurvivorsDebugSlot() = default;

	/** true のとき、レベルアップ時に新規武器の追加をスキップする */
	virtual bool GetSkipGetWeaponOnLevelUp() const = 0;

	/** true のとき、レベルアップ時に新規パッシブアイテムの追加をスキップする */
	virtual bool GetSkipGetPassiveItemOnLevelUp() const = 0;

	/**
	 * true のとき、レベルアップ時に既存スロットのレベルアップ（武器強化・進化・
	 * パッシブ強化）をすべてスキップする。
	 * "garlic_only" モードでは WeaponNew / PassiveNew の選択肢が存在しないため、
	 * このフラグが true の場合はレベルアップ処理全体をスキップすることになる。
	 * 通常モードでは WeaponUpgrade / WeaponEvolve / PassiveUpgrade が対象となる。
	 */
	virtual bool GetSkipSlotLevelUp() const = 0;
};

/**
 * Survivors デバッグコンポーネントのグローバルレジストリ。
 * ReinBalance モジュールに定義し、ReinBalanceEditor から登録する。
 */
class REINBALANCE_API FSurvivorsDebugRegistry
{
public:
	static void RegisterSlotComponent(ISurvivorsDebugSlot* Component);
	static void UnregisterSlotComponent(ISurvivorsDebugSlot* Component);
	static ISurvivorsDebugSlot* GetSlotComponent();

private:
	static ISurvivorsDebugSlot* ActiveSlotComponent;
};

#endif // WITH_EDITOR
