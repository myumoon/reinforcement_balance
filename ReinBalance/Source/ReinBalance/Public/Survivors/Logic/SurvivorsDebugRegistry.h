// Fill out your copyright notice in the Description page of Project Settings.

#pragma once
#include "CoreMinimal.h"

#if WITH_EDITOR

/** レベルアップ挙動のデバッグオーバーライドを提供するインターフェース */
class ISurvivorsDebugSlot
{
public:
	virtual ~ISurvivorsDebugSlot() = default;
	virtual bool GetSkipGetWeaponOnLevelUp() const = 0;
	virtual bool GetSkipGetPassiveItemOnLevelUp() const = 0;
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
