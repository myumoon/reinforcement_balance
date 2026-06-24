#pragma once

#include "CoreMinimal.h"
#include "Survivors/Game/SurvivorsTypes.h"

/**
 * UE ワールド空間の Z レイヤー高さ（オブジェクトの重なりを管理）。
 * 乱数で Z を決めず必ずこの型を通す。
 */
struct FWorldLayerZ
{
	float Value = 0.f;
	explicit FWorldLayerZ(float InValue = 0.f) : Value(InValue) {}

	static FWorldLayerZ GroundZone()  { return FWorldLayerZ(1.f); }
	static FWorldLayerZ Aura()        { return FWorldLayerZ(2.f); }
	static FWorldLayerZ Pickup()      { return FWorldLayerZ(3.f); }
	static FWorldLayerZ Projectile()  { return FWorldLayerZ(5.f); }
	static FWorldLayerZ Player()      { return FWorldLayerZ(8.f); }
	static FWorldLayerZ Shield()      { return FWorldLayerZ(9.f); }
};

/**
 * シム座標 → UE ワールド座標変換ヘルパー。
 * View 側の全変換はこれを経由し、直接 SimToUE 乗算を書かない。
 * Initialize() 時に 1 つだけ作り全描画処理で共有する。
 */
struct FSimToWorldConverter
{
	float Scale = 1.f;  // = ASurvivorsGame::SimToUE

	FSimToWorldConverter() = default;
	explicit FSimToWorldConverter(float InScale) : Scale(InScale) {}

	FVector ToWorld(FVector2D SimPos, FWorldLayerZ Z = FWorldLayerZ()) const
	{
		return FVector(SimPos.X * Scale, SimPos.Y * Scale, Z.Value);
	}

	float Radius(FSimRadius R) const { return R.Value * Scale; }
	float Radius(float SimRadius) const { return SimRadius * Scale; }
};
