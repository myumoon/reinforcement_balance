#pragma once
/**
 * DeployObs v2 用の deploy raw state（カメラ・自機・スロット・画面付近の entity）の型と JSON 化を定義する。
 *
 * Training 側が実機と同じ「画面に映る物だけ」から観測を作れるよう、sim の状態を世界座標のまま書き出す。
 * 値の生成は FSurvivorsGameLogic::BuildDeployRawState が行い、ここは型と文字列化だけを持つ。
 * このファイルは UObject 系ヘッダーをインクルードしてはならない（ReinBalanceLogic は Json モジュールに依存しない）。
 */

#include "CoreMinimal.h"

/**
 * entity id の種類（id 空間）。
 *
 * 敵・ジェム・武器エフェクトはそれぞれ別のカウンタで番号を振るため、
 * 種類を上位ビットに入れて種類間で id が衝突しないようにする。
 */
enum class ESurvivorsDeployRawIdSpace : uint8
{
	Enemy      = 1,
	Gem        = 2,
	Projectile = 3,
	GroundZone = 4,
	Orbit      = 5,
	Aura       = 6,
};

namespace SurvivorsDeployRaw
{
	/** JSON の schema_version 文字列 */
	inline const TCHAR* SchemaVersion = TEXT("survivors_deploy_raw.v1");

	/**
	 * 画面外除外に使うカメラ範囲の余白（ワールド単位）。
	 *
	 * |dx| <= ScreenHalfWidthU + CullMarginU かつ |dy| <= ScreenHalfHeightU + CullMarginU の entity だけを出す。
	 * 最終的な可視判定（中心が画面内）は Python 側の投影で行うので、ここは payload を抑えるための粗い除外。
	 */
	constexpr float CullMarginU = 100.f;

	/** 種類内のローカル id が使うビット幅。上位に id 空間を置く */
	constexpr int32 LocalIdBits = 40;

	/**
	 * id 空間とローカル id から entity id（int64）を合成する。
	 *
	 * 上位ビットが種類、下位 40 ビットがローカル id。JSON の数値（2^53 未満）で安全に表せる。
	 */
	inline int64 MakeEntityId(ESurvivorsDeployRawIdSpace Space, int64 LocalId)
	{
		return (static_cast<int64>(Space) << LocalIdBits) | (LocalId & ((int64(1) << LocalIdBits) - 1));
	}

	/**
	 * 中心 Center のカメラ範囲＋余白に Pos が入るかを返す（境界を含む）。
	 *
	 * 半幅・半高にそれぞれ MarginU を足した矩形の内側（ちょうど境界上も含む）なら true。
	 */
	REINBALANCELOGIC_API bool IsWithinCullRange(
		FVector2D Center, float HalfWidth, float HalfHeight, float MarginU, FVector2D Pos);
}

/**
 * deploy raw の 1 entity（敵・ジェム・武器エフェクト）。
 *
 * ClassName は Common の vocabulary と同じ文字列（enemy_normal / gem_blue / weapon_orbit など）。
 * Slot と TtlTrueS は武器エフェクトだけが持ち、それ以外は未設定（JSON では null）。
 */
struct FSurvivorsDeployRawEntity
{
	int64           EntityId    = 0;
	FString         ClassName;
	FVector2D       WorldPos    = FVector2D::ZeroVector;
	float           RadiusWorld = 0.f;
	TOptional<int32> Slot;
	TOptional<float> TtlTrueS;
	bool            bWarning    = false;
};

/**
 * 武器またはパッシブの 1 スロット（HUD に表示される種類とレベル）。
 *
 * TypeId は C++ の EWeaponType / EPassiveItemType の値。Common の weapon_vocabulary /
 * passive_vocabulary は同じ並び順なので、Python 側は添字で名前へ変換できる。空きスロットは TypeId=0。
 */
struct FSurvivorsDeployRawSlot
{
	int32 Index  = 0;
	int32 TypeId = 0;
	int32 Level  = 0;
};

/**
 * 1 tick 分の deploy raw state。
 *
 * カメラ（中心は自機）、自機、HUD 相当の値、スロット、画面付近の entity 列をまとめたもの。
 */
struct FSurvivorsDeployRawState
{
	FVector2D CameraCenter     = FVector2D::ZeroVector;
	float     CameraHalfWidth  = 0.f;
	float     CameraHalfHeight = 0.f;
	float     CullMarginU      = 0.f;
	FVector2D PlayerPos        = FVector2D::ZeroVector;
	float     HpRatio          = 0.f;
	int32     PlayerLevel      = 0;
	float     ElapsedS         = 0.f;
	float     DurationMult     = 1.f;
	TArray<FSurvivorsDeployRawSlot>   WeaponSlots;
	TArray<FSurvivorsDeployRawSlot>   PassiveSlots;
	TArray<FSurvivorsDeployRawEntity> Entities;
};

namespace SurvivorsDeployRaw
{
	/**
	 * deploy raw state を JSON オブジェクト文字列にする。
	 *
	 * HTTP の /reset・/step 応答へそのまま埋め込む形。キー順は固定で、浮動小数は %.9g で書く。
	 * Logic モジュールは Json モジュールに依存しないため、文字列を直接組み立てる。
	 */
	REINBALANCELOGIC_API FString ToJson(const FSurvivorsDeployRawState& State);
}
