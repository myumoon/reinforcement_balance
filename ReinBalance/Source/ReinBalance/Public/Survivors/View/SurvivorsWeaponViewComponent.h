#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Survivors/View/SurvivorsViewTypes.h"
#include "Survivors/View/SurvivorsViewPalette.h"
#include "SurvivorsWeaponViewComponent.generated.h"

class ASurvivorsGame;
class UInstancedStaticMeshComponent;
class UMaterial;
class UMaterialInstanceDynamic;
class USceneComponent;
class UStaticMesh;

/**
 * 武器・プロジェクタイル・フロアアイテム・特殊アイテム・破壊可能オブジェクトを描画する View コンポーネント。
 *
 * ISM 12パレット方式（SurvivorsViewPalette 参照）で描画コール数を最小化。
 * Whip/BloodyTear は DrawDebugBox、範囲武器・円形武器は DrawDebugCircle で描画（ISM 不要）。
 */
UCLASS(ClassGroup=(Custom), meta=(BlueprintSpawnableComponent))
class REINBALANCE_API USurvivorsWeaponViewComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	USurvivorsWeaponViewComponent();

	void Initialize(ASurvivorsGame* InGame, USceneComponent* AttachParent);
	void UpdateView();

private:
	UPROPERTY()
	TObjectPtr<ASurvivorsGame> Game;

	UPROPERTY()
	TArray<TObjectPtr<UInstancedStaticMeshComponent>> PaletteISMs;  // [EViewPalette::Count]

	UPROPERTY()
	TObjectPtr<UStaticMesh> SphereMesh;

	UPROPERTY()
	TObjectPtr<UMaterial> BaseMat;

	FSimToWorldConverter Converter;

	// ---- 内部ヘルパー ----
	UMaterialInstanceDynamic* CreateColorMaterial(const FLinearColor& Color);

	/** プロジェクタイルを対応パレット ISM に追加 */
	void AddProjectileInstances();

	/** フロアアイテム・特殊アイテムを対応パレット ISM に追加 */
	void AddPickupInstances();

	/** 破壊可能オブジェクトを ISM に追加 */
	void AddDestructibleInstances();

	/** Whip / BloodyTear の矩形スイープを DrawDebugBox で描画 */
	void DrawWhipInstances();

	/** Garlic / Soul Eater のオーラ円を DrawDebugCircle で描画 */
	void DrawWeaponAuras();

	/** Santa Water / La Borra ゾーンを DrawDebugCircle で描画 */
	void DrawGroundZones();

	/** Laurel シールド円を DrawDebugCircle で描画 */
	void DrawLaurelShield();

	/** KingBible / Peachone / EbonyWings / Vandalier の軌道オーブを DrawDebugCircle で描画 */
	void DrawOrbitOrbs();

	/** LightningRing / ThunderLoop の有効範囲を DrawDebugCircle で描画 */
	void DrawLightningRings();

	/** Pentagram / GorgeousMoon の有効範囲を DrawDebugCircle で描画 */
	void DrawPentagramFields();
};
