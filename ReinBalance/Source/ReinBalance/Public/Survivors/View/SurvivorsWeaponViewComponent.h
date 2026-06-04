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
 * DrawAura・DrawGroundZones・DrawLaurelShield は DrawDebugCircle で描画（少数の円形なので ISM 不要）。
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

	/** Garlic / Soul Eater のオーラ円を DrawDebugCircle で描画 */
	void DrawWeaponAuras();

	/** Santa Water ゾーン（最大 ~8 個）を DrawDebugCircle で描画 */
	void DrawGroundZones();

	/** Laurel シールド円を DrawDebugCircle で描画 */
	void DrawLaurelShield();
};
