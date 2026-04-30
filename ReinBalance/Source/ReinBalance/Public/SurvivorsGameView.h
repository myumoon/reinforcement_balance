#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Components/StaticMeshComponent.h"
#include "Components/WidgetComponent.h"
#include "SurvivorsGame.h"
#include "SurvivorsGameView.generated.h"

class USurvivorsHPWidget;

/**
 * ASurvivorsGame のシミュレーション状態をビジュアルに反映するビューアクター。
 *
 * 使い方:
 *   1. レベルに ASurvivorsGame と ASurvivorsGameView を両方配置する
 *   2. ASurvivorsGameView の Details パネルで Game プロパティに ASurvivorsGame を設定する
 *   3. 両アクターをフィールド中心（同じ場所）に配置する
 *
 * ビジュアル仕様（上空から俯瞰）:
 *   プレイヤー   : 緑コーン（速度方向を向く）
 *   アイテム     : 黄色の球
 *   敵A(低速追跡) : 赤・正方形キューブ
 *   敵B(高速直進) : オレンジ・縦長キューブ
 *   敵C(予測追跡) : 赤紫・横長キューブ
 *   敵HP色       : 満タン=タイプ色, 瀕死=白 で線形補間
 *   オーラ範囲   : DrawDebugCircle（青）
 *   HP ウィジェット: Screen空間 UMG（緑→黄→赤バー）
 *   フィールド境界: 4辺の薄いキューブ
 */
UCLASS()
class REINBALANCE_API ASurvivorsGameView : public AActor
{
	GENERATED_BODY()

public:
	ASurvivorsGameView();

	/** ビジュアルの参照先シミュレーター */
	UPROPERTY(EditAnywhere, Category = "SurvivorsGameView")
	TObjectPtr<ASurvivorsGame> Game;

	/** シミュレーション座標（m）→ UE5 単位（cm）の変換スケール */
	UPROPERTY(EditAnywhere, Category = "SurvivorsGameView")
	float SimToUE = 50.f;

protected:
	virtual void BeginPlay() override;
	virtual void Tick(float DeltaTime) override;

private:
	// ---- ビジュアルコンポーネント ----

	UPROPERTY(VisibleAnywhere, Category = "Components")
	TObjectPtr<UStaticMeshComponent> PlayerMesh;

	UPROPERTY(VisibleAnywhere, Category = "Components")
	TObjectPtr<UWidgetComponent> HPWidgetComp;

	UPROPERTY()
	TArray<TObjectPtr<UStaticMeshComponent>> ItemMeshComponents;

	UPROPERTY()
	TArray<TObjectPtr<UStaticMeshComponent>> EnemyMeshComponents;

	/** 敵HP色変更用マテリアル（GC防止） */
	UPROPERTY()
	TArray<TObjectPtr<UMaterialInstanceDynamic>> EnemyMaterials;

	UPROPERTY()
	TArray<TObjectPtr<UStaticMeshComponent>> BoundaryWalls;

	/** 敵タイプ色キャッシュ（値型, UPROPERTY不要） */
	TArray<FLinearColor> EnemyTypeColors;

	// ---- キャッシュされたアセット参照 ----

	UPROPERTY()
	TObjectPtr<UStaticMesh> ConeMeshAsset;
	UPROPERTY()
	TObjectPtr<UStaticMesh> SphereMeshAsset;
	UPROPERTY()
	TObjectPtr<UStaticMesh> CubeMeshAsset;
	UPROPERTY()
	TObjectPtr<UMaterial> BaseMaterialAsset;

	// ---- 内部メソッド ----

	void LoadAssets();
	void SetupPlayerMesh();
	void SetupItemMeshes();
	void SetupBoundaryWalls();
	void SyncItemMeshes();
	void SyncEnemyMeshes();
	void UpdateEnemyColors();
	void UpdatePositions();
	void DrawAura();

	static FLinearColor GetEnemyTypeColor(int32 Type);
	static FVector      GetEnemyTypeScale(int32 Type);
	UMaterialInstanceDynamic* CreateColorMaterial(const FLinearColor& Color);
};
