#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Components/StaticMeshComponent.h"
#include "CoinGame.h"
#include "CoinGameView.generated.h"

/**
 * ACoinGame のシミュレーション状態をビジュアルに反映するビューアクター。
 *
 * ACoinGame（シミュレーション）とビジュアルを分離する。
 * 毎 Tick で ACoinGame の状態を読み取りメッシュを更新する。
 *
 * 使い方:
 *   1. レベルに ACoinGame と ACoinGameView を両方配置する
 *   2. ACoinGameView の Details パネルで Game プロパティに ACoinGame を設定する
 *   3. 両アクターをフィールド中心（同じ場所）に配置する
 *
 * ビジュアル仕様（上空から俯瞰）:
 *   自機  : 緑コーン（速度方向を向く）
 *   コイン: 黄色の球
 *   敵A（遅い直進）: 赤・正方形キューブ
 *   敵B（速い直進）: オレンジ・縦長キューブ
 *   敵C（予測追跡）: 赤紫・横長キューブ
 */
UCLASS()
class REINBALANCE_API ACoinGameView : public AActor
{
	GENERATED_BODY()

public:
	ACoinGameView();

	/** ビジュアルの参照先シミュレーター */
	UPROPERTY(EditAnywhere, Category = "CoinGameView")
	TObjectPtr<ACoinGame> Game;

	/** シミュレーション座標（m）→ UE5 単位（cm）の変換スケール
	 *  デフォルト 50: FieldHalfSize=10m → 視覚フィールド 1000×1000 UE単位 */
	UPROPERTY(EditAnywhere, Category = "CoinGameView")
	float SimToUE = 50.f;

protected:
	virtual void BeginPlay() override;
	virtual void Tick(float DeltaTime) override;

private:
	// ---- ビジュアルコンポーネント ----

	UPROPERTY(VisibleAnywhere, Category = "Components")
	TObjectPtr<UStaticMeshComponent> PlayerMesh;

	UPROPERTY(VisibleAnywhere, Category = "Components")
	TArray<TObjectPtr<UStaticMeshComponent>> CoinMeshComponents;

	/** 動的に生成・破棄される敵メッシュ（GC 防止のため UPROPERTY） */
	UPROPERTY()
	TArray<TObjectPtr<UStaticMeshComponent>> EnemyMeshComponents;

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
	void SetupCoinMeshes();
	void SyncCoinMeshes();
	void SyncEnemyMeshes();
	void UpdatePositions();

	UStaticMeshComponent* CreateEnemyMesh(int32 EnemyIndex, int32 Type);
	class UMaterialInstanceDynamic* CreateColorMaterial(const FLinearColor& Color);
};
