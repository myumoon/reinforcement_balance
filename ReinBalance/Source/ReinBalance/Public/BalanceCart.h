#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Components/SceneComponent.h"
#include "BalanceCart.generated.h"

/**
 * カート + 直列ポール（最大2本）の物理シミュレーターアクター。
 *
 * 構造: カート -- 関節1 -- ポール1 -- 関節2 -- ポール2
 *
 * UE5 物理エンジンを使用せず解析的に積分する。
 *   - ポール1: Barto et al. (1983) の完全連成 RK4
 *   - ポール2: ポール1先端の 2D 加速度を外部入力とした RK4
 *
 * 観測値は常に 6 次元（obs 次元固定 → --resume 転移学習がそのまま動作）。
 *   [0] CartPos  (m)
 *   [1] CartVel  (m/s)
 *   [2] θ1       (rad, 鉛直からの絶対角度)
 *   [3] θ̇1       (rad/s)
 *   [4] θ2_rel   (rad, ポール1 に対する関節角度 = θ2_abs - θ1)  ← NumPoles=1 時は 0
 *   [5] θ̇2_rel   (rad/s, 関節角速度 = θ̇2 - θ̇1)                ← NumPoles=1 時は 0
 */
UCLASS()
class REINBALANCE_API ABalanceCart : public AActor
{
	GENERATED_BODY()

public:
	ABalanceCart();

	/** 正規化された力 (-1~1) を加えて 1 物理ステップ進める。GameThread から呼ぶ。 */
	void PhysicsStep(float NormalizedForce);

	/** 初期状態にリセットする。Seed が未指定の場合はランダム。 */
	void ResetState(TOptional<int32> Seed);

	/** 観測値 6 次元を返す（コメント参照） */
	TArray<float> GetObservation() const;

	/** エピソード終了判定 */
	bool IsDone() const;

	/** ステップ報酬（生存で +1.0、終了時は 0.0） */
	float GetReward() const;

	// --- 構成 ---

	/** 有効なポール本数 (1 or 2) */
	UPROPERTY(EditAnywhere, Category = "Balance|Config")
	int32 NumPoles = 1;

	// --- 物理パラメータ（エディタで調整可能）---

	UPROPERTY(EditAnywhere, Category = "Balance|Physics")
	float CartMass = 1.0f;

	UPROPERTY(EditAnywhere, Category = "Balance|Physics")
	float PoleMass = 0.1f;            // kg（両ポール共通）

	UPROPERTY(EditAnywhere, Category = "Balance|Physics")
	float PoleHalfLength = 0.5f;      // m（ポール1 全長の半分）

	UPROPERTY(EditAnywhere, Category = "Balance|Physics")
	float PoleHalfLength2 = 0.25f;    // m（ポール2 全長の半分）

	UPROPERTY(EditAnywhere, Category = "Balance|Physics")
	float MaxForceNewtons = 10.0f;

	UPROPERTY(EditAnywhere, Category = "Balance|Physics")
	float MaxCartPositionMeters = 4.8f;

	/** NumPoles=1 時のポール倒れ判定角度 */
	UPROPERTY(EditAnywhere, Category = "Balance|Physics")
	float MaxPoleAngleDegrees = 60.0f;

	/** NumPoles=2 時: ポール2先端がこの高さ (m, カート関節基準) を下回ったら終了 */
	UPROPERTY(EditAnywhere, Category = "Balance|Physics")
	float MinTipHeightMeters = 0.5f;

protected:
	virtual void BeginPlay() override;
	virtual void OnConstruction(const FTransform& Transform) override;
	virtual void Tick(float DeltaTime) override;

private:
	float CartPos    = 0.f;
	float CartVel    = 0.f;
	float PoleAngle  = 0.f;  // rad, 絶対角度
	float PoleAngVel = 0.f;
	float PoleAngle2  = 0.f; // rad, 絶対角度（内部計算用）
	float PoleAngVel2 = 0.f;

	bool bEpisodeDone = false;

	static constexpr float Gravity   = 9.8f;
	static constexpr float PhysicsDt = 1.0f / 60.0f;

	FRandomStream RandStream;

	void Integrate(float Force);

	/**
	 * ポール2 の RK4 積分。
	 * @param TipAccelX  ポール1先端の水平加速度 (m/s²)
	 * @param TipAccelZ  ポール1先端の垂直加速度 (m/s²)
	 */
	void IntegratePole2(float TipAccelX, float TipAccelZ);

	void UpdateVisuals() const;
	void ApplyPole2Visibility(bool bVisible);

	UPROPERTY(VisibleAnywhere, Category = "Components")
	TObjectPtr<UStaticMeshComponent> CartMesh;

	UPROPERTY(VisibleAnywhere, Category = "Components")
	TObjectPtr<USceneComponent> PolePivot;

	UPROPERTY(VisibleAnywhere, Category = "Components")
	TObjectPtr<UStaticMeshComponent> PoleMesh;

	/** 関節2: PolePivot の子として配置 → ポール1が傾くと先端へ追従 */
	UPROPERTY(VisibleAnywhere, Category = "Components")
	TObjectPtr<USceneComponent> PolePivot2;

	UPROPERTY(VisibleAnywhere, Category = "Components")
	TObjectPtr<UStaticMeshComponent> PoleMesh2;
};
