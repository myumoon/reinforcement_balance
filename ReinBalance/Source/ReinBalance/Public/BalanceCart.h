#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Components/SceneComponent.h"
#include "BalanceCart.generated.h"

/**
 * カート + N 本ポールの CartPole 物理シミュレーターアクター。
 *
 * UE5 物理エンジンは使用せず、Barto et al. (1983) の解析的な運動方程式を
 * RK4 で積分する。NumPoles で 1〜2 本を切り替え可能。
 *
 * 観測値は常に 6 次元 [CartPos, CartVel, Angle1, AngVel1, Angle2, AngVel2]。
 * NumPoles=1 のとき後 2 要素は 0。obs 次元を固定することで
 * --resume による転移学習（1本 → 2本）がそのまま動作する。
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

	/** 観測値 6 次元 [CartPos(m), CartVel(m/s), Angle1(rad), AngVel1(rad/s), Angle2(rad), AngVel2(rad/s)] */
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
	float CartMass = 1.0f;            // kg

	UPROPERTY(EditAnywhere, Category = "Balance|Physics")
	float PoleMass = 0.1f;            // kg（両ポール共通）

	UPROPERTY(EditAnywhere, Category = "Balance|Physics")
	float PoleHalfLength = 0.5f;      // m（ポール1 全長の半分）

	UPROPERTY(EditAnywhere, Category = "Balance|Physics")
	float PoleHalfLength2 = 0.25f;    // m（ポール2 全長の半分、短い方）

	UPROPERTY(EditAnywhere, Category = "Balance|Physics")
	float MaxForceNewtons = 10.0f;    // N

	UPROPERTY(EditAnywhere, Category = "Balance|Physics")
	float MaxCartPositionMeters = 4.8f;

	UPROPERTY(EditAnywhere, Category = "Balance|Physics")
	float MaxPoleAngleDegrees = 60.0f;

protected:
	virtual void BeginPlay() override;
	virtual void OnConstruction(const FTransform& Transform) override;
	virtual void Tick(float DeltaTime) override;

private:
	// 物理状態（SI 単位系）
	float CartPos    = 0.f;  // m
	float CartVel    = 0.f;  // m/s
	float PoleAngle  = 0.f;  // rad
	float PoleAngVel = 0.f;  // rad/s
	float PoleAngle2  = 0.f;
	float PoleAngVel2 = 0.f;

	bool bEpisodeDone = false;

	static constexpr float Gravity   = 9.8f;
	static constexpr float PhysicsDt = 1.0f / 60.0f;

	FRandomStream RandStream;

	/** ポール1 + カートの完全連成 RK4 積分 */
	void Integrate(float Force);

	/** ポール2 の RK4 積分（カート加速度を外部入力として扱う） */
	void IntegratePole2(float CartAccel);

	void UpdateVisuals() const;
	void ApplyPole2Visibility(bool bVisible);

	UPROPERTY(VisibleAnywhere, Category = "Components")
	TObjectPtr<UStaticMeshComponent> CartMesh;

	UPROPERTY(VisibleAnywhere, Category = "Components")
	TObjectPtr<USceneComponent> PolePivot;

	UPROPERTY(VisibleAnywhere, Category = "Components")
	TObjectPtr<UStaticMeshComponent> PoleMesh;

	/** ポール2 の関節（カート天面、Y 方向に +15cm オフセット） */
	UPROPERTY(VisibleAnywhere, Category = "Components")
	TObjectPtr<USceneComponent> PolePivot2;

	UPROPERTY(VisibleAnywhere, Category = "Components")
	TObjectPtr<UStaticMeshComponent> PoleMesh2;
};
