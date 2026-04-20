#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Components/SceneComponent.h"
#include "BalanceCart.generated.h"

/**
 * カート + 1本ポールの CartPole 物理シミュレーターアクター。
 *
 * UE5 物理エンジンは使用せず、Barto et al. (1983) の解析的な運動方程式を
 * RK4 で積分する。訓練中は ABalanceHttpEnvService の Tick から
 * PhysicsStep() を呼び出すことで 1 物理ステップを進める。
 *
 * 観測値は 6 次元 [CartPos, CartVel, PoleAngle, PoleAngVel, 0, 0] (SI単位)。
 * 後 2 要素は将来の 2 本ポール拡張用のプレースホルダー。
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

	/** 観測値 6 次元を返す [CartPos(m), CartVel(m/s), PoleAngle(rad), PoleAngVel(rad/s), 0, 0] */
	TArray<float> GetObservation() const;

	/** エピソード終了判定（カートが範囲外 or ポールが倒れた） */
	bool IsDone() const;

	/** ステップ報酬（生存で +1.0、終了時は 0.0） */
	float GetReward() const;

	// --- 物理パラメータ（エディタで調整可能）---

	UPROPERTY(EditAnywhere, Category = "Balance|Physics")
	float CartMass = 1.0f;            // kg

	UPROPERTY(EditAnywhere, Category = "Balance|Physics")
	float PoleMass = 0.1f;            // kg

	UPROPERTY(EditAnywhere, Category = "Balance|Physics")
	float PoleHalfLength = 0.5f;      // m（ポール全長の半分）

	UPROPERTY(EditAnywhere, Category = "Balance|Physics")
	float MaxForceNewtons = 10.0f;    // N

	UPROPERTY(EditAnywhere, Category = "Balance|Physics")
	float MaxCartPositionMeters = 4.8f;

	UPROPERTY(EditAnywhere, Category = "Balance|Physics")
	float MaxPoleAngleDegrees = 60.0f;

protected:
	virtual void BeginPlay() override;
	virtual void Tick(float DeltaTime) override;

private:
	// 物理状態（SI 単位系）
	float CartPos    = 0.f;  // m
	float CartVel    = 0.f;  // m/s
	float PoleAngle  = 0.f;  // rad（鉛直上方向を 0 とした傾き角）
	float PoleAngVel = 0.f;  // rad/s

	bool bEpisodeDone = false;

	static constexpr float Gravity   = 9.8f;
	static constexpr float PhysicsDt = 1.0f / 60.0f; // DefaultEngine.ini の FixedFrameRate と一致

	FRandomStream RandStream;

	void Integrate(float Force);
	void UpdateVisuals() const;

	UPROPERTY(VisibleAnywhere, Category = "Components")
	TObjectPtr<UStaticMeshComponent> CartMesh;

	/** ポールの回転軸（カート天面の関節位置）*/
	UPROPERTY(VisibleAnywhere, Category = "Components")
	TObjectPtr<USceneComponent> PolePivot;

	UPROPERTY(VisibleAnywhere, Category = "Components")
	TObjectPtr<UStaticMeshComponent> PoleMesh;
};
