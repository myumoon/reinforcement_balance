#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "InputCoreTypes.h"
#include "SurvivorsDebugViewComponent.generated.h"

class ASurvivorsGame;

/**
 * 画面左上にゲーム状態デバッグオーバーレイを表示するコンポーネント。
 * AddOnScreenDebugMessage を使用し毎フレーム同じ行を上書きする（残像なし）。
 * Ctrl+F10 で表示/非表示を切り替える（F10 単体は TrainingRenderToggleActor に委譲）。
 */
UCLASS(ClassGroup=(Custom), meta=(BlueprintSpawnableComponent))
class REINBALANCE_API USurvivorsDebugViewComponent : public UActorComponent
{
	GENERATED_BODY()
public:
	USurvivorsDebugViewComponent();

	void Initialize(ASurvivorsGame* InGame);
	void UpdateView();   // SurvivorsGameView::Tick() から毎フレーム呼ばれる

	/** デフォルト表示状態 */
	UPROPERTY(EditAnywhere, Category = "Survivors|Debug")
	bool bVisible = true;

	/** 表示トグルキー（修飾キー Ctrl + ToggleKey） */
	UPROPERTY(EditAnywhere, Category = "Survivors|Debug")
	FKey ToggleKey = EKeys::F10;

private:
	UPROPERTY()
	TObjectPtr<ASurvivorsGame> Game;

	bool bKeyWasDown = false;

	void DrawDebugOverlay() const;

	// 各カテゴリ描画ヘルパー
	void DrawSection_Game(int32& Key) const;
	void DrawSection_Status(int32& Key) const;
	void DrawSection_Slots(int32& Key) const;
	void DrawSection_Enemy(int32& Key) const;
	void DrawSection_Train(int32& Key) const;

	static void AddLine(int32 Key, const FString& Text, FLinearColor Color);

	// ラベル幅定数（FString::Printf の %-Ns フォーマット用）
	static constexpr int32 LabelWidth_Game   = 14;
	static constexpr int32 LabelWidth_Status = 10;
	static constexpr int32 LabelWidth_Enemy  = 24;
	static constexpr int32 LabelWidth_Train  = 10;
	static constexpr int32 SlotNameWidth     = 14;

	// AddOnScreenDebugMessage キーベース値（他モジュールと重複しない範囲）
	static constexpr int32 DebugKeyBase = 5000;

	// Enum 短縮名取得ヘルパー（"EWeaponType::Garlic" → "Garlic"）
	template<typename TEnum>
	static FString GetEnumShortName(TEnum Value)
	{
		FString FullName = UEnum::GetValueAsString(Value);
		int32 SepIdx = INDEX_NONE;
		if (FullName.FindLastChar(TEXT(':'), SepIdx))
		{
			return FullName.RightChop(SepIdx + 1);
		}
		return FullName;
	}
};
