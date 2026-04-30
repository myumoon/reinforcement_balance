#pragma once

#include "Blueprint/UserWidget.h"
#include "SurvivorsHPWidget.generated.h"

class UProgressBar;

/** プレイヤー HP を表示する UMG ウィジェット。
 *  SurvivorsGameView の UWidgetComponent にアタッチして使用する。
 *  満タン=緑, 50%=黄, 0%=赤 で線形補間する。
 */
UCLASS()
class REINBALANCE_API USurvivorsHPWidget : public UUserWidget
{
	GENERATED_BODY()

public:
	/** HP バーと色を更新する。毎 Tick 呼ぶこと。 */
	void UpdateDisplay(float HP, float MaxHP);

protected:
	virtual void NativeConstruct() override;

private:
	UPROPERTY()
	TObjectPtr<UProgressBar> HPBar;
};
