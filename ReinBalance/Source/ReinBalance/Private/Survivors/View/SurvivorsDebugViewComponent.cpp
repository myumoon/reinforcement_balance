#include "Survivors/View/SurvivorsDebugViewComponent.h"

#include "Survivors/Logic/SurvivorsGame.h"
#include "Survivors/Logic/SurvivorsGameConstants.h"
#include "Survivors/Logic/SurvivorsTypes.h"
#include "Survivors/View/SurvivorsViewPalette.h"
#include "Engine/Engine.h"
#include "GameFramework/PlayerController.h"


USurvivorsDebugViewComponent::USurvivorsDebugViewComponent()
{
	PrimaryComponentTick.bCanEverTick = false;
}

void USurvivorsDebugViewComponent::Initialize(ASurvivorsGame* InGame)
{
	Game = InGame;
}

void USurvivorsDebugViewComponent::UpdateView()
{
	UWorld* World = GetWorld();
	APlayerController* PC = World ? World->GetFirstPlayerController() : nullptr;
	if (PC)
	{
		bool bCtrlDown  = PC->IsInputKeyDown(EKeys::LeftControl) || PC->IsInputKeyDown(EKeys::RightControl);
		bool bShiftDown = PC->IsInputKeyDown(EKeys::LeftShift)   || PC->IsInputKeyDown(EKeys::RightShift);
		bool bKeyDown   = PC->IsInputKeyDown(ToggleKey);
		bool bCombo     = bCtrlDown && !bShiftDown && bKeyDown;
		if (bCombo && !bKeyWasDown) bVisible = !bVisible;
		bKeyWasDown = bCombo;
	}

	if (!bVisible) return;
	DrawDebugOverlay();
}

void USurvivorsDebugViewComponent::DrawDebugOverlay() const
{
	if (!GEngine || !Game) return;
	
	// 描画順が逆になってしまうのでリストに登録して逆順に描画する
	TArray<FDebugLine> pendingLines;
	pendingLines.Reserve(128);

	int32 Key = DebugKeyBase;
	DrawSection_Game(Key, pendingLines);
	DrawSection_Status(Key, pendingLines);
	DrawSection_Slots(Key, pendingLines);
	DrawSection_Enemy(Key, pendingLines);
	DrawSection_Train(Key, pendingLines);

	// 逆順描画
	DrawDebugOverlay(pendingLines, GEngine);
}

void USurvivorsDebugViewComponent::DrawSection_Game(int32& Key, TArray<FDebugLine>& pendingLines) const
{
	// キー: 5000..5009
	Key = DebugKeyBase + 0;
	AddLine(pendingLines, Key++, TEXT("--- Game ---"), FLinearColor::White);

	const FSurvivorsSpawnDebug& SpawnDebug = Game->GetSpawnDebug();
	const int32  CurrentWave = SpawnDebug.CurrentWaveIndex + 1;
	const int32  TotalWaves  = SpawnDebug.TotalWaveCount;
	const float  Elapsed     = SpawnDebug.ElapsedTime;
	const float  MaxTime     = SpawnDebug.MaxEpisodeTime;

	AddLine(pendingLines, Key++,
		FString::Printf(TEXT("%-14s %d/%d"), TEXT("Wave"), CurrentWave, TotalWaves),
		FLinearColor::White);
	AddLine(pendingLines, Key++,
		FString::Printf(TEXT("%-14s %.1f / %.1f"), TEXT("ElapsedTime"), Elapsed, MaxTime),
		FLinearColor::White);
}

void USurvivorsDebugViewComponent::DrawSection_Status(int32& Key, TArray<FDebugLine>& pendingLines) const
{
	// キー: 5010..5019
	Key = DebugKeyBase + 10;
	AddLine(pendingLines, Key++, TEXT("--- Status ---"), FLinearColor::White);

	const float XP    = Game->GetPlayerXP();
	const int32 Level = Game->GetPlayerLevel();
	const float HP    = Game->GetPlayerHP();
	const float MaxHP = Game->GetMaxPlayerHP();
	const float ShieldTimer = Game->GetPlayerShieldTimer();

	const float XPNeeded = Game->GetXPRequiredForNextLevel();

	AddLine(pendingLines, Key++,
		FString::Printf(TEXT("%-10s %.0f/%.0f"), TEXT("XP"), XP, XPNeeded),
		FLinearColor::White);

	AddLine(pendingLines, Key++,
		FString::Printf(TEXT("%-10s %d/%d"), TEXT("Lv"), Level, SurvivorsGameConstants::MaxPlayerLevel),
		FLinearColor::White);

	AddLine(pendingLines, Key++,
		FString::Printf(TEXT("%-10s %.0f/%.0f"), TEXT("HP"), HP, MaxHP),
		FLinearColor::White);

	const FLinearColor ShieldColor = Game->IsShieldActive()
		? FLinearColor::White
		: FLinearColor(0.5f, 0.5f, 0.5f);
	AddLine(pendingLines, Key++,
		FString::Printf(TEXT("%-10s %.1f"), TEXT("Shield"), ShieldTimer),
		ShieldColor);
}

void USurvivorsDebugViewComponent::DrawSection_Slots(int32& Key, TArray<FDebugLine>& pendingLines) const
{
	// キー: 5020..5032
	Key = DebugKeyBase + 20;
	AddLine(pendingLines, Key++, TEXT("--- Slots ---"), FLinearColor::White);

	// WeaponSlots[0..5]
	for (int32 i = 0; i < SurvivorsGameConstants::MaxWeaponSlots; ++i)
	{
		const FWeaponSlot& Slot = Game->GetWeaponSlot(i);
		if (Slot.Type == EWeaponType::None)
		{
			AddLine(pendingLines, Key++,
				FString::Printf(TEXT("[%d] ---"), i),
				FLinearColor(0.5f, 0.5f, 0.5f));
		}
		else
		{
			const FString Name = GetEnumShortName(Slot.Type);
			const int32 WLv    = Slot.Level.Value;
			AddLine(pendingLines, Key++,
				FString::Printf(TEXT("[%d] W.%-14s Lv%d/%d"), i, *Name, WLv, SurvivorsGameConstants::MaxWeaponLevel),
				GetWeaponColor(Slot.Type));
		}
	}

	// PassiveSlots[0..5]
	for (int32 i = 0; i < SurvivorsGameConstants::MaxPassiveSlots; ++i)
	{
		const FPassiveSlot& Slot = Game->GetPassiveSlot(i);
		if (Slot.Type == EPassiveItemType::None)
		{
			AddLine(pendingLines, Key++,
				FString::Printf(TEXT("[%d] ---"), 6 + i),
				FLinearColor(0.5f, 0.5f, 0.5f));
		}
		else
		{
			const FString Name    = GetEnumShortName(Slot.Type);
			const int32 PLv       = Slot.Level;
			const int32 MaxLv = Game->GetPassiveItemMaxLevel(Slot.Type);
			AddLine(pendingLines, Key++,
				FString::Printf(TEXT("[%d] P.%-14s Lv%d/%d"), 6 + i, *Name, PLv, MaxLv),
				GetPassiveItemColor(Slot.Type));
		}
	}
}

void USurvivorsDebugViewComponent::DrawSection_Enemy(int32& Key, TArray<FDebugLine>& pendingLines) const
{
	// キー: 5040〜（動的）
	Key = DebugKeyBase + 40;
	AddLine(pendingLines, Key++, TEXT("--- Enemy ---"), FLinearColor::White);

	const FSurvivorsSpawnDebug& SpawnDebugEnemy = Game->GetSpawnDebug();
	const int32 MaxEnemyTypeId = SpawnDebugEnemy.MaxEnemyTypeId;
	AddLine(pendingLines, Key++,
		FString::Printf(TEXT("%-24s %d"), TEXT("MaxEnemyId"), MaxEnemyTypeId),
		FLinearColor::White);

	// 種別行（生存数 > 0 のみ）
	const TMap<int32, int32> EnemyCounts = Game->GetEnemyCountByType();
	for (int32 TypeId = 0; TypeId <= MaxEnemyTypeId; ++TypeId)
	{
		const int32* Count = EnemyCounts.Find(TypeId);
		if (!Count || *Count <= 0) continue;

		AddLine(pendingLines, Key++,
			FString::Printf(TEXT("  %-22s %d"), *Game->GetEnemyTypeDebugLabel(TypeId), *Count),
			GetEnemyTypeColor(TypeId));
	}

	// MinMaxEnemies / EffectiveMinMaxEnemies
	const FSurvivorsSpawnDebug& SpawnDebug = Game->GetSpawnDebug();
	AddLine(pendingLines, Key++,
		FString::Printf(TEXT("%-24s %d, %d"), TEXT("MinMaxEnemies"),
			SpawnDebug.MinActiveEnemies, SpawnDebug.MaxActiveEnemies),
		FLinearColor::White);
	AddLine(pendingLines, Key++,
		FString::Printf(TEXT("%-24s %d, %d"), TEXT("EffectiveMinMaxEnemies"),
			SpawnDebug.EffectiveMinEnemies, SpawnDebug.EffectiveMaxEnemies),
		FLinearColor::White);
}

void USurvivorsDebugViewComponent::DrawSection_Train(int32& Key, TArray<FDebugLine>& pendingLines) const
{
	// キー: 5110..5119
	Key = DebugKeyBase + 110;
	AddLine(pendingLines, Key++, TEXT("--- Train ---"), FLinearColor::White);

	const float Reward = Game->GetLastReward();
	AddLine(pendingLines, Key++,
		FString::Printf(TEXT("%-10s %.4f"), TEXT("Reward"), Reward),
		FLinearColor::White);
}

void USurvivorsDebugViewComponent::AddLine(TArray<FDebugLine>& lines, int32 Key, const FString& Text, FLinearColor Color)
{
	lines.Add({Key, Text, Color});
}

void USurvivorsDebugViewComponent::DrawDebugOverlay(TArrayView<const FDebugLine> lines, UEngine* engine)

{
	if (engine == nullptr)
	{
		return;
	}
		
	for (int i = lines.Num() - 1; i >= 0; --i)
	{
		const auto& L = lines[i];
		engine->AddOnScreenDebugMessage(L.Key, 0.0f, L.Color.ToFColor(true), L.Text);
	}
}
