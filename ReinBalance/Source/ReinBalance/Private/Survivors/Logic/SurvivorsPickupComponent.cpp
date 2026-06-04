#include "Survivors/Logic/SurvivorsPickupComponent.h"
#include "Survivors/Logic/SurvivorsGame.h"
#include "Survivors/Logic/SurvivorsGemComponent.h"

USurvivorsPickupComponent::USurvivorsPickupComponent()
{
	PrimaryComponentTick.bCanEverTick = false;
}

void USurvivorsPickupComponent::Initialize(ASurvivorsGame* InGame)
{
	Game = InGame;
}

void USurvivorsPickupComponent::CheckFloorPickups()
{
	if (!Game) return;

	const float PickupRadius = Game->GemPickupRadius;

	for (FFloorPickupState& Pickup : Game->FloorPickups)
	{
		if (!Pickup.bActive) continue;

		const float DistSq = (Game->PlayerPos - Pickup.Pos).SizeSquared();
		if (DistSq > FMath::Square(PickupRadius + 5.f)) continue;

		// 収集
		Pickup.bActive = false;

		// HP 回復
		float HealAmount = 0.f;
		switch (Pickup.Type)
		{
		case EFloorPickupType::FloorChicken:
			HealAmount = 30.f;
			break;
		case EFloorPickupType::LittleHeart:
			HealAmount = 1.f;
			break;
		default:
			break;
		}

		if (HealAmount > 0.f)
		{
			Game->PlayerHP = FMath::Min(Game->PlayerHP + HealAmount, Game->MaxPlayerHP);
		}
	}

	// 非アクティブアイテムを削除
	for (int32 i = Game->FloorPickups.Num() - 1; i >= 0; --i)
	{
		if (!Game->FloorPickups[i].bActive)
			Game->FloorPickups.RemoveAt(i);
	}
}

void USurvivorsPickupComponent::CheckSpecialPickups()
{
	if (!Game) return;

	const float PickupRadius = Game->GemPickupRadius;

	for (FSpecialPickupState& Pickup : Game->SpecialPickups)
	{
		if (!Pickup.bActive) continue;

		const float DistSq = (Game->PlayerPos - Pickup.Pos).SizeSquared();
		if (DistSq > FMath::Square(PickupRadius + 10.f)) continue;

		Pickup.bActive = false;

		switch (Pickup.Type)
		{
		case ESpecialPickupType::Rosary:
			// 全敵を即撃破
			for (FEnemyState& E : Game->Enemies)
			{
				if (!E.bPendingRemove)
				{
					E.HP            = 0.f;
					E.bPendingRemove = true;
					Game->LastReward += Game->KillReward;
				}
			}
			break;

		case ESpecialPickupType::Vacuum:
			// 全ジェムをプレイヤー位置に引き寄せ
			if (Game->GemComponent)
			{
				for (FGemState& Gem : Game->Gems)
				{
					if (!Gem.bPendingRemove)
						Gem.Pos = Game->PlayerPos;
				}
			}
			break;

		case ESpecialPickupType::Orologion:
			// 全フリーズ（スタブ: 現在は bFrozen を設定しても EnemyComponent の UpdateEnemies で無視される）
			for (FEnemyState& E : Game->Enemies)
				E.bFrozen = true;
			break;

		default:
			break;
		}
	}

	// 非アクティブアイテムを削除
	for (int32 i = Game->SpecialPickups.Num() - 1; i >= 0; --i)
	{
		if (!Game->SpecialPickups[i].bActive)
			Game->SpecialPickups.RemoveAt(i);
	}
}
