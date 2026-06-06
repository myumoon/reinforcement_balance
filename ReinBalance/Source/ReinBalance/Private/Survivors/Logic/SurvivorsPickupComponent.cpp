#include "Survivors/Logic/SurvivorsPickupComponent.h"
#include "Survivors/Logic/SurvivorsGame.h"
#include "Survivors/Logic/SurvivorsGameConstants.h"
#include "Survivors/Logic/SurvivorsGemComponent.h"
#include "Survivors/Logic/SurvivorsPlayerComponent.h"

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
				const bool bResistsInstantKill = Game->EnemyTypeTable.IsValidIndex(E.TypeId)
					&& Game->EnemyTypeTable[E.TypeId].bResistsInstantKill;
				if (!E.bPendingRemove && !bResistsInstantKill)
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
			// 10秒グローバルフリーズ（ElapsedTime でタイマー管理）
			// UpdateEnemies() でグローバルフリーズ時刻を確認し自動解除される
			Game->GlobalFreezeUntilTime = Game->ElapsedTime + 10.f;
			break;

		case ESpecialPickupType::TreasureChest:
			if (Game->bEnableEvolutions && Game->PlayerComponent)
			{
				const TArray<int32> EvolvableSlots = Game->PlayerComponent->GetEvolvableWeapons();
				if (EvolvableSlots.Num() > 0)
				{
					const int32 SlotIdx = EvolvableSlots[Game->RandStream.RandRange(0, EvolvableSlots.Num() - 1)];
					const EWeaponType BaseType = Game->WeaponSlots[SlotIdx].Type;
					for (const SurvivorsGameConstants::FEvolutionRule& Rule : SurvivorsGameConstants::EvolutionTable)
					{
						if (Rule.BaseWeapon == BaseType)
						{
							Game->PlayerComponent->EvolveWeapon(SlotIdx, Rule.EvolvedWeapon);
							break;
						}
					}
				}
			}
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
