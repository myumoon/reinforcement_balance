#include "Survivors/Logic/Weapons/SurvivorsWeaponComponent.h"

#include "Survivors/Logic/SurvivorsGame.h"
#include "Survivors/Logic/SurvivorsGameConstants.h"
#include "Survivors/Logic/SurvivorsGemComponent.h"
#include "Survivors/Logic/Weapons/SurvivorsWeaponBase.h"
#include "Survivors/Logic/Weapons/Projectile/SurvivorsGarlicWeapon.h"
#include "Survivors/Logic/Weapons/Projectile/SurvivorsWhipWeapon.h"
#include "Survivors/Logic/Weapons/Projectile/SurvivorsMagicWandWeapon.h"
#include "Survivors/Logic/Weapons/Projectile/SurvivorsKnifeWeapon.h"
#include "Survivors/Logic/Weapons/Projectile/SurvivorsAxeWeapon.h"
#include "Survivors/Logic/Weapons/Projectile/SurvivorsCrossWeapon.h"
#include "Survivors/Logic/Weapons/Projectile/SurvivorsKingBibleWeapon.h"
#include "Survivors/Logic/Weapons/Projectile/SurvivorsFireWandWeapon.h"
#include "Survivors/Logic/Weapons/Projectile/SurvivorsSantaWaterWeapon.h"
#include "Survivors/Logic/Weapons/Projectile/SurvivorsRunetracerWeapon.h"
#include "Survivors/Logic/Weapons/Projectile/SurvivorsLightningRingWeapon.h"
#include "Survivors/Logic/Weapons/Projectile/SurvivorsPentagramWeapon.h"
#include "Survivors/Logic/Weapons/Projectile/SurvivorsPeachoneWeapon.h"
#include "Survivors/Logic/Weapons/Projectile/SurvivorsEbonyWingsWeapon.h"
#include "Survivors/Logic/Weapons/Projectile/SurvivorsVandalierWeapon.h"
#include "Survivors/Logic/Weapons/Projectile/SurvivorsLaurelWeapon.h"

USurvivorsWeaponComponent::USurvivorsWeaponComponent()
{
	PrimaryComponentTick.bCanEverTick = false;
}

void USurvivorsWeaponComponent::Initialize(ASurvivorsGame* InGame)
{
	Game = InGame;
	WeaponInstances.SetNum(SurvivorsGameConstants::MaxWeaponSlots);
	Projectiles.Reserve(64);
	GroundZones.Reserve(16);
}

void USurvivorsWeaponComponent::Reset()
{
	Projectiles.Empty();
	GroundZones.Empty();
	Projectiles.Reserve(64);

	// 全スロットのインスタンスをクリア（古いインスタンスが残らないよう null に設定）
	// 初期武器の再装備は呼び出し元（SurvivorsGame::ResetState）が EquipWeapon で行う
	for (int32 i = 0; i < WeaponInstances.Num(); ++i)
	{
		WeaponInstances[i] = nullptr;
	}
}

void USurvivorsWeaponComponent::EquipWeapon(int32 SlotIdx, EWeaponType Type, int32 Level)
{
	if (!WeaponInstances.IsValidIndex(SlotIdx)) return;

	USurvivorsWeaponBase* Instance = CreateWeaponInstance(Type);
	if (!Instance) return;

	Instance->Initialize(Game, this, SlotIdx);
	Instance->SetWeaponType(Type);
	Instance->SetLevel(FWeaponLevel(Level));
	WeaponInstances[SlotIdx] = Instance;
}

void USurvivorsWeaponComponent::UnequipWeapon(int32 SlotIdx)
{
	if (!WeaponInstances.IsValidIndex(SlotIdx)) return;
	WeaponInstances[SlotIdx] = nullptr;
}

void USurvivorsWeaponComponent::TickAllWeapons(float Dt)
{
	for (int32 i = 0; i < WeaponInstances.Num(); ++i)
	{
		if (WeaponInstances[i])
		{
			WeaponInstances[i]->Tick(Dt);
		}
	}

	TickProjectiles(Dt);
	TickGroundZones(Dt);
	ApplyProjectileHits();
}

void USurvivorsWeaponComponent::TickProjectiles(float Dt)
{
	for (int32 i = Projectiles.Num() - 1; i >= 0; --i)
	{
		FProjectileState& P = Projectiles[i];

		// King Bible / Orbit 系は Pos を AngleRad から再計算するため WeaponBase::Tick で行う
		// それ以外は速度で移動
		P.Pos += P.Vel * Dt;
		P.LifeTime.Tick(Dt);

		if (P.LifeTime.IsExpired())
		{
			Projectiles.RemoveAt(i);
		}
	}
}

void USurvivorsWeaponComponent::TickGroundZones(float Dt)
{
	if (!Game) return;

	for (int32 i = GroundZones.Num() - 1; i >= 0; --i)
	{
		FGroundZoneState& Z = GroundZones[i];
		Z.LifeTime -= Dt;
		if (Z.LifeTime <= 0.f)
		{
			GroundZones.RemoveAt(i);
			continue;
		}

		// 範囲内の敵にクールダウン付きダメージ
		for (int32 EIdx = Game->Enemies.Num() - 1; EIdx >= 0; --EIdx)
		{
			FEnemyState& E = Game->Enemies[EIdx];
			const float Dist = FVector2D::Distance(Z.Pos, E.Pos);
			if (Dist <= Z.Radius + E.CollisionRadius)
			{
				const float* LastHit = Z.EnemyLastHitTime.Find(E.UniqueId);
				const float  Now     = Game->ElapsedTime;
				if (!LastHit || (Now - *LastHit) >= Z.HitCooldown)
				{
					E.HP -= Z.Damage;
					Z.EnemyLastHitTime.Add(E.UniqueId, Now);

					if (E.HP <= 0.f)
					{
						Game->GemComponent->DropGem(E.TypeId, E.Pos);
						Game->Enemies.RemoveAt(EIdx);
						Game->LastReward += Game->KillReward;
					}
				}
			}
		}
	}
}

void USurvivorsWeaponComponent::ApplyProjectileHits()
{
	if (!Game) return;

	for (int32 i = Projectiles.Num() - 1; i >= 0; --i)
	{
		FProjectileState& P = Projectiles[i];
		bool bShouldRemove = false;

		for (int32 EIdx = Game->Enemies.Num() - 1; EIdx >= 0; --EIdx)
		{
			FEnemyState& E = Game->Enemies[EIdx];

			// 非 piercing 弾: ヒット済みなら判定スキップ
			if (!P.bPiercing)
			{
				bool bAlreadyHit = P.HitEnemyIds.Contains(E.UniqueId);
				if (bAlreadyHit) continue;
			}

			const float HitR = P.Radius.Value + E.CollisionRadius;
			if (FVector2D::DistSquared(P.Pos, E.Pos) < HitR * HitR)
			{
				E.HP -= P.Damage.Value;
				if (!P.bPiercing)
				{
					P.HitEnemyIds.Add(E.UniqueId);
					bShouldRemove = true;  // 非 piercing: 1体ヒットで消滅
				}

				if (E.HP <= 0.f)
				{
					Game->GemComponent->DropGem(E.TypeId, E.Pos);
					Game->Enemies.RemoveAt(EIdx);
					Game->LastReward += Game->KillReward;
					if (!P.bPiercing) break;
				}
			}
		}

		if (bShouldRemove)
		{
			Projectiles.RemoveAt(i);
		}
	}
}

TArray<FProjectileState> USurvivorsWeaponComponent::GetProjectileObsView() const
{
	// プロジェクタイル + GroundZone を統一 FProjectileState ビューとして返す
	TArray<FProjectileState> View;
	View.Reserve(Projectiles.Num() + GroundZones.Num());

	for (const FProjectileState& P : Projectiles)
	{
		View.Add(P);
	}

	// GroundZone を疑似プロジェクタイルとして混在（Vel=0, Radius=GroundZone.Radius）
	for (const FGroundZoneState& Z : GroundZones)
	{
		FProjectileState ZProj;
		ZProj.Pos          = Z.Pos;
		ZProj.Vel          = FVector2D::ZeroVector;
		ZProj.Radius       = FSimRadius(Z.Radius);
		ZProj.WeaponType   = Z.WeaponType;
		ZProj.WeaponSlotIdx= Z.WeaponSlotIdx;
		View.Add(ZProj);
	}

	return View;
}

int32 USurvivorsWeaponComponent::GetProjectileCount() const
{
	return Projectiles.Num();
}

FVector2D USurvivorsWeaponComponent::GetProjectilePos(int32 i) const
{
	return Projectiles.IsValidIndex(i) ? Projectiles[i].Pos : FVector2D::ZeroVector;
}

FSimRadius USurvivorsWeaponComponent::GetProjectileRadius(int32 i) const
{
	return Projectiles.IsValidIndex(i) ? Projectiles[i].Radius : FSimRadius(0.f);
}

EWeaponType USurvivorsWeaponComponent::GetProjectileWeaponType(int32 i) const
{
	return Projectiles.IsValidIndex(i) ? Projectiles[i].WeaponType : EWeaponType::None;
}

int32 USurvivorsWeaponComponent::GetGroundZoneCount() const
{
	return GroundZones.Num();
}

FVector2D USurvivorsWeaponComponent::GetGroundZonePos(int32 i) const
{
	return GroundZones.IsValidIndex(i) ? GroundZones[i].Pos : FVector2D::ZeroVector;
}

float USurvivorsWeaponComponent::GetGroundZoneRadius(int32 i) const
{
	return GroundZones.IsValidIndex(i) ? GroundZones[i].Radius : 0.f;
}

EWeaponType USurvivorsWeaponComponent::GetGroundZoneWeaponType(int32 i) const
{
	return GroundZones.IsValidIndex(i) ? GroundZones[i].WeaponType : EWeaponType::None;
}

USurvivorsWeaponBase* USurvivorsWeaponComponent::GetWeaponInstance(int32 SlotIdx) const
{
	return WeaponInstances.IsValidIndex(SlotIdx) ? WeaponInstances[SlotIdx].Get() : nullptr;
}

USurvivorsWeaponBase* USurvivorsWeaponComponent::CreateWeaponInstance(EWeaponType Type)
{
	switch (Type)
	{
		case EWeaponType::Garlic:
		case EWeaponType::SoulEater:      return NewObject<USurvivorsGarlicWeapon>(this);
		case EWeaponType::Whip:
		case EWeaponType::BloodyTear:     return NewObject<USurvivorsWhipWeapon>(this);
		case EWeaponType::MagicWand:
		case EWeaponType::HolyWand:       return NewObject<USurvivorsMagicWandWeapon>(this);
		case EWeaponType::Knife:
		case EWeaponType::ThousandEdge:   return NewObject<USurvivorsKnifeWeapon>(this);
		case EWeaponType::Axe:
		case EWeaponType::DeathSpiral:    return NewObject<USurvivorsAxeWeapon>(this);
		case EWeaponType::Cross:
		case EWeaponType::HeavenSword:    return NewObject<USurvivorsCrossWeapon>(this);
		case EWeaponType::KingBible:
		case EWeaponType::UnholyVespers:  return NewObject<USurvivorsKingBibleWeapon>(this);
		case EWeaponType::FireWand:
		case EWeaponType::Hellfire:       return NewObject<USurvivorsFireWandWeapon>(this);
		case EWeaponType::SantaWater:
		case EWeaponType::LaBorra:        return NewObject<USurvivorsSantaWaterWeapon>(this);
		case EWeaponType::Runetracer:
		case EWeaponType::NoFuture:       return NewObject<USurvivorsRunetracerWeapon>(this);
		case EWeaponType::LightningRing:
		case EWeaponType::ThunderLoop:    return NewObject<USurvivorsLightningRingWeapon>(this);
		case EWeaponType::Pentagram:
		case EWeaponType::GorgeousMoon:   return NewObject<USurvivorsPentagramWeapon>(this);
		case EWeaponType::Peachone:       return NewObject<USurvivorsPeachoneWeapon>(this);
		case EWeaponType::EbonyWings:     return NewObject<USurvivorsEbonyWingsWeapon>(this);
		case EWeaponType::Vandalier:      return NewObject<USurvivorsVandalierWeapon>(this);
		case EWeaponType::Laurel:         return NewObject<USurvivorsLaurelWeapon>(this);
		default:
			checkNoEntry();
			return nullptr;
	}
}
