#include "Survivors/Logic/Weapons/SurvivorsWeaponComponent.h"

#include "Survivors/Logic/SurvivorsCollisionComponent.h"
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

void USurvivorsWeaponComponent::TickWeapons(float Dt)
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
}

void USurvivorsWeaponComponent::TickAllWeapons(float Dt)
{
	ensureMsgf(false, TEXT("TickAllWeapons() は非推奨。PhysicsStep 内の TickWeapons/ComputeAllWeaponHits/ApplyWeaponHits を使うこと。"));
	TickWeapons(Dt);
	// ApplyProjectileHits() 削除: HitFrame を迂回して旧挙動を残すため
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
		}
	}
}

void USurvivorsWeaponComponent::ComputeAllWeaponHits(USurvivorsCollisionComponent* CollComp, FSurvivorsHitFrame& HitFrame)
{
	if (!CollComp) return;

	for (int32 i = 0; i < WeaponInstances.Num(); ++i)
	{
		if (WeaponInstances[i])
		{
			WeaponInstances[i]->ComputeHits(CollComp, HitFrame);
		}
	}

	ComputeGroundZoneHits(CollComp, HitFrame);
	ComputeProjectileHits(CollComp, HitFrame);
}

void USurvivorsWeaponComponent::ComputeGroundZoneHits(USurvivorsCollisionComponent* CollComp, FSurvivorsHitFrame& HitFrame)
{
	if (!Game || !CollComp) return;

	for (int32 ZIdx = 0; ZIdx < GroundZones.Num(); ++ZIdx)
	{
		const FGroundZoneState& Z = GroundZones[ZIdx];

		TArray<const FSurvivorsTargetProxy*> Contacts;
		CollComp->QueryEnemyContacts(Z.Pos, Z.Radius, Contacts);

		for (const FSurvivorsTargetProxy* Proxy : Contacts)
		{
			// narrowphase
			if ((Z.Pos - Proxy->Pos).SizeSquared() > FMath::Square(Z.Radius + Proxy->Radius)) continue;

			// UniqueId 確認
			const int32 EIdx = Proxy->Ref.IndexAtBuildTime;
			if (!Game->Enemies.IsValidIndex(EIdx) || Game->Enemies[EIdx].UniqueId != Proxy->Ref.UniqueId) continue;
			const FEnemyState& E = Game->Enemies[EIdx];
			if (E.bPendingRemove) continue;

			// EnemyLastHitTime チェック
			const float* LastHit = Z.EnemyLastHitTime.Find(E.UniqueId);
			const float Now = Game->ElapsedTime;
			if (LastHit && (Now - *LastHit) < Z.HitCooldown) continue;

			FSurvivorsHitEvent Ev;
			Ev.Type = ESurvivorsHitType::GroundZoneDamage;
			Ev.Target = Proxy->Ref;
			Ev.Damage = Z.Damage;
			Ev.WeaponSlot = ZIdx;  // GroundZone インデックスを WeaponSlot に格納
			HitFrame.Events.Add(Ev);
		}
	}
}

void USurvivorsWeaponComponent::ComputeProjectileHits(USurvivorsCollisionComponent* CollComp, FSurvivorsHitFrame& HitFrame)
{
	if (!Game || !CollComp) return;

	for (int32 PIdx = 0; PIdx < Projectiles.Num(); ++PIdx)
	{
		const FProjectileState& P = Projectiles[PIdx];

		TArray<const FSurvivorsTargetProxy*> Contacts;
		CollComp->QueryEnemyContacts(P.Pos, P.Radius.Value, Contacts);

		for (const FSurvivorsTargetProxy* Proxy : Contacts)
		{
			// narrowphase
			if ((P.Pos - Proxy->Pos).SizeSquared() > FMath::Square(P.Radius.Value + Proxy->Radius)) continue;

			// UniqueId 確認
			const int32 EIdx = Proxy->Ref.IndexAtBuildTime;
			if (!Game->Enemies.IsValidIndex(EIdx) || Game->Enemies[EIdx].UniqueId != Proxy->Ref.UniqueId) continue;
			const FEnemyState& E = Game->Enemies[EIdx];
			if (E.bPendingRemove) continue;

			// piercing 弾: ヒット済みならスキップ
			if (P.HitEnemyIds.Contains(E.UniqueId)) continue;

			FSurvivorsHitEvent Ev;
			Ev.Type = ESurvivorsHitType::ProjectileDamage;
			Ev.Target = Proxy->Ref;
			Ev.Damage = P.Damage.Value;
			Ev.WeaponSlot = PIdx;  // Projectile インデックスを WeaponSlot に格納
			HitFrame.Events.Add(Ev);

			// 非 piercing 弾: 最初の1体にだけヒットして終了
			if (!P.bPiercing)
			{
				break;
			}
		}
	}
}

void USurvivorsWeaponComponent::ApplyWeaponHits(FSurvivorsHitFrame& HitFrame)
{
	if (!Game) return;

	// Projectile 削除フラグ管理
	TSet<int32> ProjectilesToRemove;

	for (const FSurvivorsHitEvent& Ev : HitFrame.Events)
	{
		if (Ev.Type == ESurvivorsHitType::WeaponAreaDamage
			|| Ev.Type == ESurvivorsHitType::GroundZoneDamage
			|| Ev.Type == ESurvivorsHitType::ProjectileDamage)
		{
			// UniqueId + IndexAtBuildTime で FEnemyState を特定
			const int32 EIdx = Ev.Target.IndexAtBuildTime;
			if (!Game->Enemies.IsValidIndex(EIdx)) continue;
			FEnemyState& E = Game->Enemies[EIdx];
			if (E.UniqueId != Ev.Target.UniqueId) continue;
			if (E.bPendingRemove)
			{
				// 死亡済み対象でも非 piercing 弾は消費する（次 tick への持ち越し防止）
				if (Ev.Type == ESurvivorsHitType::ProjectileDamage
					&& Projectiles.IsValidIndex(Ev.WeaponSlot)
					&& !Projectiles[Ev.WeaponSlot].bPiercing)
				{
					ProjectilesToRemove.Add(Ev.WeaponSlot);
				}
				continue;
			}

			// ダメージ適用
			E.HP -= Ev.Damage;

			// 最終ヒット時刻更新
			if (Ev.Type == ESurvivorsHitType::WeaponAreaDamage)
			{
				if (E.WeaponLastHitTime[Ev.WeaponSlot].Seconds < Game->ElapsedTime)
					E.WeaponLastHitTime[Ev.WeaponSlot] = FSurvivorsElapsedTime(Game->ElapsedTime);
			}
			else if (Ev.Type == ESurvivorsHitType::GroundZoneDamage)
			{
				if (GroundZones.IsValidIndex(Ev.WeaponSlot))
					GroundZones[Ev.WeaponSlot].EnemyLastHitTime.Add(E.UniqueId, Game->ElapsedTime);
			}
			else if (Ev.Type == ESurvivorsHitType::ProjectileDamage)
			{
				if (Projectiles.IsValidIndex(Ev.WeaponSlot))
				{
					FProjectileState& P = Projectiles[Ev.WeaponSlot];
					if (!P.bPiercing)
					{
						P.HitEnemyIds.Add(E.UniqueId);
						ProjectilesToRemove.Add(Ev.WeaponSlot);
					}
				}
			}

			// ノックバック
			if (Ev.KnockbackStrength > 0.f && Ev.KnockbackResistance < 1.f)
			{
				E.Pos += Ev.KnockbackDir * Ev.KnockbackStrength * (1.f - Ev.KnockbackResistance);
			}

			// 死亡判定
			if (E.HP <= 0.f)
			{
				E.bPendingRemove = true;
				Game->LastReward += Game->KillReward;
			}
		}
	}

	// 削除対象プロジェクタイルを逆順で削除
	TArray<int32> SortedRemoval = ProjectilesToRemove.Array();
	SortedRemoval.Sort([](int32 A, int32 B) { return A > B; });
	for (int32 Idx : SortedRemoval)
	{
		if (Projectiles.IsValidIndex(Idx))
			Projectiles.RemoveAt(Idx);
	}
}

void USurvivorsWeaponComponent::ApplyProjectileHits()
{
	ensureMsgf(false, TEXT("ApplyProjectileHits() は非推奨。PhysicsStep 内の ComputeProjectileHits/ApplyWeaponHits を使うこと。HitFrame/Finalize を迂回するため直接呼び出し禁止。"));
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
