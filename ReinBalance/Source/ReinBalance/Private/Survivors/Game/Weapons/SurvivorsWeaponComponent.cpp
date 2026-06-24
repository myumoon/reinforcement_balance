#include "Survivors/Game/Weapons/SurvivorsWeaponComponent.h"

#include "Survivors/Game/SurvivorsCollisionComponent.h"
#include "Survivors/Game/SurvivorsGame.h"
#include "Survivors/SurvivorsGameConstants.h"
#include "Survivors/Game/SurvivorsGemComponent.h"
#include "Survivors/Game/Weapons/SurvivorsWeaponBase.h"
#include "Survivors/Game/Weapons/Projectile/SurvivorsWeaponGarlic.h"
#include "Survivors/Game/Weapons/Projectile/SurvivorsWeaponWhip.h"
#include "Survivors/Game/Weapons/Projectile/SurvivorsWeaponMagicWand.h"
#include "Survivors/Game/Weapons/Projectile/SurvivorsWeaponKnife.h"
#include "Survivors/Game/Weapons/Projectile/SurvivorsWeaponAxe.h"
#include "Survivors/Game/Weapons/Projectile/SurvivorsWeaponCross.h"
#include "Survivors/Game/Weapons/Projectile/SurvivorsWeaponKingBible.h"
#include "Survivors/Game/Weapons/Projectile/SurvivorsWeaponFireWand.h"
#include "Survivors/Game/Weapons/Projectile/SurvivorsWeaponSantaWater.h"
#include "Survivors/Game/Weapons/Projectile/SurvivorsWeaponRunetracer.h"
#include "Survivors/Game/Weapons/Projectile/SurvivorsWeaponLightningRing.h"
#include "Survivors/Game/Weapons/Projectile/SurvivorsWeaponPentagram.h"
#include "Survivors/Game/Weapons/Projectile/SurvivorsWeaponPeachone.h"
#include "Survivors/Game/Weapons/Projectile/SurvivorsWeaponEbonyWings.h"
#include "Survivors/Game/Weapons/Projectile/SurvivorsWeaponVandalier.h"
#include "Survivors/Game/Weapons/Projectile/SurvivorsWeaponLaurel.h"

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

		// King Bible / Orbit 系は Pos を WeaponBase::Tick で更新するためここでは移動しない
		P.Pos += P.Vel * Dt;
		P.Age += Dt;
		P.LifeTime.Tick(Dt);

		if (P.LifeTime.IsExpired())
		{
			// FireWand は LifeTime 切れ時に爆発が必要。bPendingExplosion フラグを立てて
			// 次の WeaponBase::Tick（TickWeapons 内で TickProjectiles より先に呼ばれる）
			// で爆発処理させる。既に爆発処理済みなら削除する。
			if (P.WeaponType == EWeaponType::FireWand || P.WeaponType == EWeaponType::Hellfire)
			{
				if (!P.bPendingExplosion)
				{
					// 爆発予約: このフレームは削除しない
					P.bPendingExplosion = true;
					// LifeTime を 0 に固定して次 Tick で削除されないようにする
					// （bPendingExplosion チェックで処理後に false→削除される）
					continue;
				}
				// bPendingExplosion == true: 既に処理済み（FireWandWeapon::Tick で爆発生成＆削除）
			}
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
		if (Z.bIsWarning)
		{
			Z.WarningTime = FMath::Max(0.f, Z.WarningTime - Dt);
			if (Z.WarningTime <= 0.f)
			{
				Z.bIsWarning = false;
				Z.EnemyLastHitTime.Empty();
			}
		}
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
		if (Z.bIsWarning) continue;

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
		FProjectileState& P = Projectiles[PIdx];

		// FireWand/Hellfire と Whip/BloodyTear は各武器の ComputeHits() で処理する。
		if (P.WeaponType == EWeaponType::FireWand || P.WeaponType == EWeaponType::Hellfire
			|| P.WeaponType == EWeaponType::Whip || P.WeaponType == EWeaponType::BloodyTear)
		{
			continue;
		}

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

			// Runetracer/NoFuture: wiki Hitbox Delay 0.5s で同一敵への再ヒットを制御。
			// 永続 HitEnemyIds ではなく projectile ごとの EnemyHitDelays で管理する。
			const bool bIsRunetracer = (P.WeaponType == EWeaponType::Runetracer
				|| P.WeaponType == EWeaponType::NoFuture);
			if (bIsRunetracer)
			{
				const float* LastHit = P.EnemyHitDelays.Find(E.UniqueId);
				if (LastHit && (Game->ElapsedTime - *LastHit) < SurvivorsGameConstants::RunetracerHitboxDelay)
					continue;
			}
			else
			{
				// 通常 projectile: ヒット済みなら永続スキップ
				if (P.HitEnemyIds.Contains(E.UniqueId)) continue;
			}

			FSurvivorsHitEvent Ev;
			Ev.Type = ESurvivorsHitType::ProjectileDamage;
			Ev.Target = Proxy->Ref;
			Ev.Damage = P.Damage.Value;
			Ev.WeaponSlot = PIdx;  // Projectile インデックスを WeaponSlot に格納
			// ノックバック: プロジェクタイルの進行方向へ押し出す
			if (P.KnockbackStrength > 0.f)
			{
				Ev.KnockbackDir      = P.Vel.GetSafeNormal();
				Ev.KnockbackStrength = P.KnockbackStrength;
			}
			HitFrame.Events.Add(Ev);

			if (bIsRunetracer)
			{
				// 時刻を記録して 0.5s 後に再ヒット可能にする
				P.EnemyHitDelays.Add(E.UniqueId, Game->ElapsedTime);
			}
			else
			{
				// HitEnemyIds に記録（同一弾では1回のみダメージ）
				P.HitEnemyIds.Add(E.UniqueId);
			}

			// 貫通数チェック: MaxPierceCount > 0 の場合、N 体命中で終了
			if (P.MaxPierceCount > 0 && P.HitEnemyIds.Num() >= P.MaxPierceCount)
			{
				break;
			}
			// 従来の非 piercing 弾: 最初の1体にだけヒットして終了
			else if (P.MaxPierceCount == 0 && !P.bPiercing)
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
				// 死亡済み対象でも消費数にカウントして貫通限界なら削除（持ち越し防止）
				if (Ev.Type == ESurvivorsHitType::ProjectileDamage
					&& Projectiles.IsValidIndex(Ev.WeaponSlot))
				{
					const FProjectileState& P = Projectiles[Ev.WeaponSlot];
					const bool bLimitedPierce = P.MaxPierceCount > 0 && P.HitEnemyIds.Num() >= P.MaxPierceCount;
					const bool bNonPiercing   = P.MaxPierceCount == 0 && !P.bPiercing;
					if (bLimitedPierce || bNonPiercing)
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
				// King Bible per-orb cooldown: OrbIdx >= 0 の場合のみ OrbHitTimes を更新
				if (Ev.OrbIdx >= 0)
				{
					const int32 OrbKey = Ev.WeaponSlot * 10 + Ev.OrbIdx;
					E.OrbHitTimes.Add(OrbKey, Game->ElapsedTime);
				}
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
					const bool bLimitedPierce = P.MaxPierceCount > 0 && P.HitEnemyIds.Num() >= P.MaxPierceCount;
					const bool bNonPiercing   = P.MaxPierceCount == 0 && !P.bPiercing;
					if (bLimitedPierce || bNonPiercing)
					{
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

void USurvivorsWeaponComponent::UpdateProjectilesBySlot(int32 InSlotIdx, float Dt,
	TFunctionRef<bool(FProjectileState&, float)> Callback)
{
	for (int32 i = Projectiles.Num() - 1; i >= 0; --i)
	{
		if (Projectiles[i].WeaponSlotIdx == InSlotIdx)
		{
			const bool bKeep = Callback(Projectiles[i], Dt);
			if (!bKeep)
				Projectiles.RemoveAt(i);
		}
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
		ZProj.bIsWarning   = Z.bIsWarning;
		ZProj.LifeTime     = FProjectileLifeTime(Z.LifeTime);
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

float USurvivorsWeaponComponent::GetProjectileBoxHalfWidth(int32 i) const
{
	return Projectiles.IsValidIndex(i) ? Projectiles[i].AngleRad.Value : 0.f;
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

bool USurvivorsWeaponComponent::IsGroundZoneWarning(int32 i) const
{
	return GroundZones.IsValidIndex(i) ? GroundZones[i].bIsWarning : false;
}

USurvivorsWeaponBase* USurvivorsWeaponComponent::GetWeaponInstance(int32 SlotIdx) const
{
	return WeaponInstances.IsValidIndex(SlotIdx) ? WeaponInstances[SlotIdx].Get() : nullptr;
}

int32 USurvivorsWeaponComponent::GetOrbitOrbCount() const
{
	int32 Total = 0;
	for (const auto& W : WeaponInstances)
		if (W) Total += W->GetOrbitOrbCount();
	return Total;
}

FVector2D USurvivorsWeaponComponent::GetOrbitOrbPos(int32 GlobalIdx) const
{
	int32 Offset = 0;
	for (const auto& W : WeaponInstances)
	{
		if (!W) continue;
		const int32 Count = W->GetOrbitOrbCount();
		if (GlobalIdx < Offset + Count)
			return W->GetOrbitOrbPos(GlobalIdx - Offset);
		Offset += Count;
	}
	return FVector2D::ZeroVector;
}

EWeaponType USurvivorsWeaponComponent::GetOrbitOrbWeaponType(int32 GlobalIdx) const
{
	int32 Offset = 0;
	for (const auto& W : WeaponInstances)
	{
		if (!W) continue;
		const int32 Count = W->GetOrbitOrbCount();
		if (GlobalIdx < Offset + Count)
			return W->GetWeaponType();
		Offset += Count;
	}
	return EWeaponType::None;
}

float USurvivorsWeaponComponent::GetOrbitOrbVisualRadius(int32 GlobalIdx) const
{
	int32 Offset = 0;
	for (const auto& W : WeaponInstances)
	{
		if (!W) continue;
		const int32 Count = W->GetOrbitOrbCount();
		if (GlobalIdx < Offset + Count)
			return W->GetOrbitOrbVisualRadius();
		Offset += Count;
	}
	return 0.f;
}

USurvivorsWeaponBase* USurvivorsWeaponComponent::CreateWeaponInstance(EWeaponType Type)
{
	switch (Type)
	{
		case EWeaponType::Garlic:
		case EWeaponType::SoulEater:      return NewObject<USurvivorsWeaponGarlic>(this);
		case EWeaponType::Whip:
		case EWeaponType::BloodyTear:     return NewObject<USurvivorsWeaponWhip>(this);
		case EWeaponType::MagicWand:
		case EWeaponType::HolyWand:       return NewObject<USurvivorsWeaponMagicWand>(this);
		case EWeaponType::Knife:
		case EWeaponType::ThousandEdge:   return NewObject<USurvivorsWeaponKnife>(this);
		case EWeaponType::Axe:
		case EWeaponType::DeathSpiral:    return NewObject<USurvivorsWeaponAxe>(this);
		case EWeaponType::Cross:
		case EWeaponType::HeavenSword:    return NewObject<USurvivorsWeaponCross>(this);
		case EWeaponType::KingBible:
		case EWeaponType::UnholyVespers:  return NewObject<USurvivorsWeaponKingBible>(this);
		case EWeaponType::FireWand:
		case EWeaponType::Hellfire:       return NewObject<USurvivorsWeaponFireWand>(this);
		case EWeaponType::SantaWater:
		case EWeaponType::LaBorra:        return NewObject<USurvivorsWeaponSantaWater>(this);
		case EWeaponType::Runetracer:
		case EWeaponType::NoFuture:       return NewObject<USurvivorsWeaponRunetracer>(this);
		case EWeaponType::LightningRing:
		case EWeaponType::ThunderLoop:    return NewObject<USurvivorsWeaponLightningRing>(this);
		case EWeaponType::Pentagram:
		case EWeaponType::GorgeousMoon:   return NewObject<USurvivorsWeaponPentagram>(this);
		case EWeaponType::Peachone:       return NewObject<USurvivorsWeaponPeachone>(this);
		case EWeaponType::EbonyWings:     return NewObject<USurvivorsWeaponEbonyWings>(this);
		case EWeaponType::Vandalier:      return NewObject<USurvivorsWeaponVandalier>(this);
		case EWeaponType::Laurel:         return NewObject<USurvivorsWeaponLaurel>(this);
		default:
			checkNoEntry();
			return nullptr;
	}
}
