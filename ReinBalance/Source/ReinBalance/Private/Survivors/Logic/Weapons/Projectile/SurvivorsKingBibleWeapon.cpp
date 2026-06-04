#include "Survivors/Logic/Weapons/Projectile/SurvivorsKingBibleWeapon.h"

#include "Survivors/Logic/SurvivorsCollisionComponent.h"
#include "Survivors/Logic/SurvivorsGame.h"
#include "Survivors/Logic/SurvivorsGameConstants.h"
#include "Survivors/Logic/Weapons/SurvivorsWeaponComponent.h"

void USurvivorsKingBibleWeapon::OnLevelChanged(FWeaponLevel NewLevel)
{
	CacheParams();
}

void USurvivorsKingBibleWeapon::CacheParams()
{
	const int32 Lv  = FMath::Clamp(Level.Value, 1, SurvivorsGameConstants::MaxWeaponLevel);
	const int32 Idx = Lv - 1;

	if (WeaponType == EWeaponType::UnholyVespers)
	{
		const SurvivorsGameConstants::FKingBibleParams& P = SurvivorsGameConstants::UnholyVespersTable[Idx];
		CachedDamage      = P.Damage;
		CachedOrbitRadius = P.OrbitRadius;
		CachedAmount      = P.Amount;
		CachedRotSpeed    = P.RotSpeed;
	}
	else
	{
		const SurvivorsGameConstants::FKingBibleParams& P = SurvivorsGameConstants::KingBibleTable[Idx];
		CachedDamage      = P.Damage;
		CachedOrbitRadius = P.OrbitRadius;
		CachedAmount      = P.Amount;
		CachedRotSpeed    = P.RotSpeed;
	}

	RebuildOrbProjectiles();
}

void USurvivorsKingBibleWeapon::RebuildOrbProjectiles()
{
	// 既存のオーブプロジェクタイルを削除して再構築
	// ※ WeaponComp が初期化前は何もしない
	if (!WeaponComp || !Game) return;

	// 既存スロットのプロジェクタイルを全削除
	WeaponComp->UpdateProjectilesBySlot(SlotIdx, 0.f, [](FProjectileState& P, float) -> bool
	{
		return false;  // 全削除
	});

	const FPassiveEffects& PE = GetPassiveEffects();
	const float EffRadius = CachedOrbitRadius * PE.AreaMult;
	const int32 EffAmount = CachedAmount + static_cast<int32>(PE.ExtraAmount);

	for (int32 i = 0; i < EffAmount; ++i)
	{
		const float Angle = MasterAngle + (TWO_PI * i / EffAmount);
		FProjectileState P;
		P.Pos           = Game->PlayerPos + FVector2D(FMath::Cos(Angle) * EffRadius, FMath::Sin(Angle) * EffRadius);
		P.Vel           = FVector2D::ZeroVector;
		P.Radius        = FSimRadius(10.f);
		P.Damage        = FDamage(CachedDamage * PE.DamageMult);
		P.WeaponType    = WeaponType;
		P.WeaponSlotIdx = SlotIdx;
		P.LifeTime      = FProjectileLifeTime(99999.f);  // 常時アクティブ（武器削除まで）
		P.bPiercing     = true;
		P.AngleRad      = FOrbitAngleRad(Angle);  // 現在の軌道角度
		WeaponComp->SpawnProjectile(P);
	}
}

void USurvivorsKingBibleWeapon::Tick(float Dt)
{
	if (!Game || !WeaponComp) return;

	const FPassiveEffects& PE = GetPassiveEffects();
	const float EffRotSpeed = CachedRotSpeed;
	const float EffRadius   = CachedOrbitRadius * PE.AreaMult;
	const int32 EffAmount   = CachedAmount + static_cast<int32>(PE.ExtraAmount);

	// 角度を更新し、プロジェクタイル位置を再計算
	MasterAngle += EffRotSpeed * Dt;

	int32 OrbIdx = 0;
	WeaponComp->UpdateProjectilesBySlot(SlotIdx, Dt, [&](FProjectileState& P, float InDt) -> bool
	{
		if (OrbIdx < EffAmount)
		{
			const float Angle = MasterAngle + (TWO_PI * OrbIdx / EffAmount);
			P.Pos     = Game->PlayerPos + FVector2D(FMath::Cos(Angle) * EffRadius, FMath::Sin(Angle) * EffRadius);
			P.AngleRad = FOrbitAngleRad(Angle);
			P.LifeTime = FProjectileLifeTime(99999.f);  // リセット（常時アクティブ）
		}
		++OrbIdx;
		return true;
	});

	// オーブ数が増えた場合（レベルアップ後）は追加生成
	const int32 CurrentCount = WeaponComp->GetProjectileCount();
	// 簡略: Count 管理は RebuildOrbProjectiles で行うため OnLevelChanged から呼ばれる
}

void USurvivorsKingBibleWeapon::ComputeHits(USurvivorsCollisionComponent* CollComp, FSurvivorsHitFrame& HitFrame)
{
	// KingBible のプロジェクタイルは常時アクティブなため WeaponComponent::ComputeProjectileHits が呼ぶ
	// ただし hit_interval（WeaponLastHitTime）による連打抑制が必要
	// ComputeProjectileHits ではそれがないため、ここで実装する

	if (!Game || !CollComp) return;

	const FPassiveEffects& PE = GetPassiveEffects();
	const float EffDamage = CachedDamage * PE.DamageMult;
	const float EffRadius = CachedOrbitRadius * PE.AreaMult;
	const int32 EffAmount = CachedAmount + static_cast<int32>(PE.ExtraAmount);

	for (int32 OrbIdx = 0; OrbIdx < EffAmount; ++OrbIdx)
	{
		const float Angle = MasterAngle + (TWO_PI * OrbIdx / EffAmount);
		const FVector2D OrbPos = Game->PlayerPos + FVector2D(FMath::Cos(Angle) * EffRadius, FMath::Sin(Angle) * EffRadius);
		const float OrbRadius = 10.f;

		TArray<const FSurvivorsTargetProxy*> Contacts;
		CollComp->QueryEnemyContacts(OrbPos, OrbRadius, Contacts);

		for (const FSurvivorsTargetProxy* Proxy : Contacts)
		{
			if ((OrbPos - Proxy->Pos).SizeSquared() > FMath::Square(OrbRadius + Proxy->Radius)) continue;

			const int32 EIdx = Proxy->Ref.IndexAtBuildTime;
			if (!Game->Enemies.IsValidIndex(EIdx) || Game->Enemies[EIdx].UniqueId != Proxy->Ref.UniqueId) continue;
			const FEnemyState& E = Game->Enemies[EIdx];
			if (E.bPendingRemove) continue;

			// ヒット間隔チェック（WeaponLastHitTime[SlotIdx] を流用）
			if (Game->ElapsedTime - E.WeaponLastHitTime[SlotIdx].Seconds < OrbHitInterval) continue;

			FSurvivorsHitEvent Ev;
			Ev.Type      = ESurvivorsHitType::WeaponAreaDamage;
			Ev.Target    = Proxy->Ref;
			Ev.Damage    = EffDamage;
			Ev.WeaponSlot = SlotIdx;
			HitFrame.Events.Add(Ev);
			break;  // 1 オーブにつき 1 ヒット/フレーム
		}
	}
}
