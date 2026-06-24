#include "Survivors/Game/Weapons/Projectile/SurvivorsWeaponKingBible.h"

#include "Survivors/Game/SurvivorsCollisionComponent.h"
#include "Survivors/Game/SurvivorsGame.h"
#include "Survivors/Logic/SurvivorsGameConstants.h"
#include "Survivors/Game/Weapons/SurvivorsWeaponComponent.h"

void USurvivorsWeaponKingBible::OnLevelChanged(FWeaponLevel NewLevel)
{
	CacheParams();
}

void USurvivorsWeaponKingBible::CacheParams()
{
	const int32 Lv  = FMath::Clamp(Level.Value, 1, SurvivorsGameConstants::MaxWeaponLevel);
	const int32 Idx = Lv - 1;

	if (WeaponType == EWeaponType::UnholyVespers)
	{
		const SurvivorsGameConstants::FKingBibleParams& P = SurvivorsGameConstants::UnholyVespersTable[Idx];
		CachedDamage            = P.Damage;
		CachedCooldown          = P.Cooldown;
		CachedDuration          = P.Duration;
		CachedOrbitRadius       = P.OrbitRadius;
		CachedAmount            = P.Amount;
		CachedRotSpeed          = P.RotSpeed;
		CachedKnockbackStrength = P.KnockbackStrength;
	}
	else
	{
		const SurvivorsGameConstants::FKingBibleParams& P = SurvivorsGameConstants::KingBibleTable[Idx];
		CachedDamage            = P.Damage;
		CachedCooldown          = P.Cooldown;
		CachedDuration          = P.Duration;
		CachedOrbitRadius       = P.OrbitRadius;
		CachedAmount            = P.Amount;
		CachedRotSpeed          = P.RotSpeed;
		CachedKnockbackStrength = P.KnockbackStrength;
	}

	RebuildOrbProjectiles();
}

void USurvivorsWeaponKingBible::ActivateOrbs(const FPassiveEffects& PE)
{
	bOrbsActive = true;
	ActiveTimer = CachedDuration * PE.DurationMult;

	// King Bible の次回発動は「基礎Cooldown + 基礎Duration」後。
	// Unholy Vespers はCooldownがDurationに依存しないため、基礎Cooldownのみで再発動する。
	const float CooldownInterval = (WeaponType == EWeaponType::UnholyVespers)
		? CachedCooldown * PE.CooldownMult
		: CachedCooldown * PE.CooldownMult + CachedDuration;
	CooldownTimer = FCooldownSeconds(FMath::Max(SurvivorsGameConstants::PhysicsDt, CooldownInterval));
	RebuildOrbProjectiles();
}

void USurvivorsWeaponKingBible::RebuildOrbProjectiles()
{
	// KingBible はオーブを Projectiles に登録しない。
	// ComputeHits() で直接当たり判定を行うため、Projectiles への追加は不要。
	// （登録すると ComputeProjectileHits と二重判定になる）
	// View 表示は GetOrbPositions() アクセサを使う。
	if (!Game) return;
	if (!bOrbsActive)
	{
		OrbPositions.Reset();
		return;
	}

	// オーブ位置キャッシュを再計算
	const FPassiveEffects& PE = GetPassiveEffects();
	const float EffRadius = CachedOrbitRadius * PE.AreaMult;
	const int32 EffAmount = FMath::Max(1, CachedAmount + static_cast<int32>(PE.ExtraAmount));

	OrbPositions.SetNum(EffAmount);
	for (int32 i = 0; i < EffAmount; ++i)
	{
		const float Angle  = MasterAngle + (TWO_PI * i / EffAmount);
		OrbPositions[i]    = Game->PlayerPos + FVector2D(FMath::Cos(Angle) * EffRadius, FMath::Sin(Angle) * EffRadius);
	}
}

void USurvivorsWeaponKingBible::Tick(float Dt)
{
	if (!Game) return;

	const FPassiveEffects& PE = GetPassiveEffects();
	CooldownTimer.Value = FMath::Max(0.f, CooldownTimer.Value - Dt);

	if (bOrbsActive)
	{
		ActiveTimer = FMath::Max(0.f, ActiveTimer - Dt);
		if (ActiveTimer <= 0.f)
		{
			bOrbsActive = false;
			OrbPositions.Reset();
		}
	}

	if (CooldownTimer.IsReady())
	{
		ActivateOrbs(PE);
	}

	if (!bOrbsActive) return;

	const float EffRadius   = CachedOrbitRadius * PE.AreaMult;
	const int32 EffAmount   = FMath::Max(1, CachedAmount + static_cast<int32>(PE.ExtraAmount));

	// 角度を更新してオーブ位置キャッシュを更新
	MasterAngle += CachedRotSpeed * PE.SpeedMult * Dt;

	OrbPositions.SetNum(EffAmount);
	for (int32 i = 0; i < EffAmount; ++i)
	{
		const float Angle  = MasterAngle + (TWO_PI * i / EffAmount);
		OrbPositions[i]    = Game->PlayerPos + FVector2D(FMath::Cos(Angle) * EffRadius, FMath::Sin(Angle) * EffRadius);
	}
}

void USurvivorsWeaponKingBible::ComputeHits(USurvivorsCollisionComponent* CollComp, FSurvivorsHitFrame& HitFrame)
{
	// KingBible は Projectiles を使わず ComputeHits のみでヒット判定を行う。
	// WeaponLastHitTime[SlotIdx] でヒット間隔（OrbHitInterval）を管理して毎フレームダメージを防ぐ。

	if (!Game || !CollComp || !bOrbsActive) return;

	const FPassiveEffects& PE = GetPassiveEffects();
	const float EffDamage  = CachedDamage * PE.DamageMult;
	static constexpr float OrbRadius = 10.f;

	// OrbPositions は Tick() で更新済み
	for (int32 OrbIdx = 0; OrbIdx < OrbPositions.Num(); ++OrbIdx)
	{
		const FVector2D& OrbPos = OrbPositions[OrbIdx];

		TArray<const FSurvivorsTargetProxy*> Contacts;
		CollComp->QueryEnemyContacts(OrbPos, OrbRadius, Contacts);

		for (const FSurvivorsTargetProxy* Proxy : Contacts)
		{
			if ((OrbPos - Proxy->Pos).SizeSquared() > FMath::Square(OrbRadius + Proxy->Radius)) continue;

			const int32 EIdx = Proxy->Ref.IndexAtBuildTime;
			if (!Game->Enemies.IsValidIndex(EIdx) || Game->Enemies[EIdx].UniqueId != Proxy->Ref.UniqueId) continue;
			const FEnemyState& E = Game->Enemies[EIdx];
			if (E.bPendingRemove) continue;

			// per-orb hit cooldown: Key = SlotIdx*10 + OrbIdx（同一 orb の同一敵への 1.7s 制限）
			// wiki: "same bible hits same enemy no more than every 1.7s"
			// 異なる orb は独立したクールダウンを持つため、Lv2 の2冊が同時に同じ敵を攻撃できる。
			const int32 OrbKey = SlotIdx * 10 + OrbIdx;
			const float* LastOrbHit = E.OrbHitTimes.Find(OrbKey);
			if (LastOrbHit && (Game->ElapsedTime - *LastOrbHit) < SurvivorsGameConstants::KingBibleOrbHitInterval)
				continue;

			FSurvivorsHitEvent Ev;
			Ev.Type              = ESurvivorsHitType::WeaponAreaDamage;
			Ev.Target            = Proxy->Ref;
			Ev.Damage            = EffDamage;
			Ev.WeaponSlot        = SlotIdx;
			Ev.OrbIdx            = OrbIdx;
			Ev.KnockbackDir      = (Proxy->Pos - Game->PlayerPos).GetSafeNormal();
			Ev.KnockbackStrength = CachedKnockbackStrength;
			HitFrame.Events.Add(Ev);
			break;  // 1 オーブにつき 1 ヒット/フレーム
		}
	}
}
