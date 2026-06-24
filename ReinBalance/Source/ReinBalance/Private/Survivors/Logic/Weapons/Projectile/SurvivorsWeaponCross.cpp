#include "Survivors/Logic/Weapons/Projectile/SurvivorsWeaponCross.h"

#include "Survivors/Logic/SurvivorsGame.h"
#include "Survivors/Logic/SurvivorsGameConstants.h"
#include "Survivors/Logic/Weapons/SurvivorsWeaponComponent.h"

void USurvivorsWeaponCross::OnLevelChanged(FWeaponLevel NewLevel)
{
	CacheParams();
}

void USurvivorsWeaponCross::CacheParams()
{
	const int32 Lv  = FMath::Clamp(Level.Value, 1, SurvivorsGameConstants::MaxWeaponLevel);
	const int32 Idx = Lv - 1;

	if (WeaponType == EWeaponType::HeavenSword)
	{
		const SurvivorsGameConstants::FCrossParams& P = SurvivorsGameConstants::HeavenSwordTable[Idx];
		CachedDamage            = P.Damage;
		CachedCooldown          = P.Cooldown;
		CachedSpeed             = P.Speed;
		CachedRadius            = P.Radius;
		CachedAmount            = P.Amount;
		CachedKnockbackStrength = P.KnockbackStrength;
	}
	else
	{
		const SurvivorsGameConstants::FCrossParams& P = SurvivorsGameConstants::CrossTable[Idx];
		CachedDamage            = P.Damage;
		CachedCooldown          = P.Cooldown;
		CachedSpeed             = P.Speed;
		CachedRadius            = P.Radius;
		CachedAmount            = P.Amount;
		CachedKnockbackStrength = P.KnockbackStrength;
	}
}

void USurvivorsWeaponCross::Tick(float Dt)
{
	if (!Game || !WeaponComp) return;

	// --- 既存プロジェクタイルの折り返し処理 ---
	// AngleRad.Value に折り返しまでの時間を格納
	WeaponComp->UpdateProjectilesBySlot(SlotIdx, Dt, [](FProjectileState& P, float InDt) -> bool
	{
		const float ReverseTime = FMath::Max(P.AngleRad.Value, SurvivorsGameConstants::PhysicsDt);
		if (!P.bHasReversed && P.Age >= ReverseTime)
		{
			P.Vel        = -P.Vel;
			P.bHasReversed = true;
		}
		return true;
	});

	// --- 順次発射バースト（wiki: 0.1s projectile interval）---
	if (PendingCrossShots > 0)
	{
		CrossBurstTimer -= Dt;
		while (PendingCrossShots > 0 && CrossBurstTimer <= 0.f)
		{
			SpawnCrossShot();
			--PendingCrossShots;
			CrossBurstTimer += SurvivorsGameConstants::CrossProjectileInterval;
		}
	}

	// --- クールダウン ---
	CooldownTimer.Value = FMath::Max(0.f, CooldownTimer.Value - Dt);
	if (!CooldownTimer.IsReady() || PendingCrossShots > 0) return;

	StartBurst();
}

void USurvivorsWeaponCross::StartBurst()
{
	const FPassiveEffects& PE = GetPassiveEffects();
	CooldownTimer = FCooldownSeconds(CachedCooldown * PE.CooldownMult);

	BurstDamage      = CachedDamage * PE.DamageMult;
	BurstSpeed       = CachedSpeed  * PE.SpeedMult;
	BurstRadius      = CachedRadius * PE.AreaMult;
	BurstLifeTime    = 6.0f * PE.DurationMult;
	// 折り返しまでの時間 = 距離 / 速度（距離固定で speed が変わっても往路距離が伸びない設計）。
	// OBSERVED: cross_bullet2.mp4 frame 80-110, 約75u。CrossReverseDistance = 75u。
	BurstReverseTime = SurvivorsGameConstants::CrossReverseDistance / FMath::Max(BurstSpeed, 1.f);
	BurstKnockback   = CachedKnockbackStrength;

	PendingCrossShots = FMath::Max(1, CachedAmount + static_cast<int32>(PE.ExtraAmount));
	CrossBurstTimer   = 0.f;

	if (PendingCrossShots > 0)
	{
		SpawnCrossShot();
		--PendingCrossShots;
		CrossBurstTimer = (PendingCrossShots > 0) ? SurvivorsGameConstants::CrossProjectileInterval : 0.f;
	}
}

void USurvivorsWeaponCross::SpawnCrossShot()
{
	// 発射時点で画面内最近傍敵を再評価（wiki: "each cross aims at the nearest enemy at firing time"）
	// fan spread はなく、各弾が独立して最近傍敵を狙う。画面外の敵は対象外。
	FVector2D Dir = FVector2D::ZeroVector;
	float MinDistSq = MAX_FLT;
	for (const FEnemyState& E : Game->Enemies)
	{
		if (E.bPendingRemove) continue;
		if (!Game->IsOnScreen(E.Pos)) continue;
		const float Dsq = (E.Pos - Game->PlayerPos).SizeSquared();
		if (Dsq < MinDistSq)
		{
			MinDistSq = Dsq;
			Dir       = (E.Pos - Game->PlayerPos).GetSafeNormal();
		}
	}
	if (Dir.IsNearlyZero())
	{
		const float Angle = Game->RandStream.FRand() * TWO_PI;
		Dir = FVector2D(FMath::Cos(Angle), FMath::Sin(Angle));
	}

	FProjectileState P;
	P.Pos               = Game->PlayerPos;
	P.Vel               = Dir * BurstSpeed;
	P.Radius            = FSimRadius(BurstRadius);
	P.Damage            = FDamage(BurstDamage);
	P.WeaponType        = WeaponType;
	P.WeaponSlotIdx     = SlotIdx;
	P.LifeTime          = FProjectileLifeTime(BurstLifeTime);
	P.Age               = 0.f;
	P.bPiercing         = true;    // AoE: 無限貫通
	P.bHasReversed      = false;
	P.AngleRad          = FOrbitAngleRad(BurstReverseTime);  // 流用: 折り返しまでの時間
	P.KnockbackStrength = BurstKnockback;
	WeaponComp->SpawnProjectile(P);
}
