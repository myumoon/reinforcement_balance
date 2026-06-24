#include "Survivors/Game/Weapons/Projectile/SurvivorsWeaponVandalier.h"

#include "Survivors/Game/SurvivorsCollisionComponent.h"
#include "Survivors/Game/SurvivorsGame.h"
#include "Survivors/Game/SurvivorsGameConstants.h"
#include "Survivors/Game/Weapons/SurvivorsWeaponComponent.h"

void USurvivorsWeaponVandalier::OnLevelChanged(FWeaponLevel NewLevel)
{
	CacheParams();
}

void USurvivorsWeaponVandalier::CacheParams()
{
	const int32 Lv  = FMath::Clamp(Level.Value, 1, SurvivorsGameConstants::MaxWeaponLevel);
	const int32 Idx = Lv - 1;

	// Vandalier は Peachone + EbonyWings 統合のため約 1.5 倍強化
	const SurvivorsGameConstants::FPeachoneParams& P = SurvivorsGameConstants::PeachoneTable[Idx];
	CachedDamage           = P.Damage * 1.5f;
	CachedCooldown         = P.Cooldown * 0.8f;
	CachedOrbitRadius      = P.OrbitRadius + 10.f;
	CachedOrbitRotSpeed    = P.OrbitRotSpeed;
	CachedTargetZoneRadius = P.TargetZoneRadius + 10.f;
	CachedImpactRadius     = P.ImpactRadius;
	CachedAmount           = P.Amount;
}

FVector2D USurvivorsWeaponVandalier::GetOrbitOrbPos(int32 OrbIdx) const
{
	if (!Game || OrbIdx < 0 || OrbIdx >= 2) return FVector2D::ZeroVector;
	return Game->PlayerPos + FVector2D(
		FMath::Cos(OrbitAngle[OrbIdx]) * CachedOrbitRadius,
		FMath::Sin(OrbitAngle[OrbIdx]) * CachedOrbitRadius);
}

void USurvivorsWeaponVandalier::Tick(float Dt)
{
	if (!Game) return;

	const FPassiveEffects& PE = GetPassiveEffects();
	const float RotSpeed = CachedOrbitRotSpeed * PE.SpeedMult;
	OrbitAngle[0] += RotSpeed * Dt;
	OrbitAngle[1] -= RotSpeed * Dt;  // 逆回転

	// 各 zone の砲撃バーストを処理
	if (WeaponComp)
	{
		for (int32 OrbIdx = 0; OrbIdx < 2; ++OrbIdx)
		{
			if (PendingBombShots[OrbIdx] > 0)
			{
				BombShotTimer[OrbIdx] -= Dt;
				while (PendingBombShots[OrbIdx] > 0 && BombShotTimer[OrbIdx] <= 0.f)
				{
					SpawnBombShot(OrbIdx);
					--PendingBombShots[OrbIdx];
					BombShotTimer[OrbIdx] += SurvivorsGameConstants::PeachoneProjectileInterval;
				}
			}
		}
	}

	// 両 zone の砲撃が終わったらクールダウン
	const bool bBurstDone = (PendingBombShots[0] == 0 && PendingBombShots[1] == 0);
	CooldownTimer.Value = FMath::Max(0.f, CooldownTimer.Value - Dt);
	if (CooldownTimer.IsReady() && bBurstDone)
	{
		StartBombing();
	}
}

void USurvivorsWeaponVandalier::StartBombing()
{
	const FPassiveEffects& PE = GetPassiveEffects();
	CooldownTimer = FCooldownSeconds(CachedCooldown * PE.CooldownMult);

	// Peachone 同様、セット数で割って期待 DPS を維持する（2 zone 分で × 2 倍になるため zone 数でも割る）
	BurstDamage           = CachedDamage * PE.DamageMult
		/ static_cast<float>(SurvivorsGameConstants::PeachoneSetsPerActivation) / 2.f;
	BurstImpactRadius     = CachedImpactRadius * PE.AreaMult;
	BurstTargetZoneRadius = CachedTargetZoneRadius;

	const int32 EffAmount = CachedAmount + static_cast<int32>(PE.ExtraAmount);
	const int32 TotalShots = EffAmount * SurvivorsGameConstants::PeachoneSetsPerActivation;

	for (int32 OrbIdx = 0; OrbIdx < 2; ++OrbIdx)
	{
		PendingBombShots[OrbIdx] = TotalShots;
		BombShotTimer[OrbIdx]    = 0.f;

		if (PendingBombShots[OrbIdx] > 0)
		{
			SpawnBombShot(OrbIdx);
			--PendingBombShots[OrbIdx];
			BombShotTimer[OrbIdx] = SurvivorsGameConstants::PeachoneProjectileInterval;
		}
	}
}

void USurvivorsWeaponVandalier::SpawnBombShot(int32 OrbIdx)
{
	if (!Game || !WeaponComp) return;

	const FVector2D ZoneCenter = GetOrbitOrbPos(OrbIdx);

	// target zone 内のランダム位置（uniform in circle）再現性のため RandStream を使用
	const float Angle = Game->RandStream.FRand() * 2.f * UE_PI;
	const float Dist  = FMath::Sqrt(Game->RandStream.FRand()) * BurstTargetZoneRadius;
	const FVector2D ImpactPos = ZoneCenter + FVector2D(FMath::Cos(Angle), FMath::Sin(Angle)) * Dist;

	FProjectileState P;
	P.Pos               = ImpactPos;
	P.Vel               = FVector2D::ZeroVector;
	P.Radius            = FSimRadius(BurstImpactRadius);
	P.Damage            = FDamage(BurstDamage);
	P.WeaponType        = WeaponType;
	P.WeaponSlotIdx     = SlotIdx;
	P.LifeTime          = FProjectileLifeTime(0.1f);
	P.bPiercing         = true;
	P.MaxPierceCount    = 100;
	P.KnockbackStrength = SurvivorsGameConstants::KnockbackSim_2;
	WeaponComp->SpawnProjectile(P);
}
