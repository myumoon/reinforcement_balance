#include "Survivors/Logic/Weapons/Projectile/SurvivorsPeachoneWeapon.h"

#include "Survivors/Logic/SurvivorsCollisionComponent.h"
#include "Survivors/Logic/SurvivorsGame.h"
#include "Survivors/Logic/SurvivorsGameConstants.h"
#include "Survivors/Logic/Weapons/SurvivorsWeaponComponent.h"

void USurvivorsPeachoneWeapon::OnLevelChanged(FWeaponLevel NewLevel)
{
	CacheParams();
}

void USurvivorsPeachoneWeapon::CacheParams()
{
	const int32 Lv  = FMath::Clamp(Level.Value, 1, SurvivorsGameConstants::MaxWeaponLevel);
	const int32 Idx = Lv - 1;

	const SurvivorsGameConstants::FPeachoneParams& P = SurvivorsGameConstants::PeachoneTable[Idx];
	CachedDamage      = P.Damage;
	CachedCooldown    = P.Cooldown;
	CachedOrbitRadius = P.OrbitRadius;
	CachedBombRadius  = P.BombRadius;
	CachedAmount      = P.Amount;
}

void USurvivorsPeachoneWeapon::UpdateOrbitPos()
{
	if (!Game) return;
	const FPassiveEffects& PE = GetPassiveEffects();
	const float EffRadius = CachedOrbitRadius * PE.AreaMult;
	CurrentOrbitPos = Game->PlayerPos + FVector2D(
		FMath::Cos(OrbitAngle + PhaseOff) * EffRadius,
		FMath::Sin(OrbitAngle + PhaseOff) * EffRadius);
}

void USurvivorsPeachoneWeapon::Tick(float Dt)
{
	if (!Game) return;

	const FPassiveEffects& PE = GetPassiveEffects();

	// 軌道角度を更新（Speed は target zone の周回速度に影響）
	const float RotSpeed = 3.0f * PE.SpeedMult;
	OrbitAngle += RotDir * RotSpeed * Dt;
	UpdateOrbitPos();

	// 砲撃バースト: 0.025s 間隔で target zone 内のランダム位置へ弾を発射（wiki: projectile interval）
	if (PendingBombShots > 0 && WeaponComp)
	{
		BombShotTimer -= Dt;
		while (PendingBombShots > 0 && BombShotTimer <= 0.f)
		{
			SpawnBombShot();
			--PendingBombShots;
			BombShotTimer += SurvivorsGameConstants::PeachoneProjectileInterval;
		}
	}

	// クールダウン
	CooldownTimer.Value = FMath::Max(0.f, CooldownTimer.Value - Dt);
	if (CooldownTimer.IsReady() && PendingBombShots == 0)
	{
		StartBombing();
	}
}

void USurvivorsPeachoneWeapon::StartBombing()
{
	const FPassiveEffects& PE = GetPassiveEffects();
	CooldownTimer = FCooldownSeconds(CachedCooldown * PE.CooldownMult);

	// ダメージはセット数で割って1 activation あたりの期待ダメージを wiki 値に近づける。
	// 各弾がランダム散布されるため1敵への命中数は確率的に ~1/set となる。
	BurstDamage       = CachedDamage * PE.DamageMult
		/ static_cast<float>(SurvivorsGameConstants::PeachoneSetsPerActivation);
	BurstImpactRadius = 10.f * PE.AreaMult;
	BurstBombRadius   = CachedBombRadius * PE.AreaMult;

	// wiki: Amount × SetsPerActivation 発を rapid fire
	// Duration パッシブによるセット数スケールは TODO（現在は固定 4 sets）
	const int32 EffAmount = CachedAmount + static_cast<int32>(PE.ExtraAmount);
	PendingBombShots = EffAmount * SurvivorsGameConstants::PeachoneSetsPerActivation;
	BombShotTimer    = 0.f;

	if (PendingBombShots > 0)
	{
		SpawnBombShot();
		--PendingBombShots;
		BombShotTimer = SurvivorsGameConstants::PeachoneProjectileInterval;
	}
}

void USurvivorsPeachoneWeapon::SpawnBombShot()
{
	if (!Game || !WeaponComp) return;

	// target zone 内のランダム位置（uniform in circle: sqrt で面積均一サンプリング）
	// OBSERVED: "bombard random points inside the current circular target zone" (weapon_peachone.md)
	// 再現性のため FMath::FRand 系ではなく Game->RandStream を使用する
	const float Angle = Game->RandStream.FRand() * 2.f * UE_PI;
	const float Dist  = FMath::Sqrt(Game->RandStream.FRand()) * BurstBombRadius;
	const FVector2D ImpactPos = CurrentOrbitPos + FVector2D(FMath::Cos(Angle), FMath::Sin(Angle)) * Dist;

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
