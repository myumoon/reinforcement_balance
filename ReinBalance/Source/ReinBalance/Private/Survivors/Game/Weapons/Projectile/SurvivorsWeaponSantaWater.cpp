#include "Survivors/Game/Weapons/Projectile/SurvivorsWeaponSantaWater.h"

#include "Survivors/Game/SurvivorsGame.h"
#include "Survivors/Game/SurvivorsGameConstants.h"
#include "Survivors/Game/Weapons/SurvivorsWeaponComponent.h"

void USurvivorsWeaponSantaWater::OnLevelChanged(FWeaponLevel NewLevel)
{
	CacheParams();
}

void USurvivorsWeaponSantaWater::CacheParams()
{
	const int32 Lv  = FMath::Clamp(Level.Value, 1, SurvivorsGameConstants::MaxWeaponLevel);
	const int32 Idx = Lv - 1;

	if (WeaponType == EWeaponType::LaBorra)
	{
		const SurvivorsGameConstants::FSantaWaterParams& P = SurvivorsGameConstants::LaBorraTable[Idx];
		CachedDamage   = P.Damage;
		CachedCooldown = P.Cooldown;
		CachedRadius   = P.Radius;
		CachedDuration = P.Duration;
		CachedAmount   = P.Amount;
	}
	else
	{
		const SurvivorsGameConstants::FSantaWaterParams& P = SurvivorsGameConstants::SantaWaterTable[Idx];
		CachedDamage   = P.Damage;
		CachedCooldown = P.Cooldown;
		CachedRadius   = P.Radius;
		CachedDuration = P.Duration;
		CachedAmount   = P.Amount;
	}
}

void USurvivorsWeaponSantaWater::Tick(float Dt)
{
	if (!Game || !WeaponComp) return;

	// 順次落下: 0.3s 間隔でキューから drop を処理（wiki: projectile interval = 0.3s）
	// while + += で overshoot を保持（大 Dt 時のキャッチアップ対応）
	if (PendingDropPositions.Num() > 0)
	{
		DropTimer -= Dt;
		while (DropTimer <= 0.f && PendingDropPositions.Num() > 0)
		{
			SpawnDrop(PendingDropPositions[0]);
			PendingDropPositions.RemoveAt(0);
			DropTimer += 0.30f;
		}
	}

	CooldownTimer.Value = FMath::Max(0.f, CooldownTimer.Value - Dt);
	if (!CooldownTimer.IsReady() || PendingDropPositions.Num() > 0) return;

	StartDropSequence();
}

void USurvivorsWeaponSantaWater::StartDropSequence()
{
	const FPassiveEffects& PE = GetPassiveEffects();
	CooldownTimer = FCooldownSeconds(CachedCooldown * PE.CooldownMult);

	BurstDamage   = CachedDamage   * PE.DamageMult;
	BurstRadius   = CachedRadius   * PE.AreaMult;
	BurstDuration = CachedDuration * PE.DurationMult;

	const int32 EffAmount = CachedAmount + static_cast<int32>(PE.ExtraAmount);

	PendingDropPositions.Empty();

	if (EffAmount < 4)
	{
		// Amount < 4: 1発目は最近傍敵、残りはプレイヤー周囲のランダム位置
		FVector2D NearestEnemyPos = Game->PlayerPos;
		float MinDistSq = MAX_FLT;
		for (const FEnemyState& E : Game->Enemies)
		{
			if (E.bPendingRemove) continue;
			const float Dsq = FVector2D::DistSquared(E.Pos, Game->PlayerPos);
			if (Dsq < MinDistSq) { MinDistSq = Dsq; NearestEnemyPos = E.Pos; }
		}
		PendingDropPositions.Add(NearestEnemyPos);

		for (int32 i = 1; i < EffAmount; ++i)
		{
			const float Angle = Game->RandStream.FRand() * TWO_PI;
			const float Dist  = Game->RandStream.FRandRange(0.f, SurvivorsGameConstants::SantaWaterRandomDropRadius);
			PendingDropPositions.Add(
				Game->PlayerPos + FVector2D(FMath::Cos(Angle), FMath::Sin(Angle)) * Dist);
		}
	}
	else
	{
		// Amount >= 4: プレイヤー周囲を 30°固定間隔で時計回りに配置（ランダム開始角度）
		// OBSERVED: santa_water.jpg 赤い円半径 ≈ 140u。各 drop は SantaWaterCircleDropStep(30°) ごとにずれる。
		const float StartAngle = Game->RandStream.FRand() * TWO_PI;
		for (int32 i = 0; i < EffAmount; ++i)
		{
			const float Angle = StartAngle + SurvivorsGameConstants::SantaWaterCircleDropStep * i;
			PendingDropPositions.Add(
				Game->PlayerPos + FVector2D(FMath::Cos(Angle), FMath::Sin(Angle)) * SurvivorsGameConstants::SantaWaterCircleRadius);
		}
	}

	// 最初の drop を即時処理
	if (PendingDropPositions.Num() > 0)
	{
		SpawnDrop(PendingDropPositions[0]);
		PendingDropPositions.RemoveAt(0);
		DropTimer = (PendingDropPositions.Num() > 0) ? 0.30f : 0.f;
	}
}

void USurvivorsWeaponSantaWater::SpawnDrop(FVector2D DropPos)
{
	FGroundZoneState Z;
	Z.Pos           = DropPos;
	Z.Radius        = BurstRadius;
	Z.Damage        = BurstDamage;
	Z.LifeTime      = SurvivorsGameConstants::SantaWaterWarningTime + BurstDuration;
	Z.WarningTime   = SurvivorsGameConstants::SantaWaterWarningTime;
	Z.HitCooldown   = 0.5f;
	Z.WeaponSlotIdx = SlotIdx;
	Z.WeaponType    = WeaponType;
	Z.bIsWarning    = true;
	WeaponComp->SpawnGroundZone(Z);
}
