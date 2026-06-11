#include "Survivors/Logic/Weapons/Projectile/SurvivorsCrossWeapon.h"

#include "Survivors/Logic/SurvivorsGame.h"
#include "Survivors/Logic/SurvivorsGameConstants.h"
#include "Survivors/Logic/Weapons/SurvivorsWeaponComponent.h"

void USurvivorsCrossWeapon::OnLevelChanged(FWeaponLevel NewLevel)
{
	CacheParams();
}

void USurvivorsCrossWeapon::CacheParams()
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

void USurvivorsCrossWeapon::Tick(float Dt)
{
	if (!Game || !WeaponComp) return;

	const FPassiveEffects& PE = GetPassiveEffects();

	// --- 既存プロジェクタイルの折り返し処理 ---
	// AngleRad.Value に折り返しまでの時間を格納
	// bHasReversed で折り返し済みを管理（bPiercing は共通の ApplyWeaponHits 削除判定に使用するため流用しない）
	WeaponComp->UpdateProjectilesBySlot(SlotIdx, Dt, [](FProjectileState& P, float InDt) -> bool
	{
		const float ReverseTime = FMath::Max(P.AngleRad.Value, SurvivorsGameConstants::PhysicsDt);

		// 発射後の固定時間で折り返し、その後は寿命まで画面端方向へ抜ける
		if (!P.bHasReversed && P.Age >= ReverseTime)
		{
			P.Vel        = -P.Vel;
			P.bHasReversed = true;  // 折り返し済みフラグ
		}
		return true;
	});

	// --- クールダウン ---
	CooldownTimer.Value = FMath::Max(0.f, CooldownTimer.Value - Dt);
	if (!CooldownTimer.IsReady()) return;

	CooldownTimer = FCooldownSeconds(CachedCooldown * PE.CooldownMult);

	const float EffDamage  = CachedDamage * PE.DamageMult;
	const float EffSpeed   = CachedSpeed  * PE.SpeedMult;
	const float EffLifeTime = 6.0f * PE.DurationMult;
	const float ReverseTime = 0.75f * PE.DurationMult;
	const float EffRadius  = CachedRadius * PE.AreaMult;
	const int32 EffAmount  = FMath::Max(1, CachedAmount + static_cast<int32>(PE.ExtraAmount));

	// 最近傍敵の方向へ発射
	FVector2D Dir = FVector2D(1.f, 0.f);
	float MinDistSq = MAX_FLT;
	for (const FEnemyState& E : Game->Enemies)
	{
		if (E.bPendingRemove) continue;
		const float Dsq = (E.Pos - Game->PlayerPos).SizeSquared();
		if (Dsq < MinDistSq)
		{
			MinDistSq = Dsq;
			Dir = (E.Pos - Game->PlayerPos).GetSafeNormal();
		}
	}

	const float BaseAngle = FMath::Atan2(Dir.Y, Dir.X);
	for (int32 i = 0; i < EffAmount; ++i)
	{
		const float Spread = (EffAmount <= 1)
			? 0.f
			: FMath::DegreesToRadians(12.f) * (static_cast<float>(i) - 0.5f * static_cast<float>(EffAmount - 1));
		const float Angle = BaseAngle + Spread;
		const FVector2D ShotDir(FMath::Cos(Angle), FMath::Sin(Angle));

		FProjectileState P;
		P.Pos               = Game->PlayerPos;
		P.Vel               = ShotDir * EffSpeed;
		P.Radius            = FSimRadius(EffRadius);
		P.Damage            = FDamage(EffDamage);
		P.WeaponType        = WeaponType;
		P.WeaponSlotIdx     = SlotIdx;
		P.LifeTime          = FProjectileLifeTime(EffLifeTime);
		P.Age               = 0.f;
		P.bPiercing         = true;   // AoE: 無限貫通
		P.bHasReversed      = false;
		P.AngleRad          = FOrbitAngleRad(ReverseTime);  // 流用: 折り返しまでの時間
		P.KnockbackStrength = CachedKnockbackStrength;
		WeaponComp->SpawnProjectile(P);
	}
}
