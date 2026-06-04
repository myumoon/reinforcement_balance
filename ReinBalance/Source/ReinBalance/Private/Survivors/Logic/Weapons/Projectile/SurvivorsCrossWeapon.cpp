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
		CachedDamage   = P.Damage;
		CachedCooldown = P.Cooldown;
		CachedSpeed    = P.Speed;
	}
	else
	{
		const SurvivorsGameConstants::FCrossParams& P = SurvivorsGameConstants::CrossTable[Idx];
		CachedDamage   = P.Damage;
		CachedCooldown = P.Cooldown;
		CachedSpeed    = P.Speed;
	}
}

void USurvivorsCrossWeapon::Tick(float Dt)
{
	if (!Game || !WeaponComp) return;

	const FPassiveEffects& PE = GetPassiveEffects();

	// --- 既存プロジェクタイルの折り返し処理 ---
	// AngleRad.Value に発射時の残り LifeTime（折り返しフラグ用）を格納
	// bPiercing が false になったら折り返し済み
	WeaponComp->UpdateProjectilesBySlot(SlotIdx, Dt, [](FProjectileState& P, float InDt) -> bool
	{
		// AngleRad.Value = 初期 LifeTime（秒）
		const float InitLifeTime = P.AngleRad.Value;
		const float HalfTime     = InitLifeTime * 0.5f;
		const float Elapsed      = InitLifeTime - P.LifeTime.Seconds;

		// LifeTime の半分を過ぎたら折り返す（1度だけ）
		if (!P.bPiercing && Elapsed >= HalfTime)
		{
			P.Vel     = -P.Vel;
			P.bPiercing = true;  // 折り返し済みフラグ（bPiercing を流用）
		}
		return true;
	});

	// --- クールダウン ---
	CooldownTimer.Value = FMath::Max(0.f, CooldownTimer.Value - Dt);
	if (!CooldownTimer.IsReady()) return;

	CooldownTimer = FCooldownSeconds(CachedCooldown * PE.CooldownMult);

	const float EffDamage  = CachedDamage * PE.DamageMult;
	const float EffSpeed   = CachedSpeed  * PE.SpeedMult;
	const float EffLifeTime = 2.0f * PE.DurationMult;

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

	FProjectileState P;
	P.Pos           = Game->PlayerPos;
	P.Vel           = Dir * EffSpeed;
	P.Radius        = FSimRadius(12.f);
	P.Damage        = FDamage(EffDamage);
	P.WeaponType    = WeaponType;
	P.WeaponSlotIdx = SlotIdx;
	P.LifeTime      = FProjectileLifeTime(EffLifeTime);
	P.bPiercing     = false;  // false = まだ折り返していない
	P.AngleRad      = FOrbitAngleRad(EffLifeTime);  // 流用: 初期 LifeTime を格納
	WeaponComp->SpawnProjectile(P);
}
