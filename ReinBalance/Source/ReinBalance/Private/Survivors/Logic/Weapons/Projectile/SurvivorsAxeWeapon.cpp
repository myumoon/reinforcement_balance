#include "Survivors/Logic/Weapons/Projectile/SurvivorsAxeWeapon.h"

#include "Survivors/Logic/SurvivorsGame.h"
#include "Survivors/Logic/SurvivorsGameConstants.h"
#include "Survivors/Logic/Weapons/SurvivorsWeaponComponent.h"

void USurvivorsAxeWeapon::OnLevelChanged(FWeaponLevel NewLevel)
{
	CacheParams();
}

void USurvivorsAxeWeapon::CacheParams()
{
	const int32 Lv  = FMath::Clamp(Level.Value, 1, SurvivorsGameConstants::MaxWeaponLevel);
	const int32 Idx = Lv - 1;

	if (WeaponType == EWeaponType::DeathSpiral)
	{
		const SurvivorsGameConstants::FAxeParams& P = SurvivorsGameConstants::DeathSpiralTable[Idx];
		CachedDamage    = P.Damage;
		CachedCooldown  = P.Cooldown;
		CachedSpeed     = P.Speed;
		CachedArcHeight = P.ArcHeight;
	}
	else
	{
		const SurvivorsGameConstants::FAxeParams& P = SurvivorsGameConstants::AxeTable[Idx];
		CachedDamage    = P.Damage;
		CachedCooldown  = P.Cooldown;
		CachedSpeed     = P.Speed;
		CachedArcHeight = P.ArcHeight;
	}
}

void USurvivorsAxeWeapon::Tick(float Dt)
{
	if (!Game || !WeaponComp) return;

	const FPassiveEffects& PE = GetPassiveEffects();

	// --- 既存プロジェクタイルへ重力適用 ---
	// AngleRad.Value に重力加速度を格納して流用
	WeaponComp->UpdateProjectilesBySlot(SlotIdx, Dt, [](FProjectileState& P, float InDt) -> bool
	{
		// AngleRad.Value = 重力加速度 (負値)
		P.Vel.Y += P.AngleRad.Value * InDt;
		// LifeTime は TickProjectiles で更新済み（ここでは更新不要）
		return true;  // 削除しない（寿命切れは TickProjectiles で処理）
	});

	// --- クールダウン ---
	CooldownTimer.Value = FMath::Max(0.f, CooldownTimer.Value - Dt);
	if (!CooldownTimer.IsReady()) return;

	CooldownTimer = FCooldownSeconds(CachedCooldown * PE.CooldownMult);

	const float EffDamage    = CachedDamage    * PE.DamageMult;
	const float EffSpeed     = CachedSpeed     * PE.SpeedMult;
	const float EffArcHeight = CachedArcHeight * PE.AreaMult;

	// LifeTime: 上昇+下降の合計時間（頂点でY速度=0になる条件から逆算）
	// 初速Y = gravity * (LifeTime/2) → LifeTime = 2 * InitVelY / (-gravity)
	// gravity = -2 * ArcHeight / (halfT^2), halfT = ArcHeight / InitVelY
	// 簡略: LifeTime = 2 * ArcHeight / InitVelY, InitVelY = EffSpeed
	const float InitVelY = EffSpeed;
	const float HalfTime = EffArcHeight / InitVelY;
	const float LifeTime = HalfTime * 2.f;
	const float GravityY = -InitVelY / HalfTime;  // 負値

	// 横方向は最近傍敵の X 方向
	float HorizDir = 1.f;
	float MinDistSq = MAX_FLT;
	for (const FEnemyState& E : Game->Enemies)
	{
		if (E.bPendingRemove) continue;
		const float Dsq = (E.Pos - Game->PlayerPos).SizeSquared();
		if (Dsq < MinDistSq)
		{
			MinDistSq = Dsq;
			HorizDir  = (E.Pos.X >= Game->PlayerPos.X) ? 1.f : -1.f;
		}
	}

	FProjectileState P;
	P.Pos           = Game->PlayerPos;
	P.Vel           = FVector2D(HorizDir * EffSpeed * 0.4f, InitVelY);
	P.Radius        = FSimRadius(10.f);
	P.Damage        = FDamage(EffDamage);
	P.WeaponType    = WeaponType;
	P.WeaponSlotIdx = SlotIdx;
	P.LifeTime      = FProjectileLifeTime(LifeTime);
	P.bPiercing     = false;
	P.AngleRad      = FOrbitAngleRad(GravityY);  // 流用: 重力加速度を格納
	WeaponComp->SpawnProjectile(P);
}
