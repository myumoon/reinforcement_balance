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
		CachedAmount    = P.Amount;
		CachedPierce    = P.Pierce;
	}
	else
	{
		const SurvivorsGameConstants::FAxeParams& P = SurvivorsGameConstants::AxeTable[Idx];
		CachedDamage    = P.Damage;
		CachedCooldown  = P.Cooldown;
		CachedSpeed     = P.Speed;
		CachedArcHeight = P.ArcHeight;
		CachedAmount    = P.Amount;
		CachedPierce    = P.Pierce;
	}
}

void USurvivorsAxeWeapon::Tick(float Dt)
{
	if (!Game || !WeaponComp) return;

	// --- 既存プロジェクタイルへ重力適用 ---
	// AngleRad.Value に重力加速度を格納して流用
	const float FieldLimit  = Game->FieldHalfSize + 20.f;
	const float FieldBottom = -FieldLimit;
	WeaponComp->UpdateProjectilesBySlot(SlotIdx, Dt, [FieldBottom, FieldLimit](FProjectileState& P, float InDt) -> bool
	{
		P.Vel.Y += P.AngleRad.Value * InDt;
		if (P.WeaponType == EWeaponType::DeathSpiral)
			return FMath::Abs(P.Pos.X) <= FieldLimit && FMath::Abs(P.Pos.Y) <= FieldLimit;
		return P.Pos.Y > FieldBottom;
	});

	// --- DeathSpiral: 全弾同時発射（仕様変更なし）---
	if (WeaponType == EWeaponType::DeathSpiral)
	{
		CooldownTimer.Value = FMath::Max(0.f, CooldownTimer.Value - Dt);
		if (!CooldownTimer.IsReady()) return;

		const FPassiveEffects& PE = GetPassiveEffects();
		CooldownTimer = FCooldownSeconds(CachedCooldown * PE.CooldownMult);

		const float EffDamage = CachedDamage * PE.DamageMult;
		const float EffSpeed  = CachedSpeed  * PE.SpeedMult;
		const int32 EffAmount = FMath::Max(1, CachedAmount + static_cast<int32>(PE.ExtraAmount));

		for (int32 i = 0; i < EffAmount; ++i)
		{
			const float Angle = TWO_PI * static_cast<float>(i) / static_cast<float>(EffAmount);
			FProjectileState P;
			P.Pos               = Game->PlayerPos;
			P.Vel               = FVector2D(FMath::Cos(Angle), FMath::Sin(Angle)) * EffSpeed;
			P.Radius            = FSimRadius(12.f * PE.AreaMult);
			P.Damage            = FDamage(EffDamage);
			P.WeaponType        = WeaponType;
			P.WeaponSlotIdx     = SlotIdx;
			P.LifeTime          = FProjectileLifeTime(20.f);
			P.bPiercing         = true;
			P.MaxPierceCount    = CachedPierce;
			P.KnockbackStrength = SurvivorsGameConstants::KnockbackSim_1;
			P.AngleRad          = FOrbitAngleRad(0.f);
			WeaponComp->SpawnProjectile(P);
		}
		return;
	}

	// --- Axe: 順次発射バースト（wiki/OBSERVED: ~0.2s interval）---
	if (PendingAxeShots > 0)
	{
		AxeBurstTimer -= Dt;
		while (PendingAxeShots > 0 && AxeBurstTimer <= 0.f)
		{
			SpawnAxeShot();
			--PendingAxeShots;
			AxeBurstTimer += SurvivorsGameConstants::AxeProjectileInterval;
		}
	}

	CooldownTimer.Value = FMath::Max(0.f, CooldownTimer.Value - Dt);
	if (!CooldownTimer.IsReady() || PendingAxeShots > 0) return;

	StartBurst();
}

void USurvivorsAxeWeapon::StartBurst()
{
	const FPassiveEffects& PE = GetPassiveEffects();
	CooldownTimer = FCooldownSeconds(CachedCooldown * PE.CooldownMult);

	BurstDamage    = CachedDamage * PE.DamageMult;
	BurstSpeed     = CachedSpeed  * PE.SpeedMult;
	// wiki: "Axe scales with Area × 1.3" → hitbox radius に AreaMult × 1.3 を適用
	BurstRadius    = 10.f * SurvivorsGameConstants::AxeAreaScaleFactor * PE.AreaMult;
	// wiki: "Speed determines how far up axes can be thrown" → 弧の高さに SpeedMult を適用
	BurstArcHeight = CachedArcHeight * PE.SpeedMult;
	BurstPierce    = CachedPierce;

	// 横方向: 最近傍敵の X 方向（バースト開始時に一度決定）
	BurstHorizDir = 1.f;
	float MinDistSq = MAX_FLT;
	for (const FEnemyState& E : Game->Enemies)
	{
		if (E.bPendingRemove) continue;
		const float Dsq = (E.Pos - Game->PlayerPos).SizeSquared();
		if (Dsq < MinDistSq)
		{
			MinDistSq    = Dsq;
			BurstHorizDir = (E.Pos.X >= Game->PlayerPos.X) ? 1.f : -1.f;
		}
	}

	PendingAxeShots      = FMath::Max(1, CachedAmount + static_cast<int32>(PE.ExtraAmount));
	AxeBurstTimer        = 0.f;
	BurstShotsFiredCount = 0;

	if (PendingAxeShots > 0)
	{
		SpawnAxeShot();
		--PendingAxeShots;
		AxeBurstTimer = (PendingAxeShots > 0) ? SurvivorsGameConstants::AxeProjectileInterval : 0.f;
	}
}

void USurvivorsAxeWeapon::SpawnAxeShot()
{
	// 放物線パラメータ: 頂点高さ ArcHeight に SpeedMult を適用
	const float InitVelY = BurstSpeed;
	const float HalfTime = (InitVelY > KINDA_SMALL_NUMBER) ? (BurstArcHeight / InitVelY) : 0.333f;
	const float GravityY = -InitVelY / HalfTime;

	// 1発目: 真上。追加弾: 横オフセット付き（OBSERVED: weapon_axe.md）
	const float HorizScale = (BurstShotsFiredCount == 0)
		? 0.f
		: (0.25f + 0.15f * static_cast<float>(BurstShotsFiredCount - 1));

	FProjectileState P;
	P.Pos               = Game->PlayerPos;
	P.Vel               = FVector2D(BurstHorizDir * BurstSpeed * HorizScale, InitVelY);
	P.Radius            = FSimRadius(BurstRadius);
	P.Damage            = FDamage(BurstDamage);
	P.WeaponType        = WeaponType;
	P.WeaponSlotIdx     = SlotIdx;
	P.LifeTime          = FProjectileLifeTime(20.f);  // 位置ベースで削除するため大きめ
	P.bPiercing         = false;
	P.MaxPierceCount    = BurstPierce;
	P.KnockbackStrength = SurvivorsGameConstants::KnockbackSim_1;
	P.AngleRad          = FOrbitAngleRad(GravityY);  // 流用: 重力加速度
	WeaponComp->SpawnProjectile(P);

	++BurstShotsFiredCount;
}
