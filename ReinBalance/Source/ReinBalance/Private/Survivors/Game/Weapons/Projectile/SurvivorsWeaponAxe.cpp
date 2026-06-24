#include "Survivors/Game/Weapons/Projectile/SurvivorsWeaponAxe.h"

#include "Survivors/Game/SurvivorsGame.h"
#include "Survivors/Logic/SurvivorsGameConstants.h"
#include "Survivors/Game/Weapons/SurvivorsWeaponComponent.h"

void USurvivorsWeaponAxe::OnLevelChanged(FWeaponLevel NewLevel)
{
	CacheParams();
}

void USurvivorsWeaponAxe::CacheParams()
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

void USurvivorsWeaponAxe::Tick(float Dt)
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

void USurvivorsWeaponAxe::StartBurst()
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

	// Axe は各弾が真上(+Y)を中心に ±45° のランダム方向に飛ぶ。
	// 敵ロックオンは使わない（BurstHorizDir は SpawnAxeShot 内でランダム化するため0で初期化）。
	BurstHorizDir = 0.f;  // SpawnAxeShot 内でランダム角度を生成するため未使用

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

void USurvivorsWeaponAxe::SpawnAxeShot()
{
	// 放物線パラメータ: 頂点高さ ArcHeight に SpeedMult を適用
	const float InitVelY = BurstSpeed;
	const float HalfTime = (InitVelY > KINDA_SMALL_NUMBER) ? (BurstArcHeight / InitVelY) : 0.333f;
	const float GravityY = -InitVelY / HalfTime;

	// 1発目は直上（X=0）。追加弾は真上（+Y）から ±45° のランダム方向。
	// wiki: "The first axe is thrown directly above the character"
	// ユーザー仕様: 上方向左右45度ランダム（追加弾）。Game->RandStream で再現性確保。
	const float RandomOffset = (BurstShotsFiredCount == 0)
		? 0.f  // 1発目: 真上固定
		: Game->RandStream.FRandRange(
			-SurvivorsGameConstants::AxeRandomConeHalfAngle,
			 SurvivorsGameConstants::AxeRandomConeHalfAngle);
	const float HorizScale = FMath::Sin(RandomOffset);  // 横成分: 1発目=0, 追加弾=ランダム

	FProjectileState P;
	P.Pos               = Game->PlayerPos;
	// Vel.Y = Speed * cos(offset) ≥ 0（±45°なので常に上向き成分あり）
	P.Vel               = FVector2D(BurstSpeed * HorizScale, BurstSpeed * FMath::Cos(RandomOffset));
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
