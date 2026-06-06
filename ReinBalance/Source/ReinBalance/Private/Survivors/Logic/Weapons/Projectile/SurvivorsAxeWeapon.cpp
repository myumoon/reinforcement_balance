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

	const FPassiveEffects& PE = GetPassiveEffects();

	// --- 既存プロジェクタイルへ重力適用 ---
	// AngleRad.Value に重力加速度を格納して流用
	const float FieldBottom = -(Game->FieldHalfSize + 20.f);  // 画面下端 + マージン
	const float FieldLimit = Game->FieldHalfSize + 20.f;
	WeaponComp->UpdateProjectilesBySlot(SlotIdx, Dt, [FieldBottom, FieldLimit](FProjectileState& P, float InDt) -> bool
	{
		// AngleRad.Value = 重力加速度 (負値)
		P.Vel.Y += P.AngleRad.Value * InDt;
		if (P.WeaponType == EWeaponType::DeathSpiral)
		{
			return FMath::Abs(P.Pos.X) <= FieldLimit && FMath::Abs(P.Pos.Y) <= FieldLimit;
		}
		// 画面下端を超えたら削除
		return P.Pos.Y > FieldBottom;
	});

	// --- クールダウン ---
	CooldownTimer.Value = FMath::Max(0.f, CooldownTimer.Value - Dt);
	if (!CooldownTimer.IsReady()) return;

	CooldownTimer = FCooldownSeconds(CachedCooldown * PE.CooldownMult);

	const float EffDamage    = CachedDamage    * PE.DamageMult;
	const float EffSpeed     = CachedSpeed     * PE.SpeedMult;
	const float EffArcHeight = CachedArcHeight * PE.AreaMult;
	const int32 EffAmount    = FMath::Max(1, CachedAmount + static_cast<int32>(PE.ExtraAmount));

	if (WeaponType == EWeaponType::DeathSpiral)
	{
		for (int32 i = 0; i < EffAmount; ++i)
		{
			const float Angle = TWO_PI * static_cast<float>(i) / static_cast<float>(EffAmount);
			const FVector2D Dir(FMath::Cos(Angle), FMath::Sin(Angle));

			FProjectileState P;
			P.Pos               = Game->PlayerPos;
			P.Vel               = Dir * EffSpeed;
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

	// 放物線パラメータ計算
	// 頂点に達するまでの時間 = HalfTime = ArcHeight / InitVelY
	// 画面下端まで落下する時間は UpdateProjectilesBySlot 内の位置チェックで管理するため
	// LifeTime は十分大きな値（画面外に出るまでの余裕）を設定する
	const float InitVelY = EffSpeed;
	const float HalfTime = EffArcHeight / InitVelY;
	const float LifeTime = 20.f;  // 位置ベースで削除するため大きめに設定
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

	for (int32 i = 0; i < EffAmount; ++i)
	{
		const float HorizScale = (i == 0) ? 0.f : (0.25f + 0.15f * static_cast<float>(i - 1));

		FProjectileState P;
		P.Pos               = Game->PlayerPos;
		P.Vel               = FVector2D(HorizDir * EffSpeed * HorizScale, InitVelY);
		P.Radius            = FSimRadius(10.f);
		P.Damage            = FDamage(EffDamage);
		P.WeaponType        = WeaponType;
		P.WeaponSlotIdx     = SlotIdx;
		P.LifeTime          = FProjectileLifeTime(LifeTime);
		P.bPiercing         = false;
		P.MaxPierceCount    = CachedPierce;
		P.KnockbackStrength = SurvivorsGameConstants::KnockbackSim_1;  // Knockback=1
		P.AngleRad          = FOrbitAngleRad(GravityY);  // 流用: 重力加速度を格納
		WeaponComp->SpawnProjectile(P);
	}
}
