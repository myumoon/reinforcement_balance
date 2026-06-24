#include "Survivors/Game/Weapons/Projectile/SurvivorsWeaponFireWand.h"

#include "Survivors/Game/SurvivorsCollisionComponent.h"
#include "Survivors/Game/SurvivorsGame.h"
#include "Survivors/Game/SurvivorsGameConstants.h"
#include "Survivors/Game/Weapons/SurvivorsWeaponComponent.h"

void USurvivorsWeaponFireWand::OnLevelChanged(FWeaponLevel NewLevel)
{
	CacheParams();
}

void USurvivorsWeaponFireWand::CacheParams()
{
	const int32 Lv  = FMath::Clamp(Level.Value, 1, SurvivorsGameConstants::MaxWeaponLevel);
	const int32 Idx = Lv - 1;

	if (WeaponType == EWeaponType::Hellfire)
	{
		const SurvivorsGameConstants::FFireWandParams& P = SurvivorsGameConstants::HellfireTable[Idx];
		CachedDamage          = P.Damage;
		CachedCooldown        = P.Cooldown;
		CachedSpeed           = P.Speed;
		CachedExplosionRadius = P.ExplosionRadius;
		CachedAmount          = P.Amount;
	}
	else
	{
		const SurvivorsGameConstants::FFireWandParams& P = SurvivorsGameConstants::FireWandTable[Idx];
		CachedDamage          = P.Damage;
		CachedCooldown        = P.Cooldown;
		CachedSpeed           = P.Speed;
		CachedExplosionRadius = P.ExplosionRadius;
		CachedAmount          = P.Amount;
	}
}

void USurvivorsWeaponFireWand::Tick(float Dt)
{
	if (!Game || !WeaponComp) return;

	const FPassiveEffects& PE = GetPassiveEffects();

	const float EffExplosionRadius = CachedExplosionRadius * PE.AreaMult;
	const float EffDamage          = CachedDamage * PE.DamageMult;
	const float EffDuration        = 0.2f * PE.DurationMult;  // 爆発は短時間

	// 寿命切れ弾を検出して爆発 GroundZone 生成。
	// TickProjectiles が LifeTime 切れ時に bPendingExplosion = true を立ててから
	// このフラグで処理するか、Tick が先の場合は IsExpired() で直接処理する。
	// どちらも安全に動作するよう両方のケースをカバーする。
	WeaponComp->UpdateProjectilesBySlot(SlotIdx, 0.f, [&](FProjectileState& P, float) -> bool
	{
		// bPendingExplosion（TickProjectiles が前フレームで立てた爆発予約）または
		// IsExpired()（WeaponBase::Tick が TickProjectiles より先に呼ばれた場合）
		if (P.bPendingExplosion || P.LifeTime.IsExpired())
		{
			// 爆発 GroundZone 生成
			FGroundZoneState Z;
			Z.Pos           = P.Pos;
			Z.Radius        = EffExplosionRadius;
			Z.Damage        = EffDamage;
			Z.LifeTime      = EffDuration;
			Z.HitCooldown   = EffDuration;  // 1 回だけヒット
			Z.WeaponSlotIdx = SlotIdx;
			Z.WeaponType    = WeaponType;
			WeaponComp->SpawnGroundZone(Z);
			return false;  // 削除
		}
		return true;
	});

	// クールダウン
	CooldownTimer.Value = FMath::Max(0.f, CooldownTimer.Value - Dt);
	if (!CooldownTimer.IsReady()) return;

	CooldownTimer = FCooldownSeconds(CachedCooldown * PE.CooldownMult);

	const float EffSpeed    = CachedSpeed * PE.SpeedMult;
	// 旧値1.2sでは画面内最大距離(400u)を96u/s=4.17sで到達できない。画面横断(800u÷96u/s≈8.3s)をカバー。
	const float LifeTime    = 9.0f * PE.DurationMult;
	const int32 EffAmount   = FMath::Max(1, CachedAmount + static_cast<int32>(PE.ExtraAmount));

	// 画面内の敵からランダムに選択して扇状発射。画面内に敵がいない場合はランダム方向。
	FVector2D Dir = FVector2D::ZeroVector;
	TArray<int32> Candidates;
	for (int32 EIdx = 0; EIdx < Game->Enemies.Num(); ++EIdx)
	{
		const FEnemyState& E = Game->Enemies[EIdx];
		if (E.bPendingRemove) continue;
		if (!Game->IsOnScreen(E.Pos)) continue;  // 画面外の敵は対象外
		Candidates.Add(EIdx);
	}
	if (Candidates.Num() > 0)
	{
		const int32 ChoiceIdx = Game->RandStream.RandRange(0, Candidates.Num() - 1);
		Dir = (Game->Enemies[Candidates[ChoiceIdx]].Pos - Game->PlayerPos).GetSafeNormal();
	}
	if (Dir.IsNearlyZero())
	{
		const float RandomAngle = Game->RandStream.FRand() * TWO_PI;
		Dir = FVector2D(FMath::Cos(RandomAngle), FMath::Sin(RandomAngle));
	}

	const float BaseAngle = FMath::Atan2(Dir.Y, Dir.X);
	// OBSERVED: fire_wand_bullet4.mp4 frame 290/300, 4発全体≈16°, step≈5.3° (FireWandAngleStepDeg)
	const float AngleStep = FMath::DegreesToRadians(SurvivorsGameConstants::FireWandAngleStepDeg);
	for (int32 i = 0; i < EffAmount; ++i)
	{
		const float Offset = (static_cast<float>(i) - 0.5f * static_cast<float>(EffAmount - 1)) * AngleStep;
		const float Angle = BaseAngle + Offset;
		const FVector2D ShotDir(FMath::Cos(Angle), FMath::Sin(Angle));

		FProjectileState P;
		P.Pos           = Game->PlayerPos;
		P.Vel           = ShotDir * EffSpeed;
		P.Radius        = FSimRadius(8.f);
		P.Damage        = FDamage(0.f);  // 直撃ダメージなし（爆発で処理）
		P.WeaponType    = WeaponType;
		P.WeaponSlotIdx = SlotIdx;
		P.LifeTime      = FProjectileLifeTime(LifeTime);
		P.bPiercing     = true;   // 爆発前に ComputeProjectileHits で削除されないよう貫通設定
		WeaponComp->SpawnProjectile(P);
	}
}

void USurvivorsWeaponFireWand::ComputeHits(USurvivorsCollisionComponent* CollComp, FSurvivorsHitFrame& HitFrame)
{
	if (!Game || !WeaponComp || !CollComp) return;

	const FPassiveEffects& PE = GetPassiveEffects();
	const float EffExplosionRadius = CachedExplosionRadius * PE.AreaMult;
	const float EffDamage          = CachedDamage * PE.DamageMult;
	const float EffDuration        = 0.2f * PE.DurationMult;

	// 逆順でループして弾を削除しながら処理する
	TArray<FProjectileState>& Projectiles = WeaponComp->GetProjectiles();
	for (int32 i = Projectiles.Num() - 1; i >= 0; --i)
	{
		FProjectileState& P = Projectiles[i];
		if (P.WeaponSlotIdx != SlotIdx) continue;
		if (P.WeaponType != EWeaponType::FireWand && P.WeaponType != EWeaponType::Hellfire) continue;

		// 敵との接触確認
		TArray<const FSurvivorsTargetProxy*> Contacts;
		CollComp->QueryEnemyContacts(P.Pos, P.Radius.Value, Contacts);

		bool bHitEnemy = false;
		for (const FSurvivorsTargetProxy* Proxy : Contacts)
		{
			// narrowphase
			if ((P.Pos - Proxy->Pos).SizeSquared() > FMath::Square(P.Radius.Value + Proxy->Radius)) continue;

			// UniqueId 確認
			const int32 EIdx = Proxy->Ref.IndexAtBuildTime;
			if (!Game->Enemies.IsValidIndex(EIdx) || Game->Enemies[EIdx].UniqueId != Proxy->Ref.UniqueId) continue;
			if (Game->Enemies[EIdx].bPendingRemove) continue;

			bHitEnemy = true;
			break;
		}

		if (bHitEnemy || P.bPendingExplosion)
		{
			// 爆発 GroundZone を生成して範囲ダメージを与える
			FGroundZoneState Z;
			Z.Pos           = P.Pos;
			Z.Radius        = EffExplosionRadius;
			Z.Damage        = EffDamage;
			Z.LifeTime      = EffDuration;
			Z.HitCooldown   = EffDuration;  // 1 回だけヒット
			Z.WeaponSlotIdx = SlotIdx;
			Z.WeaponType    = WeaponType;
			WeaponComp->SpawnGroundZone(Z);

			// 弾を削除
			Projectiles.RemoveAt(i);
		}
	}
}
