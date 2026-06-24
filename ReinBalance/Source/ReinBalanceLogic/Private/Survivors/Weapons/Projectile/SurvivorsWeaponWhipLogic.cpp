#include "Survivors/Weapons/Projectile/SurvivorsWeaponWhipLogic.h"
#include "Survivors/SurvivorsGameLogic.h"
#include "Survivors/SurvivorsGameConstants.h"

void FSurvivorsWeaponWhipLogic::OnLevelChanged(FWeaponLevel NewLevel)
{
	CacheParams();
}

void FSurvivorsWeaponWhipLogic::CacheParams()
{
	const int32 Lv  = FMath::Clamp(Level.Value, 1, SurvivorsGameConstants::MaxWeaponLevel);
	const int32 Idx = Lv - 1;

	if (WeaponType == EWeaponType::BloodyTear)
	{
		const SurvivorsGameConstants::FWhipParams& P = SurvivorsGameConstants::BloodyTearTable[Idx];
		CachedDamage   = P.Damage;
		CachedCooldown = P.Cooldown;
		CachedWidth    = P.Width;
		CachedHeight   = P.Height;
		CachedAmount   = P.Amount;
	}
	else
	{
		const SurvivorsGameConstants::FWhipParams& P = SurvivorsGameConstants::WhipTable[Idx];
		CachedDamage   = P.Damage;
		CachedCooldown = P.Cooldown;
		CachedWidth    = P.Width;
		CachedHeight   = P.Height;
		CachedAmount   = P.Amount;
	}
}

void FSurvivorsWeaponWhipLogic::Tick(float Dt)
{
	if (!Logic) return;

	if (FMath::Abs(Logic->PlayerVel.X) > KINDA_SMALL_NUMBER)
	{
		LastFaceSign = (Logic->PlayerVel.X >= 0.f) ? 1.f : -1.f;
	}

	CooldownTimer.Value = FMath::Max(0.f, CooldownTimer.Value - Dt);

	if (PendingWhips > 0)
	{
		WhipBurstTimer -= Dt;
		while (PendingWhips > 0 && WhipBurstTimer <= 0.f)
		{
			const float DirSign = (BurstIndex % 2 == 0) ? BurstFaceSign : -BurstFaceSign;
			SpawnWhipSwing(DirSign);
			++BurstIndex;
			--PendingWhips;
			if (PendingWhips > 0) WhipBurstTimer += 0.30f;
		}
	}

	if (!CooldownTimer.IsReady() || PendingWhips > 0) return;

	StartBurst();
}

void FSurvivorsWeaponWhipLogic::StartBurst()
{
	const FPassiveEffects& PE = GetPassiveEffects();
	const float EffCooldown = CachedCooldown * PE.CooldownMult;
	CooldownTimer = FCooldownSeconds(EffCooldown);

	BurstDamage      = CachedDamage * PE.DamageMult;
	BurstHalfWidth   = CachedWidth  * PE.AreaMult;
	BurstHalfHeight  = CachedHeight * PE.AreaMult;
	BurstLifeTime    = 0.20f * PE.DurationMult;
	BurstFaceSign    = LastFaceSign;
	BurstIndex       = 0;
	PendingWhips     = FMath::Max(1, CachedAmount + static_cast<int32>(PE.ExtraAmount));
	WhipBurstTimer   = 0.f;

	if (PendingWhips > 0)
	{
		SpawnWhipSwing(BurstFaceSign);
		++BurstIndex;
		--PendingWhips;
		WhipBurstTimer = (PendingWhips > 0) ? 0.30f : 0.f;
	}
}

void FSurvivorsWeaponWhipLogic::SpawnWhipSwing(float DirSign)
{
	FProjectileState P;
	P.Pos               = Logic->PlayerPos + FVector2D(DirSign * BurstHalfWidth, 0.f);
	P.Vel               = FVector2D::ZeroVector;
	P.Radius            = FSimRadius(BurstHalfHeight);
	P.Damage            = FDamage(BurstDamage);
	P.WeaponType        = WeaponType;
	P.WeaponSlotIdx     = SlotIdx;
	P.LifeTime          = FProjectileLifeTime(BurstLifeTime);
	P.bPiercing         = true;
	P.AngleRad          = FOrbitAngleRad(BurstHalfWidth);
	P.KnockbackStrength = SurvivorsGameConstants::KnockbackSim_1;
	Logic->SpawnProjectile(P);
}

void FSurvivorsWeaponWhipLogic::ComputeHits(FSurvivorsHitFrame& HitFrame)
{
	if (!Logic) return;

	TArray<FProjectileState>& Projectiles = Logic->GetProjectiles();
	for (int32 PIdx = 0; PIdx < Projectiles.Num(); ++PIdx)
	{
		FProjectileState& P = Projectiles[PIdx];
		if (P.WeaponSlotIdx != SlotIdx) continue;
		if (P.WeaponType != EWeaponType::Whip && P.WeaponType != EWeaponType::BloodyTear) continue;

		const float HalfWidth  = FMath::Max(P.AngleRad.Value, 1.f);
		const float HalfHeight = FMath::Max(P.Radius.Value, 1.f);
		const float QueryRadius = FMath::Sqrt(FMath::Square(HalfWidth) + FMath::Square(HalfHeight));

		TArray<const FSurvivorsTargetProxy*> Contacts;
		Logic->QueryEnemyContacts(P.Pos, QueryRadius, Contacts);

		for (const FSurvivorsTargetProxy* Proxy : Contacts)
		{
			const FVector2D Rel = Proxy->Pos - P.Pos;
			if (FMath::Abs(Rel.X) > HalfWidth  + Proxy->Radius) continue;
			if (FMath::Abs(Rel.Y) > HalfHeight + Proxy->Radius) continue;

			const int32 EIdx = Proxy->Ref.IndexAtBuildTime;
			if (!Logic->Enemies.IsValidIndex(EIdx) || Logic->Enemies[EIdx].UniqueId != Proxy->Ref.UniqueId) continue;
			const FEnemyState& E = Logic->Enemies[EIdx];
			if (E.bPendingRemove) continue;
			if (P.HitEnemyIds.Contains(E.UniqueId)) continue;

			FSurvivorsHitEvent Ev;
			Ev.Type              = ESurvivorsHitType::ProjectileDamage;
			Ev.Target            = Proxy->Ref;
			Ev.Damage            = P.Damage.Value;
			Ev.WeaponSlot        = PIdx;
			Ev.KnockbackDir      = FVector2D((P.Pos.X >= Logic->PlayerPos.X) ? 1.f : -1.f, 0.f);
			Ev.KnockbackStrength = P.KnockbackStrength;
			HitFrame.Events.Add(Ev);

			P.HitEnemyIds.Add(E.UniqueId);
		}
	}
}
