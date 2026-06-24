#include "Survivors/Game/Weapons/Projectile/SurvivorsWeaponWhip.h"

#include "Survivors/Game/SurvivorsCollisionComponent.h"
#include "Survivors/Game/SurvivorsGame.h"
#include "Survivors/SurvivorsGameConstants.h"
#include "Survivors/Game/Weapons/SurvivorsWeaponComponent.h"

void USurvivorsWeaponWhip::OnLevelChanged(FWeaponLevel NewLevel)
{
	CacheParams();
}

void USurvivorsWeaponWhip::CacheParams()
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

void USurvivorsWeaponWhip::Tick(float Dt)
{
	if (!Game || !WeaponComp) return;

	if (FMath::Abs(Game->PlayerVel.X) > KINDA_SMALL_NUMBER)
	{
		LastFaceSign = (Game->PlayerVel.X >= 0.f) ? 1.f : -1.f;
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
			if (PendingWhips > 0)
			{
				WhipBurstTimer += 0.30f;
			}
		}
	}

	if (!CooldownTimer.IsReady() || PendingWhips > 0) return;

	StartBurst();
}

void USurvivorsWeaponWhip::StartBurst()
{
	const FPassiveEffects& PE = GetPassiveEffects();
	const float EffCooldown = CachedCooldown * PE.CooldownMult;
	CooldownTimer = FCooldownSeconds(EffCooldown);

	BurstDamage = CachedDamage * PE.DamageMult;
	BurstHalfWidth = CachedWidth * PE.AreaMult;
	BurstHalfHeight = CachedHeight * PE.AreaMult;
	BurstLifeTime = 0.20f * PE.DurationMult;
	BurstFaceSign = LastFaceSign;
	BurstIndex = 0;
	PendingWhips = FMath::Max(1, CachedAmount + static_cast<int32>(PE.ExtraAmount));
	WhipBurstTimer = 0.f;
	if (PendingWhips > 0)
	{
		SpawnWhipSwing(BurstFaceSign);
		++BurstIndex;
		--PendingWhips;
		WhipBurstTimer = (PendingWhips > 0) ? 0.30f : 0.f;
	}
}

void USurvivorsWeaponWhip::SpawnWhipSwing(float DirSign)
{
	FProjectileState P;
	P.Pos               = Game->PlayerPos + FVector2D(DirSign * BurstHalfWidth, 0.f);
	P.Vel               = FVector2D::ZeroVector;
	P.Radius            = FSimRadius(BurstHalfHeight);
	P.Damage            = FDamage(BurstDamage);
	P.WeaponType        = WeaponType;
	P.WeaponSlotIdx     = SlotIdx;
	P.LifeTime          = FProjectileLifeTime(BurstLifeTime);
	P.bPiercing         = true;   // AoE: 無限貫通
	P.AngleRad          = FOrbitAngleRad(BurstHalfWidth);
	P.KnockbackStrength = SurvivorsGameConstants::KnockbackSim_1;  // Knockback=1
	WeaponComp->SpawnProjectile(P);
}

void USurvivorsWeaponWhip::ComputeHits(USurvivorsCollisionComponent* CollComp, FSurvivorsHitFrame& HitFrame)
{
	if (!Game || !WeaponComp || !CollComp) return;

	TArray<FProjectileState>& Projectiles = WeaponComp->GetProjectiles();
	for (int32 PIdx = 0; PIdx < Projectiles.Num(); ++PIdx)
	{
		FProjectileState& P = Projectiles[PIdx];
		if (P.WeaponSlotIdx != SlotIdx) continue;
		if (P.WeaponType != EWeaponType::Whip && P.WeaponType != EWeaponType::BloodyTear) continue;

		const float HalfWidth = FMath::Max(P.AngleRad.Value, 1.f);
		const float HalfHeight = FMath::Max(P.Radius.Value, 1.f);
		const float QueryRadius = FMath::Sqrt(FMath::Square(HalfWidth) + FMath::Square(HalfHeight));

		TArray<const FSurvivorsTargetProxy*> Contacts;
		CollComp->QueryEnemyContacts(P.Pos, QueryRadius, Contacts);

		for (const FSurvivorsTargetProxy* Proxy : Contacts)
		{
			const FVector2D Rel = Proxy->Pos - P.Pos;
			if (FMath::Abs(Rel.X) > HalfWidth + Proxy->Radius) continue;
			if (FMath::Abs(Rel.Y) > HalfHeight + Proxy->Radius) continue;

			const int32 EIdx = Proxy->Ref.IndexAtBuildTime;
			if (!Game->Enemies.IsValidIndex(EIdx) || Game->Enemies[EIdx].UniqueId != Proxy->Ref.UniqueId) continue;
			const FEnemyState& E = Game->Enemies[EIdx];
			if (E.bPendingRemove) continue;
			if (P.HitEnemyIds.Contains(E.UniqueId)) continue;

			FSurvivorsHitEvent Ev;
			Ev.Type = ESurvivorsHitType::ProjectileDamage;
			Ev.Target = Proxy->Ref;
			Ev.Damage = P.Damage.Value;
			Ev.WeaponSlot = PIdx;
			Ev.KnockbackDir = FVector2D((P.Pos.X >= Game->PlayerPos.X) ? 1.f : -1.f, 0.f);
			Ev.KnockbackStrength = P.KnockbackStrength;
			HitFrame.Events.Add(Ev);

			P.HitEnemyIds.Add(E.UniqueId);
		}
	}
}
