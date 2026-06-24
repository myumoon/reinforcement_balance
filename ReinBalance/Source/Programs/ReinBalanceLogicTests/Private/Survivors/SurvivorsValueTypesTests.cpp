#include "TestHarness.h"

#include "Survivors/SurvivorsValueTypes.h"

TEST_CASE("Survivors cooldown is ready only when no positive time remains", "[unit][survivors][logic][value-types]")
{
	CHECK(FCooldownSeconds(0.f).IsReady());
	CHECK(FCooldownSeconds(-0.01f).IsReady());
	CHECK(!FCooldownSeconds(0.01f).IsReady());
}

TEST_CASE("Survivors projectile lifetime expires after ticking through zero", "[unit][survivors][logic][value-types]")
{
	FProjectileLifeTime LifeTime(0.25f);
	CHECK(!LifeTime.IsExpired());

	LifeTime.Tick(0.10f);
	CHECK(!LifeTime.IsExpired());
	CHECK(FMath::IsNearlyEqual(LifeTime.Seconds, 0.15f, 0.001f));

	LifeTime.Tick(0.15f);
	CHECK(LifeTime.IsExpired());
}

TEST_CASE("Survivors weapon level reports max at level eight and above", "[unit][survivors][logic][value-types]")
{
	CHECK(!FWeaponLevel(7).IsMax());
	CHECK(FWeaponLevel(8).IsMax());
	CHECK(FWeaponLevel(9).IsMax());
}

TEST_CASE("Survivors orbit angle advances by angular speed and delta time", "[unit][survivors][logic][value-types]")
{
	FOrbitAngleRad Angle(0.25f);
	Angle.Advance(2.f, 0.5f);

	CHECK(FMath::IsNearlyEqual(Angle.Value, 1.25f, 0.001f));
}

TEST_CASE("Survivors bounce count consumes the last available bounce", "[unit][survivors][logic][value-types]")
{
	FBounceCount BounceCount(1);
	CHECK(BounceCount.HasBounces());

	BounceCount.Consume();

	CHECK(BounceCount.Value == 0);
	CHECK(!BounceCount.HasBounces());
}