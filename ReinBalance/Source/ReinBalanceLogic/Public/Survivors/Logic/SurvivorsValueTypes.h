#pragma once

#include "CoreMinimal.h"

struct FDamage
{
	float Value = 0.f;
	explicit constexpr FDamage(float InValue = 0.f) : Value(InValue) {}
};

struct FSurvivorsElapsedTime
{
	float Seconds = 0.f;
	explicit FSurvivorsElapsedTime(float InSeconds = 0.f) : Seconds(InSeconds) {}
};

struct FWeaponLevel
{
	int32 Value = 1;
	explicit FWeaponLevel(int32 InValue = 1) : Value(InValue) {}
	bool IsMax() const { return Value >= 8; }
};

struct FPlayerLevel
{
	int32 Value = 1;
	explicit FPlayerLevel(int32 InValue = 1) : Value(InValue) {}
};

struct FSimRadius
{
	float Value = 0.f;
	explicit constexpr FSimRadius(float InValue = 0.f) : Value(InValue) {}
};

struct FCooldownSeconds
{
	float Value = 0.f;
	explicit FCooldownSeconds(float InValue = 0.f) : Value(InValue) {}
	bool IsReady() const { return Value <= 0.f; }
};

struct FProjectileLifeTime
{
	float Seconds = 0.f;
	explicit FProjectileLifeTime(float InSeconds = 0.f) : Seconds(InSeconds) {}
	bool IsExpired() const { return Seconds <= 0.f; }
	void Tick(float Dt) { Seconds -= Dt; }
};

struct FOrbitAngleRad
{
	float Value = 0.f;
	explicit FOrbitAngleRad(float InValue = 0.f) : Value(InValue) {}
	void Advance(float RadPerSec, float Dt) { Value += RadPerSec * Dt; }
};

struct FBounceCount
{
	int32 Value = 0;
	explicit FBounceCount(int32 InValue = 0) : Value(InValue) {}
	bool HasBounces() const { return Value > 0; }
	void Consume() { --Value; }
};
