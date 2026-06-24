#include "Misc/AutomationTest.h"
#include "SurvivorsTestHelpers.h"
#include "Survivors/Game/Weapons/Projectile/SurvivorsKingBibleWeapon.h"

// ============================================================
// 速度上昇パッシブ（Bracer Lv1-5）武器速度スケーリングテスト
// ============================================================
// Bracer: SpeedMult += 0.10 × Level (初期値 1.0)
// 対象: MagicWand / Knife / Cross / FireWand / Runetracer (弾速)
//       Axe (弧頂点高さ)  KingBible / Peachone / EbonyWings / Vandalier (軌道回転速度)
// 非対象（SpeedMult 未使用）: Garlic, Whip, SantaWater, LightningRing, Pentagram, Laurel

static void SetupBracerPassive(ASurvivorsGame* Game, int32 BracerLevel)
{
	FPassiveSlot* Passives = FSurvivorsGameTestAccess::PassiveSlots(Game);
	Passives[0].Type  = EPassiveItemType::Bracer;
	Passives[0].Level = BracerLevel;
	FSurvivorsGameTestAccess::PlayerComp(Game)->RecalcPassiveEffects();
}

// プロジェクタイル直線速度: MagicWand, Knife, Cross, FireWand, Runetracer
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsWeaponProjectileSpeedBracerPassive,
	"ReinBalance.Survivors.Passive.BracerLv1to5.ProjectileSpeed",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FSurvivorsWeaponProjectileSpeedBracerPassive::RunTest(const FString& Parameters)
{
	struct FCase
	{
		EWeaponType Type;
		float BaseSpeed;  // Lv1 table 値 (u/s)
		const TCHAR* Label;
	};

	const FCase Cases[] = {
		{ EWeaponType::MagicWand,  140.f, TEXT("MagicWand")  },
		{ EWeaponType::Knife,      326.f, TEXT("Knife")       },
		{ EWeaponType::Cross,      320.f, TEXT("Cross")       },
		{ EWeaponType::FireWand,    96.f, TEXT("FireWand")    },
		{ EWeaponType::Runetracer, 193.f, TEXT("Runetracer")  },
	};

	for (const FCase& Case : Cases)
	{
		for (int32 BracerLv = 1; BracerLv <= 5; ++BracerLv)
		{
			FSurvivorsTestWorld S;
			if (!TestTrue(FString::Printf(TEXT("[%s BracerLv%d] world"), Case.Label, BracerLv), S.Create()))
				return false;

			SetupBracerPassive(S.Game, BracerLv);
			FSurvivorsGameTestAccess::PlayerPos(S.Game) = FVector2D::ZeroVector;
			FSurvivorsGameTestAccess::PlayerVel(S.Game) = FVector2D(1.f, 0.f);
			S.AddEnemyAt(FVector2D(200.f, 0.f), 100000.f);

			EquipTestWeapon(S.Game, Case.Type, 1);
			auto* WC = FSurvivorsGameTestAccess::WeaponComp(S.Game);

			const float SpeedMult     = 1.0f + 0.10f * static_cast<float>(BracerLv);
			const float ExpectedSpeed = Case.BaseSpeed * SpeedMult;

			if (Case.Type == EWeaponType::FireWand)
			{
				// FireWand は 0.02s 間隔の扇撃ち: 全 4 発出そろってから計測
				WC->TickWeapons(SurvivorsGameConstants::PhysicsDt);
				TickTestWeaponsForSeconds(WC, 0.08f);
				if (!TestEqual(FString::Printf(TEXT("[%s BracerLv%d] 4 fireballs"), Case.Label, BracerLv),
					WC->GetProjectileCount(), 4))
				{
					S.Destroy();
					continue;
				}

				TArray<FVector2D> Starts;
				for (int32 i = 0; i < 4; ++i)
					Starts.Add(WC->GetProjectilePos(i));

				const float Elapsed = TickTestWeaponsForSecondsMeasured(WC, 0.20f);
				if (!TestEqual(FString::Printf(TEXT("[%s BracerLv%d] 4 fireballs survive"), Case.Label, BracerLv),
					WC->GetProjectileCount(), 4))
				{
					S.Destroy();
					continue;
				}

				float TotalSpeed = 0.f;
				for (int32 i = 0; i < 4; ++i)
					TotalSpeed += FVector2D::Distance(WC->GetProjectilePos(i), Starts[i]) / Elapsed;
				const float AvgSpeed = TotalSpeed / 4.f;

				TestTrue(FString::Printf(TEXT("[%s BracerLv%d] avg speed %.1fu/s ≈ %.1fu/s (±10%%)"),
					Case.Label, BracerLv, AvgSpeed, ExpectedSpeed),
					AvgSpeed >= ExpectedSpeed * 0.90f && AvgSpeed <= ExpectedSpeed * 1.10f);

				S.Destroy();
				continue;
			}

			// Cross は固定折り返し距離(75u/BurstSpeed)があるため短い区間で計測。
			// BracerLv5 (480u/s) の折り返し時間 = 75/480 ≈ 0.156s → 0.10s 計測で安全。
			const float MeasureSecs = (Case.Type == EWeaponType::Cross) ? 0.10f : 0.25f;

			WC->TickWeapons(SurvivorsGameConstants::PhysicsDt);
			if (!TestTrue(FString::Printf(TEXT("[%s BracerLv%d] fires 1st projectile"), Case.Label, BracerLv),
				WC->GetProjectileCount() > 0))
			{
				S.Destroy();
				return false;
			}

			const FVector2D StartPos = WC->GetProjectilePos(0);
			const float Elapsed      = TickTestWeaponsForSecondsMeasured(WC, MeasureSecs);

			if (!TestTrue(FString::Printf(TEXT("[%s BracerLv%d] projectile alive"), Case.Label, BracerLv),
				WC->GetProjectileCount() > 0))
			{
				S.Destroy();
				return false;
			}

			const float ActualSpeed = FVector2D::Distance(WC->GetProjectilePos(0), StartPos) / Elapsed;
			TestTrue(FString::Printf(TEXT("[%s BracerLv%d] speed %.1fu/s ≈ %.1fu/s (±10%%)"),
				Case.Label, BracerLv, ActualSpeed, ExpectedSpeed),
				ActualSpeed >= ExpectedSpeed * 0.90f && ActualSpeed <= ExpectedSpeed * 1.10f);

			S.Destroy();
		}
	}

	return true;
}

// Axe: BurstArcHeight = CachedArcHeight × SpeedMult → 頂点高さ ≈ 60u × SpeedMult
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsAxeApexHeightBracerPassive,
	"ReinBalance.Survivors.Passive.BracerLv1to5.Axe_ApexHeight",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FSurvivorsAxeApexHeightBracerPassive::RunTest(const FString& Parameters)
{
	// AxeTable Lv1: ArcHeight=120 → 連続解析での頂点 = ArcHeight/2 = 60u
	// SpeedMult 適用後: BurstArcHeight = 120 × SpeedMult → 頂点 = 60 × SpeedMult
	static constexpr float BaseApex = 60.f;

	for (int32 BracerLv = 1; BracerLv <= 5; ++BracerLv)
	{
		FSurvivorsTestWorld S;
		if (!TestTrue(FString::Printf(TEXT("[Axe BracerLv%d] world"), BracerLv), S.Create()))
			return false;

		SetupBracerPassive(S.Game, BracerLv);
		FSurvivorsGameTestAccess::PlayerPos(S.Game) = FVector2D::ZeroVector;

		EquipTestWeapon(S.Game, EWeaponType::Axe, 1);
		auto* WC = FSurvivorsGameTestAccess::WeaponComp(S.Game);

		WC->TickWeapons(SurvivorsGameConstants::PhysicsDt);

		float MaxHeight = 0.f;
		for (int32 Step = 0; Step < SurvivorsStepsForSeconds(1.0f); ++Step)
		{
			if (WC->GetProjectileCount() > 0)
				MaxHeight = FMath::Max(MaxHeight, WC->GetProjectilePos(0).Y);
			WC->TickWeapons(SurvivorsGameConstants::PhysicsDt);
		}

		const float SpeedMult    = 1.0f + 0.10f * static_cast<float>(BracerLv);
		const float ExpectedApex = BaseApex * SpeedMult;
		TestTrue(FString::Printf(TEXT("[Axe BracerLv%d] apex %.1fu ≈ %.1fu (±15%%)"),
			BracerLv, MaxHeight, ExpectedApex),
			MaxHeight >= ExpectedApex * 0.85f && MaxHeight <= ExpectedApex * 1.15f);

		S.Destroy();
	}

	return true;
}

// KingBible: MasterAngle 回転速度 = CachedRotSpeed × SpeedMult が仕様通りか
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsKingBibleOrbitSpeedBracerPassive,
	"ReinBalance.Survivors.Passive.BracerLv1to5.KingBible_OrbitSpeed",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FSurvivorsKingBibleOrbitSpeedBracerPassive::RunTest(const FString& Parameters)
{
	// KingBibleTable Lv1 RotSpeed = 4.0 rad/s
	static constexpr float BaseRotSpeed = 4.0f;

	for (int32 BracerLv = 1; BracerLv <= 5; ++BracerLv)
	{
		FSurvivorsTestWorld S;
		if (!TestTrue(FString::Printf(TEXT("[KingBible BracerLv%d] world"), BracerLv), S.Create()))
			return false;

		SetupBracerPassive(S.Game, BracerLv);
		FSurvivorsGameTestAccess::PlayerPos(S.Game) = FVector2D::ZeroVector;

		EquipTestWeapon(S.Game, EWeaponType::KingBible, 1);
		auto* WC = FSurvivorsGameTestAccess::WeaponComp(S.Game);
		auto* KB = Cast<USurvivorsKingBibleWeapon>(WC->GetWeaponInstance(0));
		if (!TestTrue(FString::Printf(TEXT("[KingBible BracerLv%d] weapon instance"), BracerLv), IsValid(KB)))
		{
			S.Destroy();
			return false;
		}

		WC->TickWeapons(SurvivorsGameConstants::PhysicsDt);
		if (!TestTrue(FString::Printf(TEXT("[KingBible BracerLv%d] orbs active"), BracerLv),
			KB->GetOrbPositions().Num() >= 1))
		{
			S.Destroy();
			return false;
		}

		const FVector2D PlayerPos = FSurvivorsGameTestAccess::PlayerPos(S.Game);
		const float StartAngle    = FMath::Atan2(
			KB->GetOrbPositions()[0].Y - PlayerPos.Y,
			KB->GetOrbPositions()[0].X - PlayerPos.X);

		const float Elapsed = TickTestWeaponsForSecondsMeasured(WC, 0.50f);

		if (!TestTrue(FString::Printf(TEXT("[KingBible BracerLv%d] orbs active after 0.5s"), BracerLv),
			KB->GetOrbPositions().Num() >= 1))
		{
			S.Destroy();
			return false;
		}

		const float EndAngle  = FMath::Atan2(
			KB->GetOrbPositions()[0].Y - PlayerPos.Y,
			KB->GetOrbPositions()[0].X - PlayerPos.X);
		const float ActualRot = FMath::Abs(FMath::FindDeltaAngleRadians(StartAngle, EndAngle)) / Elapsed;

		const float SpeedMult        = 1.0f + 0.10f * static_cast<float>(BracerLv);
		const float ExpectedRotSpeed = BaseRotSpeed * SpeedMult;
		TestTrue(FString::Printf(TEXT("[KingBible BracerLv%d] orbit %.2frad/s ≈ %.2frad/s (±8%%)"),
			BracerLv, ActualRot, ExpectedRotSpeed),
			ActualRot >= ExpectedRotSpeed * 0.92f && ActualRot <= ExpectedRotSpeed * 1.08f);

		S.Destroy();
	}

	return true;
}

// Peachone / EbonyWings / Vandalier: 軌道回転速度 = CachedOrbitRotSpeed × SpeedMult
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSurvivorsPeachoneOrbitSpeedBracerPassive,
	"ReinBalance.Survivors.Passive.BracerLv1to5.Peachone_OrbitSpeed",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FSurvivorsPeachoneOrbitSpeedBracerPassive::RunTest(const FString& Parameters)
{
	// PeachoneTable Lv1 OrbitRotSpeed = 0.8 rad/s
	// EbonyWings: 同一テーブル使用・逆回転、Vandalier: OrbitRotSpeed 同一
	static constexpr float BaseOrbitRotSpeed = 0.8f;

	struct FCase
	{
		EWeaponType Type;
		const TCHAR* Label;
	};

	const FCase Cases[] = {
		{ EWeaponType::Peachone,   TEXT("Peachone")   },
		{ EWeaponType::EbonyWings, TEXT("EbonyWings") },
		{ EWeaponType::Vandalier,  TEXT("Vandalier")  },
	};

	for (const FCase& Case : Cases)
	{
		for (int32 BracerLv = 1; BracerLv <= 5; ++BracerLv)
		{
			FSurvivorsTestWorld S;
			if (!TestTrue(FString::Printf(TEXT("[%s BracerLv%d] world"), Case.Label, BracerLv), S.Create()))
				return false;

			SetupBracerPassive(S.Game, BracerLv);
			FSurvivorsGameTestAccess::PlayerPos(S.Game) = FVector2D::ZeroVector;

			EquipTestWeapon(S.Game, Case.Type, 1);
			auto* WC = FSurvivorsGameTestAccess::WeaponComp(S.Game);

			WC->TickWeapons(SurvivorsGameConstants::PhysicsDt);
			if (!TestTrue(FString::Printf(TEXT("[%s BracerLv%d] orbit orb exists"), Case.Label, BracerLv),
				WC->GetOrbitOrbCount() > 0))
			{
				S.Destroy();
				return false;
			}

			const FVector2D PlayerPos = FSurvivorsGameTestAccess::PlayerPos(S.Game);
			const FVector2D StartPos  = WC->GetOrbitOrbPos(0);
			const float StartAngle    = FMath::Atan2(StartPos.Y - PlayerPos.Y, StartPos.X - PlayerPos.X);

			const float Elapsed = TickTestWeaponsForSecondsMeasured(WC, 1.0f);

			const FVector2D EndPos  = WC->GetOrbitOrbPos(0);
			const float EndAngle    = FMath::Atan2(EndPos.Y - PlayerPos.Y, EndPos.X - PlayerPos.X);
			const float ActualRot   = FMath::Abs(FMath::FindDeltaAngleRadians(StartAngle, EndAngle)) / Elapsed;

			const float SpeedMult          = 1.0f + 0.10f * static_cast<float>(BracerLv);
			const float ExpectedOrbitSpeed = BaseOrbitRotSpeed * SpeedMult;
			TestTrue(FString::Printf(TEXT("[%s BracerLv%d] orbit %.3frad/s ≈ %.3frad/s (±8%%)"),
				Case.Label, BracerLv, ActualRot, ExpectedOrbitSpeed),
				ActualRot >= ExpectedOrbitSpeed * 0.92f && ActualRot <= ExpectedOrbitSpeed * 1.08f);

			S.Destroy();
		}
	}

	return true;
}
