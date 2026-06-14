#pragma once
// AutomationTest.h は各 .cpp で include する（マクロ展開が TU ごとに必要）
#include "Engine/World.h"
#include "Survivors/Logic/SurvivorsGame.h"
#include "Survivors/Logic/SurvivorsTypes.h"
#include "Survivors/Logic/SurvivorsGameConstants.h"
#include "Survivors/Logic/SurvivorsCollisionComponent.h"
#include "Survivors/Logic/SurvivorsEnemyComponent.h"
#include "Survivors/Logic/SurvivorsGemComponent.h"
#include "Survivors/Logic/SurvivorsPickupComponent.h"
#include "Survivors/Logic/SurvivorsPlayerComponent.h"
#include "Survivors/Logic/Weapons/SurvivorsWeaponComponent.h"

// ============================================================
// テストヘルパー: ASurvivorsGame の private 状態へのアクセサ
// (SurvivorsGame.h の WITH_AUTOMATION_TESTS friend 宣言が必要)
// ============================================================
struct FSurvivorsGameTestAccess
{
	static TArray<FEnemyState>& Enemies(ASurvivorsGame* G) { return G->Enemies; }
	static TArray<FGemState>&   Gems(ASurvivorsGame* G)    { return G->Gems; }
	static TArray<FSpecialPickupState>& SpecialPickups(ASurvivorsGame* G) { return G->SpecialPickups; }
	static float& PlayerHP(ASurvivorsGame* G)              { return G->PlayerHP; }
	static float& PlayerXP(ASurvivorsGame* G)              { return G->PlayerXP; }
	static FVector2D& PlayerPos(ASurvivorsGame* G)         { return G->PlayerPos; }
	static FVector2D& PlayerVel(ASurvivorsGame* G)         { return G->PlayerVel; }
	static float& PlayerRadius(ASurvivorsGame* G)          { return G->PlayerRadius; }
	static bool& bShieldActive(ASurvivorsGame* G)          { return G->bShieldActive; }
	static float& LastReward(ASurvivorsGame* G)            { return G->LastReward; }
	static float& ElapsedTime(ASurvivorsGame* G)           { return G->ElapsedTime; }
	static float& GemPickupRadius(ASurvivorsGame* G)       { return G->GemPickupRadius; }
	static int32& NextEnemyId(ASurvivorsGame* G)           { return G->NextEnemyId; }
	static int32& NextGemId(ASurvivorsGame* G)             { return G->NextGemId; }
	static FWeaponSlot* WeaponSlots(ASurvivorsGame* G)     { return G->WeaponSlots; }
	static FPassiveSlot* PassiveSlots(ASurvivorsGame* G)   { return G->PassiveSlots; }
	static FPassiveEffects& PassiveEffects(ASurvivorsGame* G) { return G->CachedPassiveEffects; }

	static USurvivorsCollisionComponent* CollComp(ASurvivorsGame* G) { return G->CollisionComponent; }
	static USurvivorsEnemyComponent*     EnemyComp(ASurvivorsGame* G){ return G->EnemyComponent; }
	static USurvivorsGemComponent*       GemComp(ASurvivorsGame* G)  { return G->GemComponent; }
	static USurvivorsPickupComponent*    PickupComp(ASurvivorsGame* G){ return G->PickupComponent; }
	static USurvivorsPlayerComponent*    PlayerComp(ASurvivorsGame* G){ return G->PlayerComponent; }
	static USurvivorsWeaponComponent*    WeaponComp(ASurvivorsGame* G){ return G->WeaponComponent; }

	static float XPRequiredForLevel(ASurvivorsGame* G, int32 Level) { return G->XPRequiredForLevel(Level); }

	static void FinalizePendingEnemies(ASurvivorsGame* G)  { G->FinalizePendingEnemies(); }
	static void FinalizePickupRemovals(ASurvivorsGame* G)  { G->FinalizePickupRemovals(); }
};

// ============================================================
// テストワールドライフサイクル
// ============================================================
struct FSurvivorsTestWorld
{
	UWorld*       World = nullptr;
	ASurvivorsGame* Game = nullptr;

	bool Create()
	{
		if (!GEngine) return false;
		World = UWorld::CreateWorld(EWorldType::Game, false);
		if (!World) return false;
		FWorldContext& Ctx = GEngine->CreateNewWorldContext(EWorldType::Game);
		Ctx.SetCurrentWorld(World);
		World->InitializeActorsForPlay(FURL(), true);
		FActorSpawnParameters P;
		P.bNoFail = true;
		Game = World->SpawnActor<ASurvivorsGame>(P);
		if (!IsValid(Game)) return false;
		// BeginPlay が呼ばれない場合のフォールバック（テスト環境ではタイミングが異なる）
		// ResetState は public なので常に明示的に呼び出して初期状態を保証する
		Game->ResetState(TOptional<int32>());
		return true;
	}

	void Destroy()
	{
		if (World && GEngine)
		{
			GEngine->DestroyWorldContext(World);
			World->DestroyWorld(false);
		}
		World = nullptr;
		Game = nullptr;
	}

	// プレイヤー位置に近い敵を追加する
	int32 AddEnemyAt(FVector2D Pos, float HP = 100.f, float ContactDamage = 5.f, float Radius = 5.f)
	{
		FEnemyState E;
		E.UniqueId        = ++FSurvivorsGameTestAccess::NextEnemyId(Game);
		E.Pos             = Pos;
		E.CollisionRadius = Radius;
		E.HP              = HP;
		E.MaxHP           = HP;
		E.ContactDamage   = ContactDamage;
		E.TypeId          = 0;
		FSurvivorsGameTestAccess::Enemies(Game).Add(E);
		return E.UniqueId;
	}

	int32 AddEnemyNearPlayer(float HP = 100.f, float ContactDamage = 5.f, float Radius = 5.f)
	{
		return AddEnemyAt(FSurvivorsGameTestAccess::PlayerPos(Game), HP, ContactDamage, Radius);
	}

	// Weapon/Enemy/Gem の一連の当たり判定ステップを実行するヘルパー
	void RunWeaponHits()
	{
		auto* CC = FSurvivorsGameTestAccess::CollComp(Game);
		auto* WC = FSurvivorsGameTestAccess::WeaponComp(Game);
		CC->BuildEnemyGrid();
		FSurvivorsHitFrame HF;
		WC->ComputeAllWeaponHits(CC, HF);
		WC->ApplyWeaponHits(HF);
	}

	void RunContactHits()
	{
		auto* CC = FSurvivorsGameTestAccess::CollComp(Game);
		auto* EC = FSurvivorsGameTestAccess::EnemyComp(Game);
		CC->BuildEnemyGrid();
		FSurvivorsHitFrame HF;
		EC->ComputeContactHits(CC, HF);
		EC->ApplyContactHits(HF);
	}

	void RunPickupHits()
	{
		auto* CC = FSurvivorsGameTestAccess::CollComp(Game);
		auto* GC = FSurvivorsGameTestAccess::GemComp(Game);
		CC->BuildPickupGrid();
		FSurvivorsHitFrame HF;
		GC->ComputePickupHits(CC, HF);
		GC->ApplyPickupHits(HF);
	}
};

static void EquipTestWeapon(ASurvivorsGame* Game, EWeaponType Type, int32 Level)
{
	FWeaponSlot* Weapons = FSurvivorsGameTestAccess::WeaponSlots(Game);
	Weapons[0].Type = Type;
	Weapons[0].Level = FWeaponLevel(Level);
	FSurvivorsGameTestAccess::WeaponComp(Game)->EquipWeapon(0, Type, Level);
}

static int32 SurvivorsStepsForSeconds(float Seconds)
{
	return FMath::CeilToInt(Seconds / SurvivorsGameConstants::PhysicsDt);
}

static void TickTestWeaponsForSteps(USurvivorsWeaponComponent* WC, int32 Steps)
{
	for (int32 i = 0; i < Steps; ++i)
	{
		WC->TickWeapons(SurvivorsGameConstants::PhysicsDt);
	}
}

static void TickTestWeaponsForSeconds(USurvivorsWeaponComponent* WC, float Seconds)
{
	TickTestWeaponsForSteps(WC, SurvivorsStepsForSeconds(Seconds));
}

static float TickTestWeaponsForSecondsMeasured(USurvivorsWeaponComponent* WC, float Seconds)
{
	const int32 Steps = SurvivorsStepsForSeconds(Seconds);
	TickTestWeaponsForSteps(WC, Steps);
	return Steps * SurvivorsGameConstants::PhysicsDt;
}
