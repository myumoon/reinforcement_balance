#pragma once
// AutomationTest.h は各 .cpp で include する（マクロ展開が TU ごとに必要）
#include "Engine/World.h"
#include "Survivors/Logic/SurvivorsGame.h"
#include "Survivors/Logic/SurvivorsGameLogic.h"
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
//
// Phase 2: 状態フィールドは ASurvivorsGame に残っているため
//          既存アクセサはそのまま維持する。
//          Phase 3 で FSurvivorsGameLogic に状態が移った後、
//          G->GetLogic()->XXX 経由に切り替える。
// ============================================================
struct FSurvivorsGameTestAccess
{
	// Phase 3: 状態フィールドは FSurvivorsGameLogic を経由してアクセス
	static TArray<FEnemyState>& Enemies(ASurvivorsGame* G) { return G->GetLogic()->Enemies; }
	static TArray<FGemState>&   Gems(ASurvivorsGame* G)    { return G->GetLogic()->Gems; }
	static TArray<FSpecialPickupState>& SpecialPickups(ASurvivorsGame* G) { return G->GetLogic()->SpecialPickups; }
	static float& PlayerHP(ASurvivorsGame* G)              { return G->GetLogic()->PlayerHP; }
	static float& PlayerXP(ASurvivorsGame* G)              { return G->GetLogic()->PlayerXP; }
	static FVector2D& PlayerPos(ASurvivorsGame* G)         { return G->GetLogic()->PlayerPos; }
	static FVector2D& PlayerVel(ASurvivorsGame* G)         { return G->GetLogic()->PlayerVel; }
	static float& PlayerRadius(ASurvivorsGame* G)          { return G->GetLogic()->CurrentConfig.PlayerRadius; }
	static bool& bShieldActive(ASurvivorsGame* G)          { return G->GetLogic()->bShieldActive; }
	static float& LastReward(ASurvivorsGame* G)            { return G->GetLogic()->LastReward; }
	static float& ElapsedTime(ASurvivorsGame* G)           { return G->GetLogic()->ElapsedTime; }
	static float& GemPickupRadius(ASurvivorsGame* G)       { return G->GetLogic()->CurrentConfig.GemPickupRadius; }
	static int32& NextEnemyId(ASurvivorsGame* G)           { return G->GetLogic()->NextEnemyId; }
	static int32& NextGemId(ASurvivorsGame* G)             { return G->GetLogic()->NextGemId; }
	static FWeaponSlot* WeaponSlots(ASurvivorsGame* G)     { return G->GetLogic()->WeaponSlots; }
	static FPassiveSlot* PassiveSlots(ASurvivorsGame* G)   { return G->GetLogic()->PassiveSlots; }
	static FPassiveEffects& PassiveEffects(ASurvivorsGame* G) { return G->GetLogic()->CachedPassiveEffects; }

	// コンポーネントアクセサ（既存テストが使用するため維持）
	static USurvivorsCollisionComponent* CollComp(ASurvivorsGame* G) { return G->CollisionComponent; }
	static USurvivorsEnemyComponent*     EnemyComp(ASurvivorsGame* G){ return G->EnemyComponent; }
	static USurvivorsGemComponent*       GemComp(ASurvivorsGame* G)  { return G->GemComponent; }
	static USurvivorsPickupComponent*    PickupComp(ASurvivorsGame* G){ return G->PickupComponent; }
	static USurvivorsPlayerComponent*    PlayerComp(ASurvivorsGame* G){ return G->PlayerComponent; }
	static USurvivorsWeaponComponent*    WeaponComp(ASurvivorsGame* G){ return G->WeaponComponent; }

	static float XPRequiredForLevel(ASurvivorsGame* G, int32 Level) { return G->GetLogic()->XPRequiredForLevel(Level); }
	static void FinalizePendingEnemies(ASurvivorsGame* G)  { G->GetLogic()->FinalizePendingEnemies(); }
	static void FinalizePickupRemovals(ASurvivorsGame* G)  { G->GetLogic()->FinalizePickupRemovals(); }

	// Phase 2 追加: FSurvivorsGameLogic へのアクセサ（テスト用）
	static FSurvivorsGameLogic* GetLogic(ASurvivorsGame* G) { return G->GetLogic(); }

	// Logic プライベートメソッドのラッパー（FSurvivorsTestWorld から呼ぶ用）
	static void BuildEnemyGrid(ASurvivorsGame* G)                               { G->GetLogic()->BuildEnemyGrid(); }
	static void BuildPickupGrid(ASurvivorsGame* G)                              { G->GetLogic()->BuildPickupGrid(); }
	static void ComputeAllWeaponHits(ASurvivorsGame* G, FSurvivorsHitFrame& HF) { G->GetLogic()->ComputeAllWeaponHits(HF); }
	static void ApplyWeaponHits(ASurvivorsGame* G, FSurvivorsHitFrame& HF)      { G->GetLogic()->ApplyWeaponHits(HF); }
	static void ComputeContactHits(ASurvivorsGame* G, FSurvivorsHitFrame& HF)   { G->GetLogic()->ComputeContactHits(HF); }
	static void ApplyContactHits(ASurvivorsGame* G, FSurvivorsHitFrame& HF)     { G->GetLogic()->ApplyContactHits(HF); }
	static void ComputePickupHits(ASurvivorsGame* G, FSurvivorsHitFrame& HF)    { G->GetLogic()->ComputePickupHits(HF); }
	static void ApplyPickupHits(ASurvivorsGame* G, FSurvivorsHitFrame& HF)      { G->GetLogic()->ApplyPickupHits(HF); }
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
		FSurvivorsHitFrame HF;
		FSurvivorsGameTestAccess::BuildEnemyGrid(Game);
		FSurvivorsGameTestAccess::ComputeAllWeaponHits(Game, HF);
		FSurvivorsGameTestAccess::ApplyWeaponHits(Game, HF);
	}

	void RunContactHits()
	{
		FSurvivorsHitFrame HF;
		FSurvivorsGameTestAccess::BuildEnemyGrid(Game);
		FSurvivorsGameTestAccess::ComputeContactHits(Game, HF);
		FSurvivorsGameTestAccess::ApplyContactHits(Game, HF);
	}

	void RunPickupHits()
	{
		FSurvivorsHitFrame HF;
		FSurvivorsGameTestAccess::BuildPickupGrid(Game);
		FSurvivorsGameTestAccess::ComputePickupHits(Game, HF);
		FSurvivorsGameTestAccess::ApplyPickupHits(Game, HF);
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
