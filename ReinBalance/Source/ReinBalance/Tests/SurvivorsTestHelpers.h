#pragma once
// AutomationTest.h は各 .cpp で include する（マクロ展開が TU ごとに必要）
#include "Engine/World.h"
#include "Survivors/Game/SurvivorsGame.h"
#include "Survivors/SurvivorsGameLogic.h"
#include "Survivors/SurvivorsTypes.h"
#include "Survivors/SurvivorsGameConstants.h"
#include "Survivors/Game/SurvivorsCollisionComponent.h"
#include "Survivors/Game/SurvivorsEnemyComponent.h"
#include "Survivors/Game/SurvivorsGemComponent.h"
#include "Survivors/Game/SurvivorsPickupComponent.h"
#include "Survivors/Game/SurvivorsPlayerComponent.h"
#include "Survivors/Game/Weapons/SurvivorsWeaponComponent.h"

// ============================================================
// テストヘルパー: FSurvivorsGameLogic の private 状態・メソッドへのアクセサ
// (SurvivorsGameLogic.h の WITH_AUTOMATION_TESTS friend 宣言が必要)
// ============================================================
struct FSurvivorsGameTestAccess
{
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
	static float& GlobalFreezeUntilTime(ASurvivorsGame* G){ return G->GetLogic()->GlobalFreezeUntilTime; }
	static float& GemPickupRadius(ASurvivorsGame* G)       { return G->GetLogic()->CurrentConfig.GemPickupRadius; }
	static int32& NextEnemyId(ASurvivorsGame* G)           { return G->GetLogic()->NextEnemyId; }
	static int32& NextGemId(ASurvivorsGame* G)             { return G->GetLogic()->NextGemId; }
	static FWeaponSlot* WeaponSlots(ASurvivorsGame* G)     { return G->GetLogic()->WeaponSlots; }
	static FPassiveSlot* PassiveSlots(ASurvivorsGame* G)   { return G->GetLogic()->PassiveSlots; }
	static FPassiveEffects& PassiveEffects(ASurvivorsGame* G) { return G->GetLogic()->CachedPassiveEffects; }

	// Legacy actor/component state accessors.
	static TArray<FEnemyState>& ActorEnemies(ASurvivorsGame* G) { return G->Enemies; }
	static TArray<FGemState>& ActorGems(ASurvivorsGame* G) { return G->Gems; }
	static TArray<FFloorPickupState>& ActorFloorPickups(ASurvivorsGame* G) { return G->FloorPickups; }
	static TArray<FSpecialPickupState>& ActorSpecialPickups(ASurvivorsGame* G) { return G->SpecialPickups; }
	static TArray<FDestructibleState>& ActorDestructibles(ASurvivorsGame* G) { return G->Destructibles; }
	static FVector2D& ActorPlayerPos(ASurvivorsGame* G) { return G->PlayerPos; }
	static FVector2D& ActorPlayerVel(ASurvivorsGame* G) { return G->PlayerVel; }
	static float& ActorPlayerHP(ASurvivorsGame* G) { return G->PlayerHP; }
	static float& ActorPlayerXP(ASurvivorsGame* G) { return G->PlayerXP; }
	static int32& ActorPlayerLevel(ASurvivorsGame* G) { return G->PlayerLevel; }
	static FWeaponSlot* ActorWeaponSlots(ASurvivorsGame* G) { return G->WeaponSlots; }
	static FPassiveSlot* ActorPassiveSlots(ASurvivorsGame* G) { return G->PassiveSlots; }
	static FPassiveEffects& ActorPassiveEffects(ASurvivorsGame* G) { return G->CachedPassiveEffects; }
	static float& ActorGlobalFreezeUntilTime(ASurvivorsGame* G) { return G->GlobalFreezeUntilTime; }
	static float& ActorPlayerShieldTimer(ASurvivorsGame* G) { return G->PlayerShieldTimer; }
	static bool& ActorShieldActive(ASurvivorsGame* G) { return G->bShieldActive; }
	static int32& ActorMaxRevivalCount(ASurvivorsGame* G) { return G->MaxRevivalCount; }
	static int32& ActorUsedRevivalCount(ASurvivorsGame* G) { return G->UsedRevivalCount; }
	static int32& ActorNextEnemyId(ASurvivorsGame* G) { return G->NextEnemyId; }
	static int32& ActorNextGemId(ASurvivorsGame* G) { return G->NextGemId; }
	static float& ActorElapsedTime(ASurvivorsGame* G) { return G->ElapsedTime; }
	static float& ActorSpawnAccumulator(ASurvivorsGame* G) { return G->SpawnAccumulator; }
	static bool& ActorBossSpawned(ASurvivorsGame* G) { return G->bBossSpawned; }
	static float& ActorLastReward(ASurvivorsGame* G) { return G->LastReward; }
	static float& ActorEpisodeBaseReward(ASurvivorsGame* G) { return G->EpisodeBaseReward; }
	static int32& ActorEpisodeStepCount(ASurvivorsGame* G) { return G->EpisodeStepCount; }
	static bool& ActorDone(ASurvivorsGame* G) { return G->bDone; }
	static bool& ActorTruncated(ASurvivorsGame* G) { return G->bTruncated; }
	static float& ActorPhysicsAccumTime(ASurvivorsGame* G) { return G->PhysicsAccumTime; }
	static FSurvivorsSpawnDebug& ActorLastSpawnDebug(ASurvivorsGame* G) { return G->LastSpawnDebug; }

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

	static FSurvivorsGameLogic* GetLogic(ASurvivorsGame* G) { return G->GetLogic(); }

	// Logic プライベートメソッドのラッパー（FSurvivorsTestWorld から呼ぶ用）
	static void RecalcPassiveEffects(ASurvivorsGame* G)                         { G->GetLogic()->RecalcPassiveEffects(); }
	static void ProcessXPGain(ASurvivorsGame* G, float Amount)                  { G->GetLogic()->ProcessXPGain(Amount); }
	static void CheckSpecialPickups(ASurvivorsGame* G)                          { G->GetLogic()->CheckSpecialPickups(); }
	static void BuildEnemyGrid(ASurvivorsGame* G)                               { G->GetLogic()->BuildEnemyGrid(); }
	static void BuildPickupGrid(ASurvivorsGame* G)                              { G->GetLogic()->BuildPickupGrid(); }
	static void TickWeapons(ASurvivorsGame* G, float Dt)                        { G->GetLogic()->TickWeapons(Dt); }
	static void ComputeAllWeaponHits(ASurvivorsGame* G, FSurvivorsHitFrame& HF) { G->GetLogic()->ComputeAllWeaponHits(HF); }
	static void ApplyWeaponHits(ASurvivorsGame* G, FSurvivorsHitFrame& HF)      { G->GetLogic()->ApplyWeaponHits(HF); }
	static void ComputeContactHits(ASurvivorsGame* G, FSurvivorsHitFrame& HF)   { G->GetLogic()->ComputeContactHits(HF); }
	static void ApplyContactHits(ASurvivorsGame* G, FSurvivorsHitFrame& HF)     { G->GetLogic()->ApplyContactHits(HF); }
	static void ComputePickupHits(ASurvivorsGame* G, FSurvivorsHitFrame& HF)    { G->GetLogic()->ComputePickupHits(HF); }
	static void ApplyPickupHits(ASurvivorsGame* G, FSurvivorsHitFrame& HF)      { G->GetLogic()->ApplyPickupHits(HF); }
	static void UpdateEnemies(ASurvivorsGame* G)                                { G->GetLogic()->UpdateEnemies(); }
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

	void RunUpdateEnemies()
	{
		FSurvivorsGameTestAccess::UpdateEnemies(Game);
	}
};

static void EquipTestWeapon(ASurvivorsGame* Game, EWeaponType Type, int32 Level)
{
	FWeaponSlot& Slot = FSurvivorsGameTestAccess::WeaponSlots(Game)[0];
	Slot.Type  = Type;
	Slot.Level = FWeaponLevel(Level);
	Game->GetLogic()->EquipWeapon(0, Type, Level);
}

static int32 SurvivorsStepsForSeconds(float Seconds)
{
	return FMath::CeilToInt(Seconds / SurvivorsGameConstants::PhysicsDt);
}

static void TickTestWeaponsForSteps(ASurvivorsGame* Game, int32 Steps)
{
	for (int32 i = 0; i < Steps; ++i)
	{
		FSurvivorsGameTestAccess::TickWeapons(Game, SurvivorsGameConstants::PhysicsDt);
	}
}

static void TickTestWeaponsForSeconds(ASurvivorsGame* Game, float Seconds)
{
	TickTestWeaponsForSteps(Game, SurvivorsStepsForSeconds(Seconds));
}

static float TickTestWeaponsForSecondsMeasured(ASurvivorsGame* Game, float Seconds)
{
	const int32 Steps = SurvivorsStepsForSeconds(Seconds);
	TickTestWeaponsForSteps(Game, Steps);
	return Steps * SurvivorsGameConstants::PhysicsDt;
}
