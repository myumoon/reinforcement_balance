/**
 * Survivors HTTP /params の loadout 境界を検証する。
 * 初心者向け: 不正 payload が facade と canonical Logic を部分更新しないことを確認する。
 */
#include "Misc/AutomationTest.h"

#include "Engine/Engine.h"
#include "Engine/World.h"
#include "Training/SurvivorsHttpEnvService.h"

/**
 * weapon/passive の全不正 loadout が HTTP error となり atomic に拒否されるテスト。
 * 初心者向け: ID・重複・slot 上限・level 上限の兄弟経路を同じ境界で網羅する。
 */
IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FSurvivorsHttpInitialLoadoutRejectsAtomically,
	"ReinBalance.Survivors.Http.InitialLoadoutRejectsAtomically",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

/**
 * 検証済み params だけが facade と Logic の両方へ反映されることを確認する。
 * 初心者向け: 各反例に別パラメータも混ぜ、loadout error 前の部分 mutation を検出する。
 */
bool FSurvivorsHttpInitialLoadoutRejectsAtomically::RunTest(const FString& Parameters)
{
	if (!GEngine)
	{
		AddError(TEXT("GEngine is required"));
		return false;
	}

	UWorld* World = UWorld::CreateWorld(EWorldType::Game, false);
	if (!World)
	{
		AddError(TEXT("test world could not be created"));
		return false;
	}
	FWorldContext& Context = GEngine->CreateNewWorldContext(EWorldType::Game);
	Context.SetCurrentWorld(World);
	World->InitializeActorsForPlay(FURL(), true);
	FActorSpawnParameters SpawnParameters;
	SpawnParameters.bNoFail = true;
	ASurvivorsGame* Game = World->SpawnActor<ASurvivorsGame>(SpawnParameters);
	if (!IsValid(Game))
	{
		GEngine->DestroyWorldContext(World);
		World->DestroyWorld(false);
		AddError(TEXT("Survivors game could not be spawned"));
		return false;
	}

	const FString ValidResponse =
		ASurvivorsHttpEnvService::ApplyParamsToGameForTesting(
			Game,
			TEXT("{\"MaxActiveEnemies\":17,\"initial_elapsed_time\":600,"
				"\"initial_weapon_slots\":[{\"weapon_id\":1,\"level\":2}],"
				"\"initial_passive_slots\":[{\"passive_id\":1,\"level\":2}]}"));
	TestEqual(TEXT("valid loadout response"), ValidResponse, FString(TEXT("{\"status\":\"ok\"}")));

	const TArray<FString> InvalidPayloads = {
		TEXT("{\"MaxActiveEnemies\":99,\"initial_elapsed_time\":1801}"),
		TEXT("{\"MaxActiveEnemies\":99,\"initial_weapon_slots\":[{\"weapon_id\":999,\"level\":1}]}"),
		TEXT("{\"MaxActiveEnemies\":99,\"initial_weapon_slots\":[{\"weapon_id\":1,\"level\":1},{\"weapon_id\":1,\"level\":2}]}"),
		TEXT("{\"MaxActiveEnemies\":99,\"initial_weapon_slots\":[{\"weapon_id\":1,\"level\":1},{\"weapon_id\":2,\"level\":1},{\"weapon_id\":3,\"level\":1},{\"weapon_id\":4,\"level\":1},{\"weapon_id\":5,\"level\":1},{\"weapon_id\":6,\"level\":1},{\"weapon_id\":7,\"level\":1}]}"),
		TEXT("{\"MaxActiveEnemies\":99,\"initial_weapon_slots\":[{\"weapon_id\":1,\"level\":9}]}"),
		TEXT("{\"MaxActiveEnemies\":99,\"initial_weapon_slots\":[{\"weapon_id\":1,\"level\":1.5}]}"),
		TEXT("{\"MaxActiveEnemies\":99,\"initial_weapon_slots\":[{\"weapon_id\":1}]}"),
		TEXT("{\"MaxActiveEnemies\":99,\"initial_passive_slots\":[{\"passive_id\":999,\"level\":1}]}"),
		TEXT("{\"MaxActiveEnemies\":99,\"initial_passive_slots\":[{\"passive_id\":1,\"level\":1},{\"passive_id\":1,\"level\":2}]}"),
		TEXT("{\"MaxActiveEnemies\":99,\"initial_passive_slots\":[{\"passive_id\":1,\"level\":1},{\"passive_id\":2,\"level\":1},{\"passive_id\":3,\"level\":1},{\"passive_id\":4,\"level\":1},{\"passive_id\":5,\"level\":1},{\"passive_id\":6,\"level\":1},{\"passive_id\":7,\"level\":1}]}"),
		TEXT("{\"MaxActiveEnemies\":99,\"initial_passive_slots\":[{\"passive_id\":1,\"level\":99}]}"),
	};
	for (const FString& Payload : InvalidPayloads)
	{
		const FString Response =
			ASurvivorsHttpEnvService::ApplyParamsToGameForTesting(Game, Payload);
		TestTrue(TEXT("invalid loadout returns HTTP error body"), Response.StartsWith(TEXT("{\"error\"")));
		TestEqual(TEXT("facade scalar remains unchanged"), Game->MaxActiveEnemies, 17);
		TestEqual(TEXT("facade weapon count remains unchanged"), Game->InitialWeaponSlots.Num(), 1);
		TestEqual(TEXT("facade passive count remains unchanged"), Game->InitialPassiveSlots.Num(), 1);
		TestEqual(TEXT("facade weapon id remains unchanged"), Game->InitialWeaponSlots[0].WeaponId, 1);
		TestEqual(TEXT("facade weapon level remains unchanged"), Game->InitialWeaponSlots[0].Level, 2);
		TestEqual(TEXT("facade passive id remains unchanged"), Game->InitialPassiveSlots[0].PassiveId, 1);
		TestEqual(TEXT("facade passive level remains unchanged"), Game->InitialPassiveSlots[0].Level, 2);
	}

	Game->GetLogic()->Reset(73013);
	TestEqual(
		TEXT("Logic retains valid weapon type"),
		static_cast<int32>(Game->GetLogic()->GetWeaponSlot(0).Type),
		static_cast<int32>(EWeaponType::Garlic));
	TestEqual(TEXT("Logic retains valid weapon level"), Game->GetLogic()->GetWeaponSlot(0).Level.Value, 2);
	TestEqual(
		TEXT("Logic retains valid passive type"),
		static_cast<int32>(Game->GetLogic()->GetPassiveSlot(0).Type),
		static_cast<int32>(EPassiveItemType::Spinach));
	TestEqual(TEXT("Logic retains valid passive level"), Game->GetLogic()->GetPassiveSlot(0).Level, 2);

	GEngine->DestroyWorldContext(World);
	World->DestroyWorld(false);
	return true;
}
