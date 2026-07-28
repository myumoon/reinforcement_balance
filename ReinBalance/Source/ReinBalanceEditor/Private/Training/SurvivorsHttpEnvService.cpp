/**
 * Survivors HTTP transport の MPSC enqueue と game-thread 適用を実装する。
 * 初心者向け: worker thread は JSON の形だけを検査し、ゲーム状態の読書きは Tick に限定する。
 */
#include "Training/SurvivorsHttpEnvService.h"
#include "HttpEnvServerBase.h"
#include "HttpServerResponse.h"
#include "Kismet/GameplayStatics.h"
#include "Dom/JsonObject.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"

namespace
{
FString SerializeJsonObject(const TSharedRef<FJsonObject>& Object)
{
	FString Json;
	const TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&Json);
	FJsonSerializer::Serialize(Object, Writer);
	return Json;
}

TUniquePtr<FHttpServerResponse> MakeStatusJsonResponse(
	const FString& Json, EHttpServerResponseCodes Code)
{
	TUniquePtr<FHttpServerResponse> Response =
		FHttpServerResponse::Create(Json, TEXT("application/json"));
	Response->Code = Code;
	return Response;
}

TSharedRef<FJsonObject> BuildLevelUpInfoObject(
	const FSurvivorsPendingLevelUpDecision& Pending,
	int32 Backlog,
	const FString* SpawnDebugJson)
{
	TSharedRef<FJsonObject> Info = MakeShared<FJsonObject>();
	if (SpawnDebugJson)
	{
		TSharedPtr<FJsonObject> SpawnDebug;
		const TSharedRef<TJsonReader<>> Reader =
			TJsonReaderFactory<>::Create(*SpawnDebugJson);
		if (!FJsonSerializer::Deserialize(Reader, SpawnDebug) || !SpawnDebug.IsValid())
		{
			SpawnDebug = MakeShared<FJsonObject>();
		}
		Info->SetObjectField(TEXT("spawn_debug"), SpawnDebug);
	}

	Info->SetBoolField(TEXT("level_up_pending"), Pending.IsSet());
	Info->SetStringField(TEXT("level_up_decision_id"), Pending.DecisionId);
	Info->SetNumberField(TEXT("level_up_player_level"), Pending.PlayerLevel);
	Info->SetNumberField(TEXT("level_up_backlog"), Backlog);

	TArray<TSharedPtr<FJsonValue>> ChoicesJson;
	ChoicesJson.Reserve(Pending.Choices.Num());
	for (const FSurvivorsLevelUpChoiceOffer& Offer : Pending.Choices)
	{
		const FLevelUpChoice& Choice = Offer.Choice;
		TSharedRef<FJsonObject> ChoiceJson = MakeShared<FJsonObject>();
		ChoiceJson->SetStringField(TEXT("choice_id"), Offer.ChoiceId);
		ChoiceJson->SetStringField(
			TEXT("type"), SurvivorsLevelUpChoiceTypeToString(Choice.ChoiceType));
		const bool bWeapon = Choice.WeaponType != EWeaponType::None;
		ChoiceJson->SetStringField(
			TEXT("item_kind"), bWeapon ? TEXT("weapon") : TEXT("passive"));
		ChoiceJson->SetNumberField(
			TEXT("item_id"),
			bWeapon
				? static_cast<int32>(Choice.WeaponType)
				: static_cast<int32>(Choice.PassiveType));
		ChoiceJson->SetNumberField(TEXT("slot_index"), Choice.SlotIdx);
		ChoiceJson->SetNumberField(TEXT("new_level"), Choice.NewLevel);
		ChoicesJson.Add(MakeShared<FJsonValueObject>(ChoiceJson));
	}
	Info->SetArrayField(TEXT("level_up_choices"), MoveTemp(ChoicesJson));
	return Info;
}

FString BuildLevelUpApplyResponseJson(
	const FSurvivorsLevelUpApplyResult& Result,
	const FString& ObsSchemaHash)
{
	TSharedRef<FJsonObject> Response = MakeShared<FJsonObject>();
	Response->SetStringField(TEXT("status"), TEXT("applied"));
	Response->SetStringField(TEXT("decision_id"), Result.DecisionId);
	Response->SetStringField(TEXT("choice_id"), Result.ChoiceId);
	Response->SetStringField(TEXT("obs_schema_hash"), ObsSchemaHash);

	TArray<TSharedPtr<FJsonValue>> ObsJson;
	ObsJson.Reserve(Result.PostChoiceObs.Num());
	for (float Value : Result.PostChoiceObs)
	{
		ObsJson.Add(MakeShared<FJsonValueNumber>(Value));
	}
	Response->SetArrayField(TEXT("obs"), MoveTemp(ObsJson));
	Response->SetObjectField(
		TEXT("info"),
		BuildLevelUpInfoObject(Result.PendingAfter, Result.BacklogAfter, nullptr));
	return SerializeJsonObject(Response);
}
}

// ============================================================
// FSurvivorsEnvServer: FHttpEnvServerBase の Survivors 固有派生クラス
// ============================================================

class ASurvivorsHttpEnvService::FSurvivorsEnvServer : public FHttpEnvServerBase
{
public:
	explicit FSurvivorsEnvServer(ASurvivorsGame* InGame) : Game(InGame) {}

	// ---- obs_schema キャッシュ構築（BeginPlay / StartServer 前に呼ぶ） ----

	void BuildObsSchemaCache()
	{
		if (!Game)
		{
			CachedObsSchemaJson = TEXT("{\"error\":\"game not set\"}");
			return;
		}
		TArray<FSurvivorsObsSegment> Schema = Game->GetObsSchema();
		FString SegmentsStr;
		for (int32 i = 0; i < Schema.Num(); ++i)
		{
			SegmentsStr += FString::Printf(TEXT("{\"name\":\"%s\",\"dim\":%d}"),
				*Schema[i].Name, Schema[i].Dim);
			if (i < Schema.Num() - 1) SegmentsStr += TEXT(",");
		}
		CachedObsSchemaJson = FString::Printf(
			TEXT("{\"segments\":[%s],\"total_dim\":%d,\"obs_schema_hash\":\"%s\"}"),
			*SegmentsStr, Game->GetObsDim(), *Game->GetObsSchemaHash());
		CachedContentSchemaJson = FString::Printf(
			TEXT("{\"schema_version\":\"survivors.content_schema.v1\",\"content\":%s,\"action_time\":%s}"),
			*Game->GetContentSchema(), *Game->GetActionTimeSchema());
	}

	// ---- ParamsQueue 外部制御 API ----

	struct FParamsRequest
	{
		FString             JsonBody;
		FHttpResultCallback Callback;
	};

	struct FLevelUpChoiceRequest
	{
		FString DecisionId;
		FString ChoiceId;
		FHttpResultCallback Callback;
	};

	bool TakeParamsRequest(FString& OutJson, FHttpResultCallback& OutCallback)
	{
		FParamsRequest Req;
		if (!ParamsQueue.Dequeue(Req)) return false;
		OutJson     = MoveTemp(Req.JsonBody);
		OutCallback = MoveTemp(Req.Callback);
		return true;
	}

	bool TakeLevelUpChoiceRequest(
		FString& OutDecisionId,
		FString& OutChoiceId,
		FHttpResultCallback& OutCallback)
	{
		FLevelUpChoiceRequest Request;
		if (!LevelUpChoiceQueue.Dequeue(Request)) return false;
		OutDecisionId = MoveTemp(Request.DecisionId);
		OutChoiceId = MoveTemp(Request.ChoiceId);
		OutCallback = MoveTemp(Request.Callback);
		return true;
	}

	// ---- IHttpEnvServer 実装 ----

	virtual FEnvResetResult ProcessReset(TOptional<int32> Seed) override
	{
		FEnvResetResult Result;
		if (Game)
		{
			Game->ResetState(Seed);
			Result.Obs           = Game->GetObservation();
			Result.ObsSchemaHash = Game->GetObsSchemaHash();
		}
		return Result;
	}

	virtual FEnvStepResult ProcessStep(const TArray<float>& Action, int32 Steps) override
	{
		FEnvStepResult Result;
		if (Game)
		{
			const int32 ActionIdx = Action.Num() > 0
				? FMath::Clamp(static_cast<int32>(Action[0]), 0, 8)
				: 8;
			float AccumulatedReward = 0.f;
			for (int32 i = 0; i < Steps; ++i)
			{
				Game->PhysicsStep(ActionIdx);
				AccumulatedReward += Game->GetReward();
				if (Game->IsLevelUpPending())
				{
					break;
				}
				if (Game->IsDone())
				{
					Result.bDone = true;
					break;
				}
				if (Game->IsTruncated())
				{
					Result.bTruncated = true;
					break;
				}
			}
			Result.Obs      = Game->GetObservation();
			Result.Reward   = AccumulatedReward;
			const FSurvivorsPendingLevelUpDecision& Pending =
				Game->GetPendingLevelUpDecision();
			const FString SpawnDebugJson = Game->GetSpawnDebugJson();
			Result.InfoJson = SerializeJsonObject(BuildLevelUpInfoObject(
				Pending, Game->GetLevelUpBacklog(), &SpawnDebugJson));
		}
		return Result;
	}

protected:
	virtual void RegisterAdditionalRoutes(TSharedPtr<IHttpRouter> Router) override
	{
		ObsSchemaRoute = Router->BindRoute(
			FHttpPath(TEXT("/obs_schema")), EHttpServerRequestVerbs::VERB_GET,
			FHttpRequestHandler::CreateRaw(this, &FSurvivorsEnvServer::HandleObsSchema));
		ContentSchemaRoute = Router->BindRoute(
			FHttpPath(TEXT("/content_schema")), EHttpServerRequestVerbs::VERB_GET,
			FHttpRequestHandler::CreateRaw(this, &FSurvivorsEnvServer::HandleContentSchema));

		ParamsRoute = Router->BindRoute(
			FHttpPath(TEXT("/params")), EHttpServerRequestVerbs::VERB_POST,
			FHttpRequestHandler::CreateRaw(this, &FSurvivorsEnvServer::HandleParams));
		LevelUpChoiceRoute = Router->BindRoute(
			FHttpPath(TEXT("/level_up_choice")), EHttpServerRequestVerbs::VERB_POST,
			FHttpRequestHandler::CreateRaw(
				this, &FSurvivorsEnvServer::HandleLevelUpChoice));
	}

	virtual void UnregisterAdditionalRoutes(TSharedPtr<IHttpRouter> Router) override
	{
		if (Router)
		{
			Router->UnbindRoute(ObsSchemaRoute);
			Router->UnbindRoute(ContentSchemaRoute);
			Router->UnbindRoute(ParamsRoute);
			Router->UnbindRoute(LevelUpChoiceRoute);
		}
	}

private:
	// キャッシュ済み obs_schema JSON（HTTPワーカースレッドから Game に触れないよう事前構築）
	FString CachedObsSchemaJson;
	// HTTP worker が Game に触れないよう game thread で構築した content schema。
	FString CachedContentSchemaJson;

	// /params は Survivors 固有のためここで管理（FHttpEnvServerBase には追加しない）
	TQueue<FParamsRequest, EQueueMode::Mpsc> ParamsQueue;
	// worker thread は typed IDs を積むだけで Game へ直接アクセスしない。
	TQueue<FLevelUpChoiceRequest, EQueueMode::Mpsc> LevelUpChoiceQueue;

	bool HandleObsSchema(const FHttpServerRequest& Request, const FHttpResultCallback& OnComplete)
	{
		OnComplete(MakeJsonResponse(CachedObsSchemaJson));
		return true;
	}

	/** キャッシュ済み one-way schema を薄く返す。 */
	bool HandleContentSchema(const FHttpServerRequest& Request, const FHttpResultCallback& OnComplete)
	{
		OnComplete(MakeJsonResponse(CachedContentSchemaJson));
		return true;
	}

	bool HandleParams(const FHttpServerRequest& Request, const FHttpResultCallback& OnComplete)
	{
		// ワーカースレッドから Game を直接変更できないため、キューに積んでゲームスレッドで適用する。
		FString BodyStr = ParseBodyString(Request);
		if (BodyStr.IsEmpty())
		{
			OnComplete(MakeStatusJsonResponse(
				TEXT("{\"error\":\"empty body\"}"),
				EHttpServerResponseCodes::BadRequest));
			return true;
		}
		ParamsQueue.Enqueue({ MoveTemp(BodyStr), OnComplete });
		return true;  // 非同期応答
	}

	bool HandleLevelUpChoice(
		const FHttpServerRequest& Request,
		const FHttpResultCallback& OnComplete)
	{
		const FString Body = ParseBodyString(Request);
		TSharedPtr<FJsonObject> JsonObject;
		const TSharedRef<TJsonReader<>> Reader =
			TJsonReaderFactory<>::Create(Body);
		if (Body.IsEmpty()
			|| !FJsonSerializer::Deserialize(Reader, JsonObject)
			|| !JsonObject.IsValid()
			|| JsonObject->Values.Num() != 2)
		{
			OnComplete(MakeStatusJsonResponse(
				TEXT("{\"error\":\"malformed request\"}"),
				EHttpServerResponseCodes::BadRequest));
			return true;
		}

		const TSharedPtr<FJsonValue>* DecisionValue =
			JsonObject->Values.Find(TEXT("decision_id"));
		const TSharedPtr<FJsonValue>* ChoiceValue =
			JsonObject->Values.Find(TEXT("choice_id"));
		if (!DecisionValue || !ChoiceValue
			|| !DecisionValue->IsValid() || !ChoiceValue->IsValid()
			|| (*DecisionValue)->Type != EJson::String
			|| (*ChoiceValue)->Type != EJson::String)
		{
			OnComplete(MakeStatusJsonResponse(
				TEXT("{\"error\":\"decision_id and choice_id must be strings\"}"),
				EHttpServerResponseCodes::BadRequest));
			return true;
		}

		FString DecisionId = (*DecisionValue)->AsString();
		FString ChoiceId = (*ChoiceValue)->AsString();
		if (DecisionId.IsEmpty() || ChoiceId.IsEmpty())
		{
			OnComplete(MakeStatusJsonResponse(
				TEXT("{\"error\":\"decision_id and choice_id are required\"}"),
				EHttpServerResponseCodes::BadRequest));
			return true;
		}

		LevelUpChoiceQueue.Enqueue({
			MoveTemp(DecisionId), MoveTemp(ChoiceId), OnComplete});
		return true;
	}

	FHttpRouteHandle ObsSchemaRoute;
	FHttpRouteHandle ContentSchemaRoute;
	FHttpRouteHandle ParamsRoute;
	FHttpRouteHandle LevelUpChoiceRoute;
	ASurvivorsGame*  Game;  // non-owning

	friend class ASurvivorsHttpEnvService;
};

// ============================================================
// ApplyParamsToGame: /params ロジック（ゲームスレッド専用）
// 旧 HandleParams から移植。SyncConfigToLogic() を末尾で必ず呼ぶ。
// ============================================================

static FString ApplyParamsToGame(ASurvivorsGame* Game, const FString& BodyStr)
{
	if (!Game)
		return TEXT("{\"error\":\"game not set\"}");

	TSharedPtr<FJsonObject> JsonObj;
	TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(BodyStr);
	if (!FJsonSerializer::Deserialize(Reader, JsonObj) || !JsonObj.IsValid())
		return TEXT("{\"error\":\"invalid json\"}");

	// 全 mode は他 field を mutation する前に fail-closed で検証する。
	// 初心者向け: 一部だけ設定された後で mode エラーになる半端な更新を防ぎます。
	FString ItemSelectionMode;
	if (JsonObj->Values.Contains(TEXT("item_selection_mode")))
	{
		if (!JsonObj->TryGetStringField(
				TEXT("item_selection_mode"), ItemSelectionMode)
			|| (ItemSelectionMode != TEXT("auto")
				&& ItemSelectionMode != TEXT("external")))
		{
			return TEXT("{\"error\":\"unknown item_selection_mode\"}");
		}
	}
	FString WeaponPoolMode;
	if (JsonObj->Values.Contains(TEXT("weapon_pool_mode")))
	{
		static const TSet<FString> ValidPoolModes = {
			TEXT("garlic_only"), TEXT("fixed_subset"), TEXT("all_base"),
			TEXT("all_with_evolutions"), TEXT("weighted")
		};
		if (!JsonObj->TryGetStringField(TEXT("weapon_pool_mode"), WeaponPoolMode)
			|| !ValidPoolModes.Contains(WeaponPoolMode))
		{
			return TEXT("{\"error\":\"unknown weapon_pool_mode\"}");
		}
	}
	FString StartingWeaponMode;
	if (JsonObj->Values.Contains(TEXT("starting_weapon_mode")))
	{
		static const TSet<FString> ValidStartingModes = {
			TEXT("garlic"), TEXT("whip"), TEXT("random"),
			TEXT("pool_random"), TEXT("custom")
		};
		if (!JsonObj->TryGetStringField(
				TEXT("starting_weapon_mode"), StartingWeaponMode)
			|| !ValidStartingModes.Contains(StartingWeaponMode))
		{
			return TEXT("{\"error\":\"unknown starting_weapon_mode\"}");
		}
	}
	if (Game->IsLevelUpPending()
		&& !ItemSelectionMode.IsEmpty()
		&& ItemSelectionMode != Game->ItemSelectionMode)
	{
		return TEXT("{\"error\":\"cannot change item_selection_mode while pending\"}");
	}

	// 各パラメータを上書き（存在するフィールドのみ）
	int32 MinActiveEnemies;
	if (JsonObj->TryGetNumberField(TEXT("MinActiveEnemies"), MinActiveEnemies))
		Game->MinActiveEnemies = FMath::Clamp(MinActiveEnemies, 0, 600);

	int32 MaxActiveEnemies;
	if (JsonObj->TryGetNumberField(TEXT("MaxActiveEnemies"), MaxActiveEnemies))
		Game->MaxActiveEnemies = FMath::Clamp(MaxActiveEnemies, 1, 600);

	double EnemySpeedMult;
	if (JsonObj->TryGetNumberField(TEXT("EnemySpeedMult"), EnemySpeedMult))
		Game->EnemySpeedMult = FMath::Clamp(static_cast<float>(EnemySpeedMult), 0.5f, 5.f);

	double SpawnRateMult;
	if (JsonObj->TryGetNumberField(TEXT("SpawnRateMult"), SpawnRateMult))
		Game->SpawnRateMult = FMath::Clamp(static_cast<float>(SpawnRateMult), 0.1f, 5.f);

	int32 MaxEnemyTypeId;
	if (JsonObj->TryGetNumberField(TEXT("MaxEnemyTypeId"), MaxEnemyTypeId))
		Game->MaxEnemyTypeId = FMath::Clamp(MaxEnemyTypeId, 0, 10);

	double EnemyHPScale;
	if (JsonObj->TryGetNumberField(TEXT("EnemyHPScale"), EnemyHPScale))
		Game->EnemyHPScale = FMath::Clamp(static_cast<float>(EnemyHPScale), 0.1f, 10.f);

	double EnemyDamageScale;
	if (JsonObj->TryGetNumberField(TEXT("EnemyDamageScale"), EnemyDamageScale))
		Game->EnemyDamageScale = FMath::Clamp(static_cast<float>(EnemyDamageScale), 0.1f, 10.f);

	bool bTimeScalingEnabled;
	if (JsonObj->TryGetBoolField(TEXT("TimeScalingEnabled"), bTimeScalingEnabled))
		Game->bTimeScalingEnabled = bTimeScalingEnabled;

	double MaxEpisodeTime;
	if (JsonObj->TryGetNumberField(TEXT("MaxEpisodeTime"), MaxEpisodeTime))
		Game->MaxEpisodeTime = FMath::Clamp(static_cast<float>(MaxEpisodeTime), 30.f, 1800.f);

	// weapon_pool_mode
	if (!WeaponPoolMode.IsEmpty())
	{
		Game->WeaponPoolMode = WeaponPoolMode;
	}

	const TArray<TSharedPtr<FJsonValue>>* AllowedWeaponTypesArr;
	if (JsonObj->TryGetArrayField(TEXT("allowed_weapon_types"), AllowedWeaponTypesArr))
	{
		static const TSet<int32> ValidBaseWeaponIds = {
			1,   // Garlic
			2,   // Whip
			3,   // MagicWand
			4,   // Knife
			5,   // Axe
			6,   // Cross
			7,   // KingBible
			8,   // FireWand
			9,   // SantaWater
			10,  // Runetracer
			11,  // LightningRing
			12,  // Pentagram
			13,  // Peachone
			14,  // EbonyWings
			15,  // Laurel
		};
		Game->AllowedWeaponTypes.Empty();
		for (const TSharedPtr<FJsonValue>& Val : *AllowedWeaponTypesArr)
		{
			if (!Val.IsValid()) continue;
			const int32 Id = static_cast<int32>(Val->AsNumber());
			if (ValidBaseWeaponIds.Contains(Id))
				Game->AllowedWeaponTypes.Add(Id);
		}
		if (Game->WeaponPoolMode.Equals(TEXT("fixed_subset")) && Game->AllowedWeaponTypes.IsEmpty())
			Game->AllowedWeaponTypes.Add(1);
	}

	const TSharedPtr<FJsonObject>* WeaponWeightsObj;
	if (JsonObj->TryGetObjectField(TEXT("weapon_weights"), WeaponWeightsObj) && WeaponWeightsObj)
	{
		Game->AllowedWeaponTypes.Empty();
		Game->WeaponWeights.Empty();
		for (const auto& Pair : (*WeaponWeightsObj)->Values)
		{
			const int32 Id = FCString::Atoi(*Pair.Key);
			const float Weight = Pair.Value.IsValid() ? static_cast<float>(Pair.Value->AsNumber()) : 0.f;
			if (Weight > 0.f)
			{
				Game->AllowedWeaponTypes.Add(Id);
				Game->WeaponWeights.Add(Id, Weight);
			}
		}
		if (Game->WeaponPoolMode.Equals(TEXT("weighted")) && Game->AllowedWeaponTypes.IsEmpty())
		{
			Game->AllowedWeaponTypes.Add(1);
			Game->WeaponWeights.Add(1, 1.f);
		}
	}

	bool bEnablePassives;
	if (JsonObj->TryGetBoolField(TEXT("enable_passives"), bEnablePassives))
		Game->bEnablePassives = bEnablePassives;

	bool bEnableEvolutions;
	if (JsonObj->TryGetBoolField(TEXT("enable_evolutions"), bEnableEvolutions))
		Game->bEnableEvolutions = bEnableEvolutions;

	double ReplayOldPhaseFraction;
	if (JsonObj->TryGetNumberField(TEXT("replay_old_phase_fraction"), ReplayOldPhaseFraction))
		Game->ReplayOldPhaseFraction = FMath::Clamp(static_cast<float>(ReplayOldPhaseFraction), 0.f, 1.f);

	if (!StartingWeaponMode.IsEmpty())
		Game->StartingWeaponMode = StartingWeaponMode;

	if (!ItemSelectionMode.IsEmpty())
		Game->ItemSelectionMode = ItemSelectionMode;

	// RSI: initial_elapsed_time
	double InitialElapsedTime = 0.0;
	if (JsonObj->TryGetNumberField(TEXT("initial_elapsed_time"), InitialElapsedTime))
	{
		Game->InitialElapsedTime = FMath::Clamp(static_cast<float>(InitialElapsedTime), 0.f, 1800.f);
		Game->bHasInitialOverride = true;
	}

	// RSI: initial_weapon_slots
	const TArray<TSharedPtr<FJsonValue>>* WSlots;
	if (JsonObj->TryGetArrayField(TEXT("initial_weapon_slots"), WSlots))
	{
		Game->InitialWeaponSlots.Empty();
		for (const TSharedPtr<FJsonValue>& Val : *WSlots)
		{
			const TSharedPtr<FJsonObject>* SlotObj;
			if (!Val->TryGetObject(SlotObj)) continue;
			int32 WId = 0, WLv = 1;
			double TmpId = 0, TmpLv = 0;
			if ((*SlotObj)->TryGetNumberField(TEXT("weapon_id"), TmpId)) WId = static_cast<int32>(TmpId);
			if ((*SlotObj)->TryGetNumberField(TEXT("level"),     TmpLv)) WLv = static_cast<int32>(TmpLv);
			Game->InitialWeaponSlots.Add({WId, FMath::Clamp(WLv, 1, 8)});
		}
		if (!Game->InitialWeaponSlots.IsEmpty())
			Game->bHasInitialOverride = true;
	}

	// RSI: initial_passive_slots
	const TArray<TSharedPtr<FJsonValue>>* PSlots;
	if (JsonObj->TryGetArrayField(TEXT("initial_passive_slots"), PSlots))
	{
		Game->InitialPassiveSlots.Empty();
		for (const TSharedPtr<FJsonValue>& Val : *PSlots)
		{
			const TSharedPtr<FJsonObject>* SlotObj;
			if (!Val->TryGetObject(SlotObj)) continue;
			int32 PId = 0, PLv = 1;
			double TmpId = 0, TmpLv = 0;
			if ((*SlotObj)->TryGetNumberField(TEXT("passive_id"), TmpId)) PId = static_cast<int32>(TmpId);
			if ((*SlotObj)->TryGetNumberField(TEXT("level"),      TmpLv)) PLv = static_cast<int32>(TmpLv);

			if (PId <= 0 || PId >= SurvivorsGameConstants::MaxPassiveTypeCountReserved)
				continue;

			const EPassiveItemType PType  = static_cast<EPassiveItemType>(PId);
			const int32            MaxLv  = Game->GetPassiveItemMaxLevel(PType);
			if (MaxLv <= 0) continue;

			Game->InitialPassiveSlots.Add({PId, FMath::Clamp(PLv, 1, MaxLv)});
		}
		if (!Game->InitialPassiveSlots.IsEmpty())
			Game->bHasInitialOverride = true;
	}

	// RSI: clear_initial_override
	bool bClearOverride = false;
	if (JsonObj->TryGetBoolField(TEXT("clear_initial_override"), bClearOverride) && bClearOverride)
	{
		Game->bHasInitialOverride = false;
		Game->InitialWeaponSlots.Empty();
		Game->InitialPassiveSlots.Empty();
		Game->InitialElapsedTime = 0.f;
	}

	// UPROPERTY 更新後に Logic に同期（必須）
	Game->SyncConfigToLogic();

	return TEXT("{\"status\":\"ok\"}");
}

// ============================================================
// ASurvivorsHttpEnvService: Actor 実装
// ============================================================

ASurvivorsHttpEnvService::ASurvivorsHttpEnvService()
{
	PrimaryActorTick.bCanEverTick = true;
}

void ASurvivorsHttpEnvService::BeginPlay()
{
	Super::BeginPlay();

	if (!SurvivorsGame)
	{
		SurvivorsGame = Cast<ASurvivorsGame>(
			UGameplayStatics::GetActorOfClass(GetWorld(), ASurvivorsGame::StaticClass()));
	}

	if (!SurvivorsGame)
	{
		UE_LOG(LogTemp, Error,
			TEXT("ASurvivorsHttpEnvService: ASurvivorsGame が見つかりません。レベルに配置してください。"));
		return;
	}

	auto* Server = new FSurvivorsEnvServer(SurvivorsGame.Get());

	Server->BuildObsSchemaCache();

	EnvServer = TUniquePtr<FHttpEnvServerBase>(Server);
	EnvServer->StartServer(static_cast<uint32>(ServerPort));
}

void ASurvivorsHttpEnvService::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	if (EnvServer)
	{
		EnvServer->StopServer();
		EnvServer.Reset();
	}
	Super::EndPlay(EndPlayReason);
}

void ASurvivorsHttpEnvService::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	if (bManagedExternally) return;

	if (!EnvServer) return;

	ProcessLevelUpChoiceRequests();

	{
		FString Json;
		FHttpResultCallback Cb;
		while (TakeParamsRequest(Json, Cb))
		{
			FString ResponseJson = ApplyParams(Json);
			CompleteParams(ResponseJson, MoveTemp(Cb));
		}
	}

	EnvServer->Tick();
}



bool ASurvivorsHttpEnvService::TakeStepRequest(
	TArray<float>& OutAction, int32& OutSteps, FHttpResultCallback& OutCallback)
{
	if (!EnvServer) return false;
	return EnvServer->TakeStepRequest(OutAction, OutSteps, OutCallback);
}

bool ASurvivorsHttpEnvService::TakeResetRequest(
	TOptional<int32>& OutSeed, FHttpResultCallback& OutCallback)
{
	if (!EnvServer) return false;
	return EnvServer->TakeResetRequest(OutSeed, OutCallback);
}

bool ASurvivorsHttpEnvService::TakeParamsRequest(FString& OutJson, FHttpResultCallback& OutCallback)
{
	if (!EnvServer) return false;
	auto* Server = static_cast<FSurvivorsEnvServer*>(EnvServer.Get());
	return Server ? Server->TakeParamsRequest(OutJson, OutCallback) : false;
}

void ASurvivorsHttpEnvService::CompleteStep(FEnvStepResult Result, FHttpResultCallback Callback)
{
	if (EnvServer) EnvServer->CompleteStep(MoveTemp(Result), MoveTemp(Callback));
}

void ASurvivorsHttpEnvService::CompleteReset(FEnvResetResult Result, FHttpResultCallback Callback)
{
	if (EnvServer) EnvServer->CompleteReset(MoveTemp(Result), MoveTemp(Callback));
}

FString ASurvivorsHttpEnvService::ApplyParams(const FString& Json)
{
	return ApplyParamsToGame(SurvivorsGame.Get(), Json);
}

void ASurvivorsHttpEnvService::CompleteParams(
	const FString& ResponseJson, FHttpResultCallback Callback)
{
	if (ResponseJson.StartsWith(TEXT("{\"error\"")))
	{
		Callback(MakeStatusJsonResponse(
			ResponseJson,
			EHttpServerResponseCodes::BadRequest));
	}
	else
	{
		Callback(FHttpEnvServerBase::MakeJsonResponse(ResponseJson));
	}
}

FSurvivorsGameLogic* ASurvivorsHttpEnvService::GetGameLogic()
{
	return SurvivorsGame ? SurvivorsGame->GetLogic() : nullptr;
}

void ASurvivorsHttpEnvService::ProcessLevelUpChoiceRequests()
{
	if (!EnvServer) return;
	auto* Server = static_cast<FSurvivorsEnvServer*>(EnvServer.Get());
	if (!Server) return;

	FString DecisionId;
	FString ChoiceId;
	FHttpResultCallback Callback;
	while (Server->TakeLevelUpChoiceRequest(DecisionId, ChoiceId, Callback))
	{
		if (!SurvivorsGame)
		{
			Callback(MakeStatusJsonResponse(
				TEXT("{\"error\":\"game not set\"}"),
				EHttpServerResponseCodes::Conflict));
			continue;
		}

		const FSurvivorsLevelUpApplyResult Result =
			SurvivorsGame->ApplyExternalLevelUpChoice(DecisionId, ChoiceId);
		if (Result.Status != ESurvivorsLevelUpApplyStatus::Applied)
		{
			const FString Error =
				Result.Status == ESurvivorsLevelUpApplyStatus::InvalidChoice
					? TEXT("invalid choice")
					: TEXT("stale decision");
			TSharedRef<FJsonObject> ErrorObject = MakeShared<FJsonObject>();
			ErrorObject->SetStringField(TEXT("error"), Error);
			ErrorObject->SetStringField(TEXT("decision_id"), DecisionId);
			ErrorObject->SetStringField(TEXT("choice_id"), ChoiceId);
			Callback(MakeStatusJsonResponse(
				SerializeJsonObject(ErrorObject),
				EHttpServerResponseCodes::Conflict));
			continue;
		}

		Callback(FHttpEnvServerBase::MakeJsonResponse(
			BuildLevelUpApplyResponseJson(
				Result, SurvivorsGame->GetObsSchemaHash())));
	}
}

FString ASurvivorsHttpEnvService::BuildInfoJson() const
{
	if (!SurvivorsGame)
	{
		return TEXT(
			"{\"spawn_debug\":{},\"level_up_pending\":false,"
			"\"level_up_decision_id\":\"\",\"level_up_player_level\":0,"
			"\"level_up_backlog\":0,\"level_up_choices\":[]}");
	}
	const FString SpawnDebug = SurvivorsGame->GetSpawnDebugJson();
	return SerializeJsonObject(BuildLevelUpInfoObject(
		SurvivorsGame->GetPendingLevelUpDecision(),
		SurvivorsGame->GetLevelUpBacklog(),
		&SpawnDebug));
}
