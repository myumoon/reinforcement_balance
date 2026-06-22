#include "Training/SurvivorsHttpEnvService.h"
#include "HttpEnvServerBase.h"
#include "Kismet/GameplayStatics.h"
#include "Dom/JsonObject.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"

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
	}

	// ---- ParamsQueue 外部制御 API ----

	struct FParamsRequest
	{
		FString             JsonBody;
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
			Result.InfoJson = FString::Printf(TEXT("{\"spawn_debug\":%s}"), *Game->GetSpawnDebugJson());
		}
		return Result;
	}

protected:
	virtual void RegisterAdditionalRoutes(TSharedPtr<IHttpRouter> Router) override
	{
		ObsSchemaRoute = Router->BindRoute(
			FHttpPath(TEXT("/obs_schema")), EHttpServerRequestVerbs::VERB_GET,
			FHttpRequestHandler::CreateRaw(this, &FSurvivorsEnvServer::HandleObsSchema));

		ParamsRoute = Router->BindRoute(
			FHttpPath(TEXT("/params")), EHttpServerRequestVerbs::VERB_POST,
			FHttpRequestHandler::CreateRaw(this, &FSurvivorsEnvServer::HandleParams));
	}

	virtual void UnregisterAdditionalRoutes(TSharedPtr<IHttpRouter> Router) override
	{
		if (Router)
		{
			Router->UnbindRoute(ObsSchemaRoute);
			Router->UnbindRoute(ParamsRoute);
		}
	}

private:
	// キャッシュ済み obs_schema JSON（HTTPワーカースレッドから Game に触れないよう事前構築）
	FString CachedObsSchemaJson;

	// /params は Survivors 固有のためここで管理（FHttpEnvServerBase には追加しない）
	TQueue<FParamsRequest, EQueueMode::Mpsc> ParamsQueue;

	bool HandleObsSchema(const FHttpServerRequest& Request, const FHttpResultCallback& OnComplete)
	{
		// Phase 2: キャッシュを返すだけ（HTTP ワーカースレッドから Game に触れない）
		OnComplete(MakeJsonResponse(CachedObsSchemaJson));
		return true;
	}

	bool HandleParams(const FHttpServerRequest& Request, const FHttpResultCallback& OnComplete)
	{
		// Phase 2: ワーカースレッドでは Game フィールドを直接変更せずキューに積む。
		// ゲームスレッドの Tick または ParallelSetupActor の Tick で ApplyParams を呼ぶ。
		FString BodyStr = ParseBodyString(Request);
		if (BodyStr.IsEmpty())
		{
			OnComplete(MakeJsonResponse(TEXT("{\"error\":\"empty body\"}")));
			return true;
		}
		ParamsQueue.Enqueue({ MoveTemp(BodyStr), OnComplete });
		return true;  // 非同期応答
	}

	FHttpRouteHandle ObsSchemaRoute;
	FHttpRouteHandle ParamsRoute;
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
	FString WeaponPoolMode;
	if (JsonObj->TryGetStringField(TEXT("weapon_pool_mode"), WeaponPoolMode))
	{
		static const TSet<FString> ValidPoolModes = {
			TEXT("garlic_only"), TEXT("fixed_subset"), TEXT("all_base"),
			TEXT("all_with_evolutions"), TEXT("weighted")
		};
		if (ValidPoolModes.Contains(WeaponPoolMode))
			Game->WeaponPoolMode = WeaponPoolMode;
		else
		{
			UE_LOG(LogTemp, Warning,
				TEXT("SurvivorsHttpEnvService: 未知の weapon_pool_mode \"%s\" -> \"garlic_only\" にフォールバック"),
				*WeaponPoolMode);
			Game->WeaponPoolMode = TEXT("garlic_only");
		}
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

	FString StartingWeaponMode;
	if (JsonObj->TryGetStringField(TEXT("starting_weapon_mode"), StartingWeaponMode))
		Game->StartingWeaponMode = StartingWeaponMode;

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

	// Phase 2: obs_schema キャッシュを StartServer 前に構築する
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

	// Phase 2: bManagedExternally=true の場合は ParallelSetupActor が制御するためスキップ
	if (bManagedExternally) return;

	if (!EnvServer) return;

	// Phase 2: Params を先にゲームスレッドで適用してから Step/Reset を処理する
	{
		FString Json;
		FHttpResultCallback Cb;
		while (TakeParamsRequest(Json, Cb))
		{
			FString ResponseJson = ApplyParams(Json);
			Cb(FHttpEnvServerBase::MakeJsonResponse(ResponseJson));
		}
	}

	EnvServer->Tick();
}

// ---- Phase 2: 外部制御 API 実装 ----

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

FSurvivorsGameLogic* ASurvivorsHttpEnvService::GetGameLogic()
{
	return SurvivorsGame ? SurvivorsGame->GetLogic() : nullptr;
}
