#include "Training/SurvivorsHttpEnvService.h"
#include "HttpEnvServerBase.h"
#include "Kismet/GameplayStatics.h"
#include "Dom/JsonObject.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"

class ASurvivorsHttpEnvService::FSurvivorsEnvServer : public FHttpEnvServerBase
{
public:
	explicit FSurvivorsEnvServer(ASurvivorsGame* InGame) : Game(InGame) {}

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
			Result.Obs    = Game->GetObservation();
			Result.Reward = AccumulatedReward;
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
	bool HandleObsSchema(const FHttpServerRequest& Request, const FHttpResultCallback& OnComplete)
	{
		if (!Game)
		{
			OnComplete(MakeJsonResponse(TEXT("{\"error\":\"game not set\"}")));
			return true;
		}

		TArray<FSurvivorsObsSegment> Schema = Game->GetObsSchema();
		FString SegmentsStr;
		for (int32 i = 0; i < Schema.Num(); ++i)
		{
			SegmentsStr += FString::Printf(TEXT("{\"name\":\"%s\",\"dim\":%d}"),
				*Schema[i].Name, Schema[i].Dim);
			if (i < Schema.Num() - 1) SegmentsStr += TEXT(",");
		}
		FString Json = FString::Printf(
			TEXT("{\"segments\":[%s],\"total_dim\":%d,\"obs_schema_hash\":\"%s\"}"),
			*SegmentsStr, Game->GetObsDim(), *Game->GetObsSchemaHash());

		OnComplete(MakeJsonResponse(Json));
		return true;
	}

	bool HandleParams(const FHttpServerRequest& Request, const FHttpResultCallback& OnComplete)
	{
		if (!Game)
		{
			OnComplete(MakeJsonResponse(TEXT("{\"error\":\"game not set\"}")));
			return true;
		}

		FString BodyStr = ParseBodyString(Request);

		TSharedPtr<FJsonObject> JsonObj;
		TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(BodyStr);
		if (!FJsonSerializer::Deserialize(Reader, JsonObj) || !JsonObj.IsValid())
		{
			OnComplete(MakeJsonResponse(TEXT("{\"error\":\"invalid json\"}")));
			return true;
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

		// PR2 拡張パラメータ
		// weapon_pool_mode: 受け付ける値は以下の5値。未知値は "garlic_only" にフォールバック。
		//   "garlic_only"         → Garlic のみをレベルアップ候補に出す
		//   "fixed_subset"        → allowed_weapon_types で指定した武器のみ
		//   "all_base"            → 全基本武器（Garlic〜Laurel）
		//   "all_with_evolutions" → all_base と同じ扱い（進化後武器は進化システムで処理）
		//   "weighted"            → weights=0 の武器を除外した固定サブセット扱い
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
			// 基本武器 ID の有効範囲（進化後武器 ID=16〜 はデフォルトで除外）
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
				// 無効 ID は無視
			}

			// fixed_subset モードで空になった場合は Garlic にフォールバック
			if (Game->WeaponPoolMode.Equals(TEXT("fixed_subset")) && Game->AllowedWeaponTypes.IsEmpty())
				Game->AllowedWeaponTypes.Add(1);  // Garlic fallback
		}

		// weapon_weights: weighted モード時に重み0以下の武器を AllowedWeaponTypes から除外する
		// Python 側が weights={id: weight, ...} を送る; 重み0の武器は除外する
		const TSharedPtr<FJsonObject>* WeaponWeightsObj;
		if (JsonObj->TryGetObjectField(TEXT("weapon_weights"), WeaponWeightsObj) && WeaponWeightsObj)
		{
			// 有効な武器 ID のうち重み > 0 のもののみ AllowedWeaponTypes に残す
			// また WeaponWeights マップに重みを保存する（加重サンプリング用）
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

			// weighted モードで空になった場合は Garlic にフォールバック
			if (Game->WeaponPoolMode.Equals(TEXT("weighted")) && Game->AllowedWeaponTypes.IsEmpty())
			{
				Game->AllowedWeaponTypes.Add(1);  // Garlic fallback
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

		// RSI: initial_weapon_slots [{weapon_id: int, level: int}, ...]
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

		// RSI: initial_passive_slots [{passive_id: int, level: int}, ...]
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
				Game->InitialPassiveSlots.Add({PId, FMath::Clamp(PLv, 1, 9)});
			}
			if (!Game->InitialPassiveSlots.IsEmpty())
				Game->bHasInitialOverride = true;
		}

		// RSI: clear_initial_override — true を送ると次のリセットでオーバーライドを適用しない
		bool bClearOverride = false;
		if (JsonObj->TryGetBoolField(TEXT("clear_initial_override"), bClearOverride) && bClearOverride)
		{
			Game->bHasInitialOverride = false;
			Game->InitialWeaponSlots.Empty();
			Game->InitialPassiveSlots.Empty();
			Game->InitialElapsedTime = 0.f;
		}

		OnComplete(MakeJsonResponse(TEXT("{\"status\":\"ok\"}")));
		return true;
	}

	FHttpRouteHandle ObsSchemaRoute;
	FHttpRouteHandle ParamsRoute;
	ASurvivorsGame*  Game; // non-owning
};

// ---- Actor ------------------------------------------------------------------

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

	EnvServer = TUniquePtr<FHttpEnvServerBase>(new FSurvivorsEnvServer(SurvivorsGame.Get()));
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
	if (EnvServer) EnvServer->Tick();
}
