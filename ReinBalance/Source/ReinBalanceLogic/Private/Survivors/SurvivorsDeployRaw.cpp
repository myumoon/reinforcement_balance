/**
 * deploy raw state のカメラ範囲判定と JSON 文字列化を実装する。
 *
 * HTTP service と LLT（fixture 書き出し）が同じ関数を使うので、応答の形はここだけで決まる。
 */
#include "Survivors/SurvivorsDeployRaw.h"

namespace
{
	/**
	 * float を JSON の数値文字列にする（%.9g）。
	 *
	 * 値は float 精度へ丸めてから書く。9 桁あれば float を往復しても値が変わらない。
	 * 非有限値は JSON として不正になり、受け手が拒否する。
	 */
	FString Num(double Value)
	{
		return FString::Printf(TEXT("%.9g"), static_cast<double>(static_cast<float>(Value)));
	}

	/**
	 * スロット配列を JSON 配列文字列にする。
	 *
	 * 各要素は {"index","type_id","level"} の 3 キー。
	 */
	FString SlotsToJson(const TArray<FSurvivorsDeployRawSlot>& Slots)
	{
		TArray<FString> Parts;
		for (const FSurvivorsDeployRawSlot& S : Slots)
		{
			Parts.Add(FString::Printf(
				TEXT("{\"index\":%d,\"type_id\":%d,\"level\":%d}"), S.Index, S.TypeId, S.Level));
		}
		return TEXT("[") + FString::Join(Parts, TEXT(",")) + TEXT("]");
	}

	/**
	 * entity 1 件を JSON オブジェクト文字列にする。
	 *
	 * 全 entity が同じ 8 キーを持ち、slot / ttl_true_s は武器エフェクト以外で null になる。
	 */
	FString EntityToJson(const FSurvivorsDeployRawEntity& E)
	{
		return FString::Printf(
			TEXT("{\"entity_id\":%lld,\"class_name\":\"%s\",\"world_x\":%s,\"world_y\":%s,\"radius_world\":%s,"
			     "\"slot\":%s,\"ttl_true_s\":%s,\"warning\":%s}"),
			static_cast<long long>(E.EntityId), *E.ClassName,
			*Num(E.WorldPos.X), *Num(E.WorldPos.Y), *Num(E.RadiusWorld),
			E.Slot.IsSet() ? *FString::FromInt(E.Slot.GetValue()) : TEXT("null"),
			E.TtlTrueS.IsSet() ? *Num(E.TtlTrueS.GetValue()) : TEXT("null"),
			E.bWarning ? TEXT("true") : TEXT("false"));
	}
}

bool SurvivorsDeployRaw::IsWithinCullRange(
	FVector2D Center, float HalfWidth, float HalfHeight, float MarginU, FVector2D Pos)
{
	const FVector2D D = Pos - Center;
	return FMath::Abs(D.X) <= HalfWidth + MarginU && FMath::Abs(D.Y) <= HalfHeight + MarginU;
}

FString SurvivorsDeployRaw::ToJson(const FSurvivorsDeployRawState& State)
{
	TArray<FString> Entities;
	Entities.Reserve(State.Entities.Num());
	for (const FSurvivorsDeployRawEntity& E : State.Entities)
	{
		Entities.Add(EntityToJson(E));
	}
	return FString::Printf(
		TEXT("{\"schema_version\":\"%s\",\"elapsed_s\":%s,"
		     "\"camera\":{\"center_x\":%s,\"center_y\":%s,\"half_width\":%s,\"half_height\":%s,\"cull_margin\":%s},"
		     "\"player\":{\"world_x\":%s,\"world_y\":%s,\"hp_ratio\":%s,\"level\":%d},"
		     "\"duration_mult\":%s,\"weapon_slots\":%s,\"passive_slots\":%s,\"entities\":[%s]}"),
		SchemaVersion, *Num(State.ElapsedS),
		*Num(State.CameraCenter.X), *Num(State.CameraCenter.Y),
		*Num(State.CameraHalfWidth), *Num(State.CameraHalfHeight), *Num(State.CullMarginU),
		*Num(State.PlayerPos.X), *Num(State.PlayerPos.Y), *Num(State.HpRatio), State.PlayerLevel,
		*Num(State.DurationMult), *SlotsToJson(State.WeaponSlots), *SlotsToJson(State.PassiveSlots),
		*FString::Join(Entities, TEXT(",")));
}
