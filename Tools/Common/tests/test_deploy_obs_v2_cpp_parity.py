"""DeployObs v2 特徴量パラメータ YAML が C++ の定数・enum・武器実装と一致することを検証する。

runtime は C++ を読まず YAML の表だけを使うため、表が C++ からずれたらこのテストで止めます。
C++ ソースはリポジトリ内のヘッダ・実装を正規表現で読みます（無ければ skip せず失敗させます）。
"""

from __future__ import annotations

import re
from pathlib import Path

from reinbalance_survivors_contracts.deploy_obs_v2_features import load_deploy_obs_v2_feature_params

ROOT = Path(__file__).resolve().parents[3]
LOGIC = ROOT / "ReinBalance/Source/ReinBalanceLogic"
PUBLIC = LOGIC / "Public/Survivors"
PRIVATE = LOGIC / "Private/Survivors"


def _read(path: Path) -> str:
    """C++ ソースを読む。

    無い環境では C++ との一致を検証できないので、skip ではなく失敗させます。
    """
    assert path.is_file(), f"C++ source missing: {path}"
    return path.read_text(encoding="utf-8-sig")


def _enum_names(source: str, enum_name: str) -> list[str]:
    """enum class 本体の ``Name = N,`` を値の順に並べる。

    値が 0 から連番でないと語彙の index と enum 値がずれるので、そのことも確かめます。
    """
    match = re.search(rf"enum\s+class\s+{enum_name}\b[^{{]*{{(?P<body>.*?)}};", source, re.DOTALL)
    assert match is not None, f"{enum_name} missing"
    pairs = re.findall(r"^\s*([A-Za-z_]\w*)\s*=\s*(\d+)\s*,?", match.group("body"), re.MULTILINE)
    assert [int(value) for _, value in pairs] == list(range(len(pairs))), f"{enum_name} ids not contiguous"
    return [name for name, _ in pairs]


def _constant(source: str, name: str) -> float:
    """``static constexpr <型> Name = 値;`` の数値を読む。

    最初に見つかった定義の値を返し、定数が見つからなければ比較できないので失敗させます。
    """
    match = re.search(rf"\b{name}\s*=\s*([0-9.]+)f?\s*;", source)
    assert match is not None, f"{name} missing"
    return float(match.group(1))


def _table_column(source: str, table: str, column: int) -> list[float]:
    """``inline constexpr F...Params Table[MaxWeaponLevel] = {{...}, ...};`` の指定列を行順に読む。

    レベル別の表（1行が1レベル）から、持続時間など1つの列だけを取り出して list にします。
    """
    match = re.search(rf"\b{table}\s*\[\s*MaxWeaponLevel\s*\]\s*=\s*{{(?P<body>.*?)\n\s*}};", source, re.DOTALL)
    assert match is not None, f"{table} missing"
    rows = re.findall(r"{([^{}]*)}", match.group("body"))
    return [float(row.split(",")[column].strip().rstrip("f")) for row in rows]


def test_weapon_and_passive_vocabularies_match_cpp_enums():
    """武器・パッシブ語彙が EWeaponType / EPassiveItemType の値順と一致し、末尾が unknown。

    sim の enum に武器が足されたのに yaml の語彙を直し忘れると、種類 id がずれるのでここで検出します。
    """
    params = load_deploy_obs_v2_feature_params()
    types = _read(PUBLIC / "SurvivorsTypes.h")
    assert list(params["weapon_vocabulary"]) == _enum_names(types, "EWeaponType") + ["unknown"]
    assert list(params["passive_vocabulary"]) == _enum_names(types, "EPassiveItemType") + ["unknown"]


def test_limits_and_direction_bins_match_cpp_constants():
    """スロット数・最大レベル・ttl 上限・方向ビン数が C++ 定数と一致する。

    スロット番号や残り時間の正規化に使う分母が sim と同じであることを確かめます。
    """
    params = load_deploy_obs_v2_feature_params()
    constants = _read(PUBLIC / "SurvivorsGameConstants.h")
    wiki = _read(PUBLIC / "SurvivorsWikiSpec.h")
    assert params["max_weapon_slots"] == _constant(constants, "MaxWeaponSlots")
    assert params["max_passive_slots"] == _constant(constants, "MaxPassiveSlots")
    assert params["max_projectile_obs_ttl_s"] == _constant(constants, "MaxProjectileObsTtl")
    assert re.search(r"\bMaxWeaponLevel\s*=\s*SurvivorsWikiSpec::BaseWeaponMaxLevel\s*;", constants)
    assert params["max_weapon_level"] == _constant(wiki, "BaseWeaponMaxLevel")
    passive_levels = re.search(r"PassiveMaxLevel\[PassiveTypeCount\]\s*=\s*{([^}]*)}", wiki)
    assert passive_levels is not None
    assert params["max_passive_level"] == max(int(v) for v in passive_levels.group(1).split(","))
    assert params["direction_bins"] == _constant(constants, "EnemyDensityDirCount") == _constant(constants, "GemDensityDirCount")


def test_direction_bin_formula_is_the_cpp_build_dir_density_formula():
    """C++ BuildDirDensity が Common と同じ方向ビン式・距離除外を使っている。

    C++ 側の式の文字列を確かめ、角度から16分割する規則と距離0の除外が
    Common の実装と食い違っていないことを保証します。
    """
    source = _read(PRIVATE / "SurvivorsGameLogic.cpp")
    assert "if (D <= KINDA_SMALL_NUMBER) continue;" in source
    assert "const float A01 = (FMath::Atan2(Rel.Y, Rel.X) + PI) / (2.f * PI);" in source
    assert "const int32 Dir = FMath::Clamp(FMath::FloorToInt(A01 * DC), 0, DC - 1);" in source


def test_effect_duration_tables_match_cpp_constants_and_formulas():
    """zone / orbit の持続時間表と式（固定分・倍率の掛かり方）が C++ と一致する。

    残り時間の推定に使う持続時間が sim と同じになるよう、レベル別の値と
    持続時間倍率の掛かる/掛からない部分を C++ の定数と実装から確かめます。
    """
    durations = load_deploy_obs_v2_feature_params()["effect_durations"]
    constants = _read(PUBLIC / "SurvivorsGameConstants.h")
    for weapon, table, column in (("SantaWater", "SantaWaterTable", 3), ("LaBorra", "LaBorraTable", 3), ("KingBible", "KingBibleTable", 2), ("UnholyVespers", "UnholyVespersTable", 2)):
        assert list(durations[weapon]["scaled_by_level_s"]) == _table_column(constants, table, column), weapon
    warning = _constant(constants, "SantaWaterWarningTime")
    strike = _constant(constants, "LightningRingStrikeLifeTime")
    assert durations["SantaWater"]["fixed_s"] == durations["LaBorra"]["fixed_s"] == warning
    assert durations["KingBible"]["fixed_s"] == durations["UnholyVespers"]["fixed_s"] == 0.0
    for weapon in ("LightningRing", "ThunderLoop"):
        assert durations[weapon]["fixed_s"] == strike and set(durations[weapon]["scaled_by_level_s"]) == {0.0}
    for weapon in ("FireWand", "Hellfire"):
        assert durations[weapon]["fixed_s"] == 0.0 and set(durations[weapon]["scaled_by_level_s"]) == {0.2}
    weapons = PRIVATE / "Weapons/Projectile"
    santa = _read(weapons / "SurvivorsWeaponSantaWaterLogic.cpp")
    assert "BurstDuration = CachedDuration * PE.DurationMult;" in santa
    assert "Z.LifeTime      = SurvivorsGameConstants::SantaWaterWarningTime + BurstDuration;" in santa
    assert _read(weapons / "SurvivorsWeaponFireWandLogic.cpp").count("const float EffDuration        = 0.2f * PE.DurationMult;") == 2
    assert "Marker.LifeTime      = SurvivorsGameConstants::LightningRingStrikeLifeTime;" in _read(weapons / "SurvivorsWeaponLightningRingLogic.cpp")
    assert "ActiveTimer = CachedDuration * PE.DurationMult;" in _read(weapons / "SurvivorsWeaponKingBibleLogic.cpp")


def _cpp_weapon_effect_kinds() -> dict[str, set[str]]:
    """C++ の武器実装から 武器→エフェクト種類 を組み立てる。

    CreateWeaponLogic の case→実装クラス、実装の SpawnProjectile / SpawnGroundZone 呼び出し、
    基底クラスの継承、GetProjectileObsView の orbit / aura 対象武器を合わせて求めます。
    """
    game = _read(PRIVATE / "SurvivorsGameLogic.cpp")
    factory = re.search(r"CreateWeaponLogic\(EWeaponType Type\)\s*{(?P<body>.*?)default:", game, re.DOTALL)
    assert factory is not None
    weapon_class: dict[str, str] = {}
    pending: list[str] = []
    for line in factory.group("body").splitlines():
        if case := re.search(r"case EWeaponType::(\w+):", line):
            pending.append(case.group(1))
        elif made := re.search(r"MakeUnique<FSurvivorsWeapon(\w+)Logic>", line):
            weapon_class.update({name: made.group(1) for name in pending})
            pending = []

    def class_kinds(name: str) -> set[str]:
        """実装クラス（と基底クラス）が呼ぶ spawn から種類を求める。

        進化形の武器は基底クラスの spawn を使うことがあるので、基底クラスもたどります。
        """
        source = _read(next((PRIVATE / "Weapons").rglob(f"SurvivorsWeapon{name}Logic.cpp")))
        kinds = {"projectile"} if "SpawnProjectile" in source else set()
        kinds |= {"zone"} if "SpawnGroundZone" in source else set()
        header = _read(next((PUBLIC / "Weapons").rglob(f"SurvivorsWeapon{name}Logic.h")))
        base = re.search(rf"class \w+ FSurvivorsWeapon{name}Logic\s*:\s*public FSurvivorsWeapon(\w*)Logic", header)
        assert base is not None, name
        return kinds | (class_kinds(base.group(1)) if base.group(1) else set())

    result = {weapon: class_kinds(name) for weapon, name in weapon_class.items()}
    view = re.search(r"GetProjectileObsView\(\) const\s*{(?P<body>.*?)\n}", game, re.DOTALL)
    assert view is not None
    orbit = re.findall(r"OrbWType != EWeaponType::(\w+)", view.group("body"))
    aura = re.findall(r"Slot\.Type != EWeaponType::(\w+)", view.group("body"))
    assert sorted(orbit) == ["KingBible", "UnholyVespers"] and sorted(aura) == ["Garlic", "SoulEater"]
    for weapon in orbit:
        result[weapon].add("orbit")
    for weapon in aura:
        result[weapon].add("aura")
    return {weapon: kinds for weapon, kinds in result.items() if kinds}


def test_weapon_effect_kinds_match_cpp_projectile_obs_view():
    """武器→エフェクト種類の集合が C++ の spawn 呼び出しと GetProjectileObsView から求めたものと一致する。

    スロット割り当ては「同じ種類を出す武器が1つだけか」で決まるので、
    この対応表が sim とずれるとスロットの有効・無効が変わってしまいます。
    """
    table = {weapon: set(kinds) for weapon, kinds in load_deploy_obs_v2_feature_params()["weapon_effect_kinds"].items()}
    expected = _cpp_weapon_effect_kinds()
    assert table == expected
    assert table["FireWand"] == table["Hellfire"] == {"projectile", "zone"}
    assert table["LightningRing"] == table["ThunderLoop"] == {"zone"}
