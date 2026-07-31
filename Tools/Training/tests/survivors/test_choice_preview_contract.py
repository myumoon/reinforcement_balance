"""反実仮想 level-up preview の静的・wire 契約を検証する。

初心者向け: Python が観測値を手計算せず、C++ Logic の複製結果を厳密に受け取ることを確認する。
"""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]


def _read(relative: str) -> str:
    """repository 内の契約対象を UTF-8 で読む。

    初心者向け: テストごとのpath組み立てを共通化し、別ファイルを誤検査する事故を防ぐ。
    """

    return (ROOT / relative).read_text(encoding="utf-8")


def _class_functions(relative: str, class_name: str) -> set[str]:
    """Python class が公開する関数名を AST から取得する。

    初心者向け: コメント中の文字列ではなく、実際に定義されたmethodだけを数える。
    """

    tree = ast.parse(_read(relative))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                child.name
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    raise AssertionError(f"class not found: {class_name}")


def test_logic_preview_is_const_and_uses_production_transition() -> None:
    """preview が const かつproduction apply/observation経路を再利用することを検証する。

    初心者向け: 本番と別の計算式を作ると将来ずれるため、同じLogic methodの呼出しを要求する。
    """

    header = _read(
        "ReinBalance/Source/ReinBalanceLogic/Public/Survivors/SurvivorsGameLogic.h"
    )
    source = _read(
        "ReinBalance/Source/ReinBalanceLogic/Private/Survivors/SurvivorsGameLogic.cpp"
    )

    assert "FSurvivorsChoicePreview PreviewLevelUpChoice(" in header
    declaration = header.split("FSurvivorsChoicePreview PreviewLevelUpChoice(", 1)[1]
    assert ") const;" in declaration.split("// ---- ParallelFor", 1)[0]
    body = source.split(
        "FSurvivorsChoicePreview FSurvivorsGameLogic::PreviewLevelUpChoice", 1
    )[1].split("FPassiveEffects FSurvivorsGameLogic::ComputePassiveEffects", 1)[0]
    assert "CloneForPreview()" in body
    assert "ApplyExternalLevelUpChoice(" in body
    assert "Sandbox->GetObservation()" in body
    assert "Sandbox->GetObsSchema()" in body


def test_preview_clone_covers_runtime_state_and_rejects_uncloneable_weapons() -> None:
    """sandbox が観測・乱数・pending・武器runtimeをdeep cloneすることを検証する。

    初心者向け: pointerをそのまま共有するとpreviewが本体を変更するため、全武器にclone契約を課す。
    """

    logic = _read(
        "ReinBalance/Source/ReinBalanceLogic/Private/Survivors/SurvivorsGameLogic.cpp"
    )
    weapon_headers = list(
        (
            ROOT
            / "ReinBalance/Source/ReinBalanceLogic/Public/Survivors/Weapons/Projectile"
        ).glob("*.h")
    )
    clone_body = logic.split(
        "TUniquePtr<FSurvivorsGameLogic> FSurvivorsGameLogic::CloneForPreview", 1
    )[1].split(
        "FSurvivorsChoicePreview FSurvivorsGameLogic::PreviewLevelUpChoice", 1
    )[0]

    for token in (
        "WeaponSlots",
        "PassiveSlots",
        "CachedPassiveEffects",
        "RandStream",
        "Projectiles",
        "GroundZones",
        "LevelUpDecisionState",
        "LastAppliedLevelUpResult",
        "CloneForPreview",
        "BuildEnemyGrid",
        "BuildPickupGrid",
    ):
        assert token in clone_body
    assert "return nullptr" in clone_body
    for path in weapon_headers:
        assert "CloneForPreview" in path.read_text(encoding="utf-8"), path.name


def test_component_observation_is_only_a_logic_delegation() -> None:
    """legacy ObservationComponent がcanonical Logicへ委譲するだけか検証する。

    初心者向け: Component側に観測式を残さず、訓練と画面表示の値を同じsourceに揃える。
    """

    source = _read(
        "ReinBalance/Source/ReinBalance/Private/Survivors/Game/"
        "SurvivorsObservationComponent.cpp"
    )
    assert "Game->GetObservation()" in source
    assert "Game->GetObsSchema()" in source
    assert "Game->GetObsSchemaHash()" in source
    assert "BuildDirectionalDensityFeatures" not in source
    assert "Obs.Add(" not in source


def test_http_preview_is_queued_and_validated_fail_closed() -> None:
    """HTTP worker分離と全obs response境界の共通検証を確認する。

    初心者向け: Game stateはgame threadだけで読み、壊れたshapeやNaNを送信前に止める。
    """

    source = _read(
        "ReinBalance/Source/ReinBalanceEditor/Private/Training/"
        "SurvivorsHttpEnvService.cpp"
    )
    handler = source.split("bool HandlePreviewLevelUp(", 1)[1].split(
        "FHttpRouteHandle ObsSchemaRoute", 1
    )[0]
    assert 'TEXT("/preview_level_up")' in source
    assert "PreviewLevelUpQueue.Enqueue" in handler
    assert "Game->" not in handler
    assert "ValidateObservationForResponse" in source
    for boundary in (
        "ProcessReset",
        "ProcessStep",
        "BuildLevelUpApplyResponseJson",
        "BuildLevelUpPreviewResponseJson",
    ):
        body = source.split(boundary, 1)[1]
        assert "ValidateObservationForResponse" in body
    for token in (
        "pending decision does not match",
        "choice set changed",
        "obs schema changed",
    ):
        assert token in source


def test_python_publishes_typed_preview_without_projection_formula() -> None:
    """Python client がtyped previewを公開し手動projectionを持たないことを検証する。

    初心者向け: raw observationはUE5から受け取り、Pythonではshape/hash/ID集合だけを検査する。
    """

    env_path = "Tools/Training/games/survivors/survivors_env.py"
    source = _read(env_path)
    contract = _read("Tools/Training/games/survivors/choice_preview.py")

    assert "preview_level_up" in _class_functions(env_path, "SurvivorsEnv")
    assert "preview_level_up" in _class_functions(env_path, "SurvivorsMonitor")
    assert "SurvivorsLevelUpPreview" in contract
    assert "SurvivorsChoicePreview" in contract
    assert "parse_level_up_preview" in contract
    forbidden = (
        "absolute_index",
        "observation_calculator",
        "project_choice_observation",
        "projected_obs[",
    )
    lowered = (source + contract).lower()
    for token in forbidden:
        assert token not in lowered


def test_low_level_tests_cover_all_choice_kinds_and_immutability() -> None:
    """Windows LLTが全choice kind・runtime・100回不変性を列挙することを検証する。

    初心者向け: WSLではLLTを実行できないため、必須scenarioがtest sourceから消えた回帰を先に止める。
    """

    source = _read(
        "ReinBalance/Source/Programs/ReinBalanceLogicTests/Private/Survivors/"
        "SurvivorsChoiceProjectionTests.cpp"
    )
    for scenario in (
        "WeaponNew",
        "WeaponUpgrade",
        "WeaponEvolve",
        "WeaponUnion",
        "PassiveNew",
        "PassiveUpgrade",
    ):
        assert f"EPreviewScenario::{scenario}" in source
    for runtime_state in (
        "GetCooldownRemaining",
        "Projectiles",
        "GroundZones",
        "RandStream.GetCurrentSeed",
        "GetLastAppliedDecisionId",
        "GetLastAppliedChoiceId",
    ):
        assert runtime_state in source
    assert "<= 1.e-6f" in source
    assert "Iteration < 100" in source
