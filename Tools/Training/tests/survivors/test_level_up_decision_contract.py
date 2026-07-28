"""外部 level-up decision の production 契約を回帰テストする。

初心者向け: C++ の責務境界と Python の公開 API が片側だけ変わる事故を検出する。
"""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]


def _function_names(path: Path, class_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                child.name
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    raise AssertionError(f"class not found: {class_name}")


def test_logic_owns_external_pending_pause_and_default_auto_mode() -> None:
    header = (
        ROOT
        / "ReinBalance/Source/ReinBalanceLogic/Public/Survivors/SurvivorsGameLogic.h"
    ).read_text(encoding="utf-8")
    source = (
        ROOT
        / "ReinBalance/Source/ReinBalanceLogic/Private/Survivors/SurvivorsGameLogic.cpp"
    ).read_text(encoding="utf-8")

    assert 'ItemSelectionMode = TEXT("auto")' in header
    assert "IsLevelUpPending()" in header
    assert "ApplyExternalLevelUpChoice" in header
    assert "if (IsLevelUpPending())" in source
    assert "LevelUpBacklog" in header


def test_component_has_no_duplicate_production_level_up_body() -> None:
    component = (
        ROOT
        / "ReinBalance/Source/ReinBalance/Private/Survivors/Game/SurvivorsPlayerComponent.cpp"
    ).read_text(encoding="utf-8")

    assert "USurvivorsPlayerComponent::OnLevelUp" not in component
    assert "USurvivorsPlayerComponent::BuildLevelUpChoices" not in component
    assert "USurvivorsPlayerComponent::ApplyLevelUpChoice" not in component
    assert "Game->AddExperience(" in component


def test_python_environment_and_monitor_publish_choice_api() -> None:
    path = ROOT / "Tools/Training/games/survivors/survivors_env.py"
    assert "choose_level_up" in _function_names(path, "SurvivorsEnv")
    assert "choose_level_up" in _function_names(path, "SurvivorsMonitor")
    assert "SurvivorsUE5Env = SurvivorsEnv" in path.read_text(encoding="utf-8")


def test_http_worker_only_enqueues_choice_and_info_fields_coexist() -> None:
    source = (
        ROOT
        / "ReinBalance/Source/ReinBalanceEditor/Private/Training"
        / "SurvivorsHttpEnvService.cpp"
    ).read_text(encoding="utf-8")
    handler = source.split("bool HandleLevelUpChoice(", 1)[1].split(
        "FHttpRouteHandle ObsSchemaRoute", 1
    )[0]

    assert "LevelUpChoiceQueue.Enqueue" in handler
    assert "Game->" not in handler
    assert "TQueue<FLevelUpChoiceRequest, EQueueMode::Mpsc>" in source
    assert 'TEXT("/level_up_choice")' in source
    assert 'TEXT("level_up_pending")' in source
    assert 'TEXT("level_up_choices")' in source
    assert "unknown item_selection_mode" in source
    assert "unknown weapon_pool_mode" in source
    assert "unknown starting_weapon_mode" in source


def test_decision_id_generation_does_not_use_game_rng() -> None:
    source = (
        ROOT
        / "ReinBalance/Source/ReinBalanceLogic/Private/Survivors"
        / "SurvivorsLevelUpDecision.cpp"
    ).read_text(encoding="utf-8")
    assert "EpisodeSerial" in source
    assert "DecisionSequence" in source
    assert "RandStream" not in source
    assert "FRandom" not in source
