"""教師 policy を UE5 で動かし、同じ step の DeployObs v2 と教師出力を記録する正式蒸留収集。

1 step ごとに UE5 応答の flat obs（教師の入力）と、同じ応答の deploy_raw を受け取ります。
教師は flat obs から行動分布の logits・value・行動を出し、deploy_raw は DeployObsWrapper.release()
（Common の共有ビルダー）で DeployObs v2 にします。両者を同じ時刻の行として CombatDistillationDataset に格納し、
fail-closed の開始条件（教師 descriptor・integration fidelity verdict）を通った場合だけ artifact store へ保存します。
"""
from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from games.survivors.combat_distillation_dataset import DATASET_SCHEMA_VERSION, CombatDistillationDataset
from games.survivors.deploy_obs_wrapper import DeployObsWrapper
from games.survivors.deploy_raw_env import DEPLOY_RAW_SCHEMA_VERSION, DeployRawEnv
from reinbalance_survivors_contracts.artifact_dag import validate_artifact_dag
from reinbalance_survivors_contracts.artifact_identity import ArtifactDescriptor
from reinbalance_survivors_contracts.artifact_store import ArtifactStore
from reinbalance_survivors_contracts.canonical_json import canonical_hash, canonical_json_bytes, sha256_hex
from reinbalance_survivors_contracts.deploy_obs import DeployObsSchema
from reinbalance_survivors_contracts.fidelity_verdict import FidelityVerdict, verify_current_fidelity

PRODUCER_ID = "collect_survivors_combat_distillation"
PRODUCER_VERSION = "v1"
REQUIRED_FIDELITY_STAGE = "integration"
# release DeployObs が持ちうる取得元（unobservable は release で必ず neutral なので dataset には記録しない）
OBSERVATION_SOURCE_CLASSES = ("hud_inventory", "screen_world_observed", "temporal_inferred", "constant")
_SPLITS = ("train", "validation")


class CollectionError(ValueError):
    """正式蒸留収集の開始条件・応答・教師出力が契約に合わないときの例外。

    CLI はこの例外（と ValueError 系全般）を終了コード 3 にまとめ、収集結果を保存しません。
    """


class Teacher(Protocol):
    """収集ループが使う教師 policy の最小 interface。

    本番は LoadedValueSource を包む ValueSourceTeacher、テストは決定的な fake teacher を渡します。
    act は正規化前の flat obs を受け取り、正規化は教師側で行います。
    """

    action_dim: int
    identity_sha256: str

    def initial_state(self) -> Any:
        """episode 開始時の再帰状態（LSTM なら零）を返す。

        収集ループは episode の最初の step でこの状態を act に渡します。
        """

    def act(self, obs: np.ndarray, state: Any, episode_start: bool) -> tuple[int, np.ndarray, float, Any]:
        """1 step 推論して (行動, logits[A], value, 次の状態) を返す。

        obs は正規化前の flat obs、episode_start は episode の最初の step だけ True です。
        """


class ValueSourceTeacher:
    """01-01 で検証済みの LoadedValueSource を Teacher interface に合わせる adapter。

    VecNormalize で flat obs を正規化し、policy.forward で行動・value・次の LSTM 状態を、
    同じ入力状態の get_distribution で行動分布の logits を求めます（手本: RecurrentPolicySession）。
    """

    def __init__(self, source: Any) -> None:
        """検証済み source と、行動次元・identity・再帰かどうかを保持する。

        Discrete 行動以外の policy は logits を持たないので拒否します。
        """
        space = source.policy.action_space
        if not hasattr(space, "n"):
            raise CollectionError("teacher policy must have a Discrete action space")
        self.source, self.action_dim = source, int(space.n)
        self.identity_sha256 = str(source.descriptor["identity_sha256"])
        self.recurrent = source.algorithm == "RecurrentPPO"

    def initial_state(self) -> Any:
        """RecurrentPPO なら pi/vf の (h, c) を零で返し、PPO なら None を返す。

        形は (n_lstm_layers, 1, lstm_hidden_size) で、episode 境界ごとにこれへ戻します。
        """
        if not self.recurrent:
            return None
        schema = self.source.policy_state_schema
        shape = (schema["n_lstm_layers"], 1, schema["lstm_hidden_size"])
        return tuple(tuple(np.zeros(shape, np.float32) for _ in range(2)) for _ in range(2))

    def act(self, obs: np.ndarray, state: Any, episode_start: bool) -> tuple[int, np.ndarray, float, Any]:
        """flat obs 1 つを正規化して決定的に推論し、(行動, logits, value, 次の状態) を返す。

        episode_start=True のとき SB3 の LSTM は状態を内部で零に戻します（収集ループも零状態を渡す）。
        """
        import torch as th

        batch = np.asarray(obs, dtype=np.float32)
        if batch.shape != (self.source.observation_dim,):
            raise CollectionError(f"teacher observation shape must be ({self.source.observation_dim},)")
        policy = self.source.policy
        obs_tensor, _ = policy.obs_to_tensor(self.source.normalize_raw_obs(batch[None, :]))
        with th.no_grad():
            if self.recurrent:
                from sb3_contrib.common.recurrent.type_aliases import RNNStates

                device = policy.device
                pi, vf = (tuple(th.as_tensor(x, dtype=th.float32, device=device) for x in pair) for pair in state)
                starts = th.as_tensor([float(episode_start)], dtype=th.float32, device=device)
                actions, values, _, next_states = policy.forward(obs_tensor, RNNStates(pi, vf), starts, deterministic=True)
                distribution, _ = policy.get_distribution(obs_tensor, pi, starts)
                next_state = tuple(
                    tuple(x.detach().cpu().numpy().astype(np.float32, copy=True) for x in pair)
                    for pair in (next_states.pi, next_states.vf)
                )
            else:
                actions, values, _ = policy.forward(obs_tensor, deterministic=True)
                distribution, next_state = policy.get_distribution(obs_tensor), None
        logits = distribution.distribution.logits.detach().cpu().numpy().reshape(-1).astype(np.float32)
        return int(actions.cpu().numpy().reshape(-1)[0]), logits, float(values.cpu().numpy().reshape(-1)[0]), next_state


def load_teacher(source_descriptor: Path) -> ValueSourceTeacher:
    """source descriptor を 01-01 の loader で検証・ロードし、Teacher にして返す。

    descriptor・hash・model・VecNormalize のどれかが不正なら ValueSourceLoadError（ValueError）で止まります。
    """
    from games.survivors.value_source_loader import load_value_source

    return ValueSourceTeacher(load_value_source(Path(source_descriptor)))


@dataclass(frozen=True)
class CollectedSequences:
    """収集結果。split ごとの dataset と、同じ添字の教師行動（padding は -1）。

    教師行動は release dataset に入れない（03-05 の規則）ため、診断用に別配列で持ちます。
    """

    datasets: Mapping[str, CombatDistillationDataset]
    teacher_actions: Mapping[str, np.ndarray]
    environment_steps: int


def _flat_obs(value: Any) -> np.ndarray:
    """UE5 の flat obs を有限な 1 次元 float32 配列として検証して返す。

    欠損（None）・多次元・NaN/inf は黙って教師へ渡さず拒否します。
    """
    if value is None:
        raise CollectionError("response has no flat observation")
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise CollectionError("flat observation must be a finite 1-D vector")
    return array


def _teacher_output(output: Any, action_dim: int) -> tuple[int, np.ndarray, float, Any]:
    """教師の (行動, logits, value, 状態) を形・型・有限性で検証する。

    logits は (action_dim,) の有限値、value は有限、行動は 0..action_dim-1 の int でなければ拒否します。
    """
    if not isinstance(output, tuple) or len(output) != 4:
        raise CollectionError("teacher must return (action, logits, value, state)")
    action, logits, value, state = output
    logits = np.asarray(logits, dtype=np.float32)
    if logits.shape != (action_dim,) or not np.all(np.isfinite(logits)):
        raise CollectionError("teacher logits must be finite with shape (action_dim,)")
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)) or not np.isfinite(value):
        raise CollectionError("teacher value must be a finite number")
    if isinstance(action, bool) or not isinstance(action, (int, np.integer)) or not 0 <= int(action) < action_dim:
        raise CollectionError("teacher action out of range")
    return int(action), logits, float(value), state


def _done(value: Any, label: str) -> bool:
    """terminated / truncated が bool であることを確かめて返す。

    数値や None を真偽として解釈せず、応答の型違いを拒否します。
    """
    if not isinstance(value, (bool, np.bool_)):
        raise CollectionError(f"{label} must be bool")
    return bool(value)


def collect_sequences(
    env: Any,
    teacher: Teacher,
    schema: DeployObsSchema,
    *,
    episodes: int,
    sequence_length: int,
    burn_in: int,
    validation_every: int = 5,
    seed: int | None = None,
    viewport: tuple[int, int] = (1920, 1080),
) -> CollectedSequences:
    """教師で episode を回し、各 step の DeployObs v2 と教師出力を同じ添字で記録する。

    episode ごとに教師の再帰状態を初期化し、最初の step だけ episode_start=True を渡します。
    episode は sequence_length ごとの行に分け、各行の先頭を reset 境界・burn-in 開始にします。
    split は episode 単位（validation_every 個目ごとに validation）で割り当て、行間で混ぜません。
    """
    if type(episodes) is not int or episodes <= 0:
        raise CollectionError("episodes must be a positive int")
    if type(sequence_length) is not int or sequence_length <= 0:
        raise CollectionError("sequence_length must be a positive int")
    if type(burn_in) is not int or not 0 <= burn_in < sequence_length:
        raise CollectionError("burn_in must be in [0, sequence_length)")
    if type(validation_every) is not int or validation_every < 0:
        raise CollectionError("validation_every must be a non-negative int")
    raw_env = DeployRawEnv(env, viewport)
    wrapper = DeployObsWrapper.release(raw_env, schema)
    action_dim = int(teacher.action_dim)
    rows: dict[str, list[tuple[str, list[np.ndarray], list[np.ndarray], list[float], list[int]]]] = {s: [] for s in _SPLITS}
    total = 0
    for episode in range(episodes):
        split = "validation" if validation_every and (episode + 1) % validation_every == 0 else "train"
        deploy_obs, _ = wrapper.reset(seed=None if seed is None else seed + episode)
        state, start = teacher.initial_state(), True
        obs_list, logit_list, value_list, action_list = [], [], [], []
        while True:
            action, logits, value, state = _teacher_output(
                teacher.act(_flat_obs(raw_env.last_flat_obs), state, start), action_dim,
            )
            obs_list.append(np.asarray(deploy_obs, dtype=np.float32))
            logit_list.append(logits)
            value_list.append(value)
            action_list.append(action)
            start = False
            deploy_obs, _reward, terminated, truncated, _info = wrapper.step(action)
            if _done(terminated, "terminated") | _done(truncated, "truncated"):
                break
        total += len(obs_list)
        for chunk, begin in enumerate(range(0, len(obs_list), sequence_length)):
            end = begin + sequence_length
            rows[split].append((
                f"episode-{episode:05d}-chunk-{chunk:04d}",
                obs_list[begin:end], logit_list[begin:end], value_list[begin:end], action_list[begin:end],
            ))
    datasets, actions = {}, {}
    for split, split_rows in rows.items():
        if split_rows:
            datasets[split], actions[split] = _build_dataset(split, split_rows, schema, sequence_length, burn_in, action_dim)
    return CollectedSequences(datasets, actions, total)


def _build_dataset(
    split: str, rows: list, schema: DeployObsSchema, sequence_length: int, burn_in: int, action_dim: int,
) -> tuple[CombatDistillationDataset, np.ndarray]:
    """1 split 分の行を padding 付き配列にし、release 観測検査を通した dataset を返す。

    padding は logits/value を 0、valid_mask を False、教師行動を -1 にします。
    oracle 値（unobservable segment の非 neutral 値）が 1 step でも入っていれば拒否します。
    """
    batch, dim = len(rows), schema.dim * 3
    observations = np.zeros((batch, sequence_length, dim), np.float32)
    logits = np.zeros((batch, sequence_length, action_dim), np.float32)
    values = np.zeros((batch, sequence_length), np.float32)
    valid = np.zeros((batch, sequence_length), np.bool_)
    resets = np.zeros((batch, sequence_length), np.bool_)
    burn = np.zeros((batch, sequence_length), np.bool_)
    actions = np.full((batch, sequence_length), -1, np.int64)
    for row, (_id, obs, row_logits, row_values, row_actions) in enumerate(rows):
        count = len(obs)
        observations[row, :count] = np.stack(obs)
        logits[row, :count] = np.stack(row_logits)
        values[row, :count] = row_values
        actions[row, :count] = row_actions
        valid[row, :count] = True
        resets[row, 0] = True
        burn[row, :min(burn_in, count)] = True
    dataset = CombatDistillationDataset(
        observations=observations, action_logits=logits, teacher_values=values, valid_mask=valid,
        burn_in_mask=burn, episode_reset_mask=resets, episode_ids=tuple(r[0] for r in rows),
        splits=tuple(split for _ in rows), deploy_schema_hash=schema.schema_hash, burn_in_steps=burn_in,
        observation_source_classes=OBSERVATION_SOURCE_CLASSES,
    )
    dataset.assert_release_observations(schema)
    return dataset, actions


def checked_fidelity_verdict(verdict: FidelityVerdict | Mapping[str, Any], current_gating_producer_hashes: Mapping[str, str]) -> FidelityVerdict:
    """verdict を現在の producer hash と integration stage で再検証し、blocking が無いことを確かめる。

    stale・hash 違い・baseline・blocking あり はすべて例外にし、収集・保存を始めさせません。
    """
    checked = verify_current_fidelity(verdict, current_gating_producer_hashes, REQUIRED_FIDELITY_STAGE)
    if checked.blocking_reasons:
        raise CollectionError("fidelity verdict contains blocking reasons")
    return checked


def read_fidelity_verdict(path: Path) -> FidelityVerdict:
    """fidelity verdict JSON を読み、FidelityVerdict の exact-key 検証を通して返す。

    ファイル欠損・JSON 不正・未知キーはすべて例外になります。
    """
    try:
        return FidelityVerdict.from_wire(json.loads(Path(path).read_bytes()))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CollectionError(f"fidelity verdict could not be read: {exc}") from exc


def save_dataset_artifact(
    collected: CollectedSequences,
    *,
    schema: DeployObsSchema,
    output: Path,
    artifact_store: Path,
    teacher_identity_sha256: str,
    teacher_descriptor_path: Path,
    verdict: FidelityVerdict,
    current_gating_producer_hashes: Mapping[str, str],
    collection_config: Mapping[str, Any],
) -> ArtifactDescriptor:
    """split ごとの dataset を保存し、親 identity 付きの descriptor と一緒に artifact store へ登録する。

    保存直前にも fidelity verdict を再検証し、train は release 学習 gate、validation は release 観測検査を通します。
    教師 descriptor と verdict は source_descriptor の root node、dataset は combat_distillation_dataset node になります。
    dataset の logical id は収集設定と保存ファイルの内容 digest から作るので、別の収集が同じ id に当たりません。
    書き出しは output の隣の一時 directory で行い、store 登録まで成功したときだけ output へ rename します。
    """
    checked = checked_fidelity_verdict(verdict, current_gating_producer_hashes)
    if "train" not in collected.datasets:
        raise CollectionError("formal collection produced no train sequences")
    for split, dataset in collected.datasets.items():
        if split == "train":
            dataset.assert_release_training_ready(schema)
        else:
            dataset.assert_release_observations(schema)
    output = Path(output)
    if output.exists():
        raise CollectionError("dataset output already exists")
    store = ArtifactStore(artifact_store)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", suffix=".partial", dir=output.parent))
    try:
        node = _publish_dataset(
            collected, schema=schema, staging=staging, store=store, checked=checked,
            teacher_identity_sha256=teacher_identity_sha256, teacher_descriptor_path=Path(teacher_descriptor_path),
            collection_config=collection_config,
        )
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return node


def _publish_dataset(
    collected: CollectedSequences,
    *,
    schema: DeployObsSchema,
    staging: Path,
    store: ArtifactStore,
    checked: FidelityVerdict,
    teacher_identity_sha256: str,
    teacher_descriptor_path: Path,
    collection_config: Mapping[str, Any],
) -> ArtifactDescriptor:
    """一時 directory に split を書き、内容 digest 入りの logical id で store へ登録して descriptor を返す。

    logical id の下に置く bytes はすべて id の材料（設定・教師 descriptor・verdict・dataset ファイル）から決まるため、
    同じ id への再登録は同じ bytes の冪等な put になり、別内容での再束縛は起きません。
    """
    for split, dataset in collected.datasets.items():
        dataset.save(staging / split)
    split_files = {
        f"{split}/{name}": (staging / split / name, media)
        for split in collected.datasets
        for name, media in (("data.npz", "application/x-npz"), ("manifest.json", "application/json"))
    }
    digest = canonical_hash({
        "collection_config": dict(collection_config),
        "teacher_identity_sha256": teacher_identity_sha256,
        "teacher_descriptor_sha256": sha256_hex(teacher_descriptor_path.read_bytes()),
        "fidelity_verdict_identity_hash": checked.identity_hash,
        "files": {rel: sha256_hex(path.read_bytes()) for rel, (path, _media) in split_files.items()},
    })
    dataset_id = (
        f"survivors-combat-distillation-{teacher_identity_sha256[:12]}-{checked.identity_hash[:12]}-{digest[:16]}"
    )
    teacher_node = ArtifactDescriptor(
        logical_id=f"{dataset_id}/parents/teacher_source_descriptor", node_kind="source_descriptor",
        producer_id="value_source_descriptor", producer_version="v1",
        identity_metadata={"source_role": "teacher_value_source", "identity_sha256": teacher_identity_sha256},
        files=(store.put(
            logical_id=f"{dataset_id}/parents/teacher_source_descriptor.json",
            source_path=teacher_descriptor_path, media_type="application/json",
        ),),
    )
    fidelity_node = ArtifactDescriptor(
        logical_id=f"{dataset_id}/parents/fidelity_verdict", node_kind="source_descriptor",
        producer_id="fidelity_verdict", producer_version="v2",
        identity_metadata={
            "source_role": "integration_fidelity_verdict", "verdict_identity_hash": checked.identity_hash,
            "verdict_stage": checked.verdict_stage, "required_stage": REQUIRED_FIDELITY_STAGE,
        },
        files=(store.put_bytes(
            logical_id=f"{dataset_id}/parents/fidelity_verdict.json",
            data=canonical_json_bytes(checked.to_wire()), media_type="application/json",
        ),),
    )
    files = [
        store.put(logical_id=f"{dataset_id}/{rel}", source_path=path, media_type=media)
        for rel, (path, media) in split_files.items()
    ]
    dataset_node = ArtifactDescriptor(
        logical_id=dataset_id, node_kind="combat_distillation_dataset",
        producer_id=PRODUCER_ID, producer_version=PRODUCER_VERSION,
        identity_metadata={
            **dict(collection_config),
            "dataset_schema_version": DATASET_SCHEMA_VERSION,
            "deploy_schema_hash": schema.schema_hash,
            "deploy_schema_version": schema.schema_version,
            "deploy_raw_schema_version": DEPLOY_RAW_SCHEMA_VERSION,
            "teacher_identity_sha256": teacher_identity_sha256,
            "fidelity_verdict_identity_hash": checked.identity_hash,
            "environment_steps": collected.environment_steps,
            "sequences": {split: len(d.episode_ids) for split, d in collected.datasets.items()},
        },
        parents=(teacher_node.node_ref(), fidelity_node.node_ref()),
        files=tuple(files),
    )
    validate_artifact_dag([dataset_node, teacher_node, fidelity_node])
    for node in (teacher_node, fidelity_node, dataset_node):
        store.put_bytes(
            logical_id=f"{dataset_id}/descriptors/{node.identity_hash}.json",
            data=canonical_json_bytes(node.to_wire()), media_type="application/json",
        )
    (staging / "artifact_descriptor.json").write_bytes(canonical_json_bytes(dataset_node.to_wire()))
    return dataset_node


def run_formal_collection(
    *,
    source_descriptor: Path,
    fidelity_verdict: Path,
    current_gating_producer_hashes: Mapping[str, str],
    env_factory: Callable[[], Any],
    artifact_store: Path,
    output: Path,
    episodes: int,
    sequence_length: int,
    burn_in: int,
    validation_every: int = 5,
    seed: int = 0,
    schema: DeployObsSchema | None = None,
    teacher_factory: Callable[[Path], Teacher] = load_teacher,
) -> ArtifactDescriptor:
    """開始条件を固定の順序で検査してから収集し、artifact store へ保存する正式経路の入口。

    順序: 教師 descriptor のロード → verdict の読込と current-hash 再検証 → blocking なし → output 未使用・store を開ける
    → env 接続・収集 → 保存。どこかで失敗すると env には接続せず（または収集結果を捨てて）例外を返します。
    """
    schema = DeployObsSchema.default_v2() if schema is None else schema
    teacher = teacher_factory(Path(source_descriptor))
    verdict = checked_fidelity_verdict(read_fidelity_verdict(fidelity_verdict), current_gating_producer_hashes)
    if Path(output).exists():
        raise CollectionError("dataset output already exists")
    ArtifactStore(artifact_store)  # store root を作れない・壊れている場合は UE5 に接続する前に止める
    env = env_factory()
    try:
        collected = collect_sequences(
            env, teacher, schema, episodes=episodes, sequence_length=sequence_length,
            burn_in=burn_in, validation_every=validation_every, seed=seed,
        )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    return save_dataset_artifact(
        collected, schema=schema, output=output, artifact_store=artifact_store,
        teacher_identity_sha256=teacher.identity_sha256, teacher_descriptor_path=Path(source_descriptor),
        verdict=verdict, current_gating_producer_hashes=current_gating_producer_hashes,
        collection_config={
            "episodes": episodes, "sequence_length": sequence_length, "burn_in_steps": burn_in,
            "validation_every": validation_every, "seed": seed,
        },
    )
