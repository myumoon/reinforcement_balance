import pytest

from reinbalance_survivors_contracts.launch_lifecycle import (
    LaunchLifecycle,
    LaunchState,
    PlatformGate,
)


def test_launch_intent_and_activation_cardinality():
    lifecycle = LaunchLifecycle.begin("attempt-1")
    with pytest.raises(ValueError):
        lifecycle.activate("proc-1")
    lifecycle = lifecycle.reserve("run-1", "gameplay-1", "nonce-1", durable=True)
    with pytest.raises(ValueError):
        lifecycle.reserve("run-2", "gameplay-2", "nonce-2", durable=True)
    lifecycle = lifecycle.activate("proc-1")
    assert lifecycle.state is LaunchState.FORMAL_RUN_ACTIVATED
    assert lifecycle.counts_toward_outcome_denominator
    with pytest.raises(ValueError):
        lifecycle.activate("proc-2")


def test_prelaunch_and_uncertain_failures_do_not_mix_with_outcomes():
    failed = LaunchLifecycle.begin("a").preflight_failure("cloud_sync_unknown")
    assert failed.reserved_run_id is None and failed.process_ref is None
    gate = LaunchLifecycle.begin("b").reserve("r", "g", "n", durable=True)
    gate = gate.create_process_failed()
    assert gate.state is LaunchState.LAUNCH_GATE_FAILED
    assert not gate.counts_toward_outcome_denominator
    uncertain = LaunchLifecycle.begin("c").reserve("r2", "g2", "n2", durable=True)
    uncertain = uncertain.launch_uncertain()
    assert uncertain.campaign_blocked and not uncertain.replacement_allowed


@pytest.mark.parametrize("field", ["local_fixed_volume", "ntfs", "wal", "synchronous_full", "integrity_ok", "flush_ok", "broker_attested"])
def test_platform_gate_is_fail_closed(field):
    values = dict(local_fixed_volume=True, ntfs=True, wal=True, synchronous_full=True,
                  integrity_ok=True, flush_ok=True, broker_attested=True)
    values[field] = False
    with pytest.raises(ValueError):
        PlatformGate(**values).validate()


def test_unknown_wire_fields_are_rejected():
    data = LaunchLifecycle.begin("a").to_wire()
    data["future"] = True
    with pytest.raises(ValueError):
        LaunchLifecycle.from_wire(data)
