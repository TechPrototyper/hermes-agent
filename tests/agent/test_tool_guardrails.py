"""Pure tool-call guardrail primitive tests."""

import json

from agent.tool_guardrails import (
    ToolCallGuardrailConfig,
    ToolCallGuardrailController,
    ToolCallSignature,
    canonical_tool_args,
    classify_tool_failure,
)


def test_tool_call_signature_hashes_canonical_nested_unicode_args_without_exposing_raw_args():
    args_a = {
        "z": [{"β": "☤", "a": 1}],
        "a": {"y": 2, "x": "secret-token-value"},
    }
    args_b = {
        "a": {"x": "secret-token-value", "y": 2},
        "z": [{"a": 1, "β": "☤"}],
    }

    assert canonical_tool_args(args_a) == canonical_tool_args(args_b)
    sig_a = ToolCallSignature.from_call("web_search", args_a)
    sig_b = ToolCallSignature.from_call("web_search", args_b)

    assert sig_a == sig_b
    assert len(sig_a.args_hash) == 64
    metadata = sig_a.to_metadata()
    assert metadata == {"tool_name": "web_search", "args_hash": sig_a.args_hash}
    assert "secret-token-value" not in json.dumps(metadata)
    assert "☤" not in json.dumps(metadata)




def test_config_parses_nested_warn_and_hard_stop_thresholds():
    cfg = ToolCallGuardrailConfig.from_mapping(
        {
            "warnings_enabled": False,
            "hard_stop_enabled": True,
            "warn_after": {
                "exact_failure": 3,
                "same_tool_failure": 4,
                "idempotent_no_progress": 5,
            },
            "hard_stop_after": {
                "exact_failure": 6,
                "same_tool_failure": 7,
                "idempotent_no_progress": 8,
            },
        }
    )

    assert cfg.warnings_enabled is False
    assert cfg.hard_stop_enabled is True
    assert cfg.exact_failure_warn_after == 3
    assert cfg.same_tool_failure_warn_after == 4
    assert cfg.no_progress_warn_after == 5
    assert cfg.exact_failure_block_after == 6
    assert cfg.same_tool_failure_halt_after == 7
    assert cfg.no_progress_block_after == 8


def test_default_repeated_identical_failed_call_warns_without_blocking():
    controller = ToolCallGuardrailController()
    args = {"query": "same"}

    decisions = []
    for _ in range(5):
        assert controller.before_call("web_search", args).action == "allow"
        decisions.append(
            controller.after_call("web_search", args, '{"error":"boom"}', failed=True)
        )

    assert decisions[0].action == "allow"
    assert [d.action for d in decisions[1:]] == ["warn", "warn", "warn", "warn"]
    assert {d.code for d in decisions[1:]} == {"repeated_exact_failure_warning"}
    assert controller.before_call("web_search", args).action == "allow"
    assert controller.halt_decision is None


def test_hard_stop_enabled_blocks_repeated_exact_failure_before_next_execution():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            hard_stop_enabled=True,
            exact_failure_warn_after=2,
            exact_failure_block_after=2,
            same_tool_failure_halt_after=99,
        )
    )
    args = {"query": "same"}

    assert controller.before_call("web_search", args).action == "allow"
    first = controller.after_call("web_search", args, '{"error":"boom"}', failed=True)
    assert first.action == "allow"

    assert controller.before_call("web_search", args).action == "allow"
    second = controller.after_call("web_search", args, '{"error":"boom"}', failed=True)
    assert second.action == "warn"
    assert second.code == "repeated_exact_failure_warning"

    blocked = controller.before_call("web_search", args)
    assert blocked.action == "block"
    assert blocked.code == "repeated_exact_failure_block"
    assert blocked.count == 2














def test_mutating_or_unknown_tools_are_not_blocked_for_repeated_identical_success_output_by_default():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(no_progress_warn_after=2, no_progress_block_after=2)
    )

    # Mutating/unknown tools are never blocked by the no-progress detector —
    # but alternating *identical* calls form a period-2 cycle, so from the
    # third full round the cycle detector steers (soft block, no turn halt).
    for _ in range(2):
        assert controller.before_call("write_file", {"path": "/tmp/x", "content": "x"}).allows_execution
        assert controller.after_call("write_file", {"path": "/tmp/x", "content": "x"}, "ok", failed=False).allows_execution
        assert controller.before_call("custom_tool", {"x": 1}).allows_execution
        assert controller.after_call("custom_tool", {"x": 1}, "ok", failed=False).allows_execution
    assert controller.before_call("write_file", {"path": "/tmp/x", "content": "x"}).allows_execution
    assert controller.after_call("write_file", {"path": "/tmp/x", "content": "x"}, "ok", failed=False).allows_execution
    third_round = controller.before_call("custom_tool", {"x": 1})
    assert third_round.action == "block"
    assert third_round.code == "tool_call_cycle_block"
    assert controller.halt_decision is None
    # Fresh args break the pattern immediately.
    assert controller.before_call("custom_tool", {"x": 2}).allows_execution






# ── Per-turn runaway-loop caps (Claude Code v2.1.212, Week 29) ──────────────

from agent.tool_guardrails import LoopCapConfig  # noqa: E402






def test_loop_cap_zero_disables_and_junk_falls_back():
    # 0 is a legitimate "unlimited" value; negatives / junk fall back to default.
    assert LoopCapConfig.from_mapping({"max_web_searches": 0}).max_web_searches == 0
    assert LoopCapConfig.from_mapping({"max_web_searches": -5}).max_web_searches == 50
    assert LoopCapConfig.from_mapping({"max_subagents": "nope"}).max_subagents == 50
    assert LoopCapConfig.from_mapping({"max_identical_calls": 0}).max_identical_calls == 0
    assert LoopCapConfig.from_mapping({}).max_identical_calls == 50


def test_distinct_web_searches_are_unlimited_only_identical_queries_capped():
    # The cap counts identical signatures, not total calls: 20 distinct
    # queries sail through a cap of 3, while the 4th identical query blocks —
    # without halting the turn, so different queries keep working afterwards.
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            hard_stop_enabled=False,
            loop_caps=LoopCapConfig(max_web_searches=3),
        )
    )
    for i in range(20):
        assert controller.before_call("web_search", {"query": f"q{i}"}).action == "allow"
    for _ in range(3):
        assert controller.before_call("web_search", {"query": "same"}).action == "allow"
    decision = controller.before_call("web_search", {"query": "same"})
    assert decision.action == "block"
    assert decision.code == "loop_web_search_cap"
    assert controller.halt_decision is None
    # Turn continues: a different query is still allowed.
    assert controller.before_call("web_search", {"query": "different"}).action == "allow"


def test_identical_call_cap_applies_to_any_tool_and_carries_steering_message():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            hard_stop_enabled=False,
            loop_caps=LoopCapConfig(max_identical_calls=2),
            escalation_ladder=("Antigravity (Gemini Flash)", "Codex (GPT Sol)", "Claude Code (Fable)"),
        )
    )
    for _ in range(2):
        assert controller.before_call("create_booking", {"betrag": "10"}).action == "allow"
    decision = controller.before_call("create_booking", {"betrag": "10"})
    assert decision.action == "block"
    assert decision.code == "loop_identical_call_cap"
    assert "escalate" in decision.message
    assert "Antigravity (Gemini Flash) -> Codex (GPT Sol) -> Claude Code (Fable)" in decision.message
    assert controller.halt_decision is None
    # Same tool with different args is untouched.
    assert controller.before_call("create_booking", {"betrag": "20"}).action == "allow"


def _execute(controller, tool, args):
    decision = controller.before_call(tool, args)
    if decision.allows_execution:
        controller.after_call(tool, args, '{"ok": true}', failed=False)
    return decision


def test_cycle_detection_warns_then_soft_blocks_identical_round_trips():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=False)
    )
    calls = [("tool_a", {"x": 1}), ("tool_b", {"y": 2}), ("tool_c", {"z": 3})]
    # Round 1 executes silently.
    for tool, args in calls:
        assert _execute(controller, tool, args).action == "allow"
    # Round 2 completes the second full cycle -> warning on its last call.
    actions = [_execute(controller, tool, args).action for tool, args in calls]
    assert actions[-1] == "warn"
    # Round 3 completes the third cycle -> soft block (no halt).
    decisions = [_execute(controller, tool, args) for tool, args in calls]
    assert decisions[-1].action == "block"
    assert decisions[-1].code == "tool_call_cycle_block"
    assert "escalate" in decisions[-1].message
    assert controller.halt_decision is None


def test_cycle_detection_halts_only_after_steering_is_repeatedly_ignored():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=False)
    )
    calls = [("tool_a", {"x": 1}), ("tool_b", {"y": 2})]
    seen_halt = False
    for _ in range(12):
        for tool, args in calls:
            decision = _execute(controller, tool, args)
            if decision.action == "halt":
                seen_halt = True
                break
        if seen_halt:
            break
    assert seen_halt
    assert controller.halt_decision is not None
    assert controller.halt_decision.code == "tool_call_cycle_halt"


def test_batch_work_with_varying_args_is_never_a_cycle():
    # Template -> fill -> create over many items uses fresh args every round:
    # signatures differ, so no warning and no block — 30 rounds pass clean.
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=False)
    )
    for i in range(30):
        for tool in ("get_template", "fill_template", "create_invoice"):
            decision = _execute(controller, tool, {"item": i})
            assert decision.action == "allow", (tool, i, decision.code)


def test_cycle_and_escalation_config_parse_from_mapping():
    cfg = ToolCallGuardrailConfig.from_mapping(
        {
            "cycle_detection": {
                "enabled": True,
                "min_period": 1,  # coerced up to 2
                "max_period": 6,
                "warn_after_cycles": 3,
                "block_after_cycles": 4,
            },
            "escalation_ladder": ["Antigravity", "  Codex  ", ""],
            "loop_caps": {"max_identical_calls": 7},
        }
    )
    assert cfg.cycle_detection.enabled is True
    assert cfg.cycle_detection.min_period == 2
    assert cfg.cycle_detection.max_period == 6
    assert cfg.cycle_detection.warn_after_cycles == 3
    assert cfg.cycle_detection.block_after_cycles == 4
    assert cfg.escalation_ladder == ("Antigravity", "Codex")
    assert cfg.loop_caps.max_identical_calls == 7
    disabled = ToolCallGuardrailConfig.from_mapping(
        {"cycle_detection": {"enabled": False}}
    )
    assert disabled.cycle_detection.enabled is False










