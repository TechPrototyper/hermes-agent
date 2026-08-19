"""Pure tool-call loop guardrail primitives.

The controller in this module is intentionally side-effect free: it tracks
per-turn tool-call observations and returns decisions. Runtime code owns whether
those decisions become warning guidance, synthetic tool results, or controlled
turn halts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from utils import safe_json_loads
from agent.tool_result_classification import file_mutation_result_landed


IDEMPOTENT_TOOL_NAMES = frozenset(
    {
        "read_file",
        "search_files",
        "web_search",
        "web_extract",
        "session_search",
        "browser_snapshot",
        "browser_console",
        "browser_get_images",
        "mcp_filesystem_read_file",
        "mcp_filesystem_read_text_file",
        "mcp_filesystem_read_multiple_files",
        "mcp_filesystem_list_directory",
        "mcp_filesystem_list_directory_with_sizes",
        "mcp_filesystem_directory_tree",
        "mcp_filesystem_get_file_info",
        "mcp_filesystem_search_files",
    }
)

MUTATING_TOOL_NAMES = frozenset(
    {
        "terminal",
        "execute_code",
        "write_file",
        "patch",
        "todo",
        "memory",
        "skill_manage",
        "browser_click",
        "browser_type",
        "browser_press",
        "browser_scroll",
        "browser_navigate",
        "send_message",
        "cronjob",
        "delegate_task",
        "process",
    }
)


@dataclass(frozen=True)
class ToolCallGuardrailConfig:
    """Thresholds for per-turn tool-call loop detection.

    Warnings are enabled by default and never prevent tool execution. Hard stops
    are explicit opt-in so interactive CLI/TUI sessions get a gentle nudge unless
    the user enables circuit-breaker behavior in config.yaml.
    """

    warnings_enabled: bool = True
    hard_stop_enabled: bool = False
    exact_failure_warn_after: int = 2
    exact_failure_block_after: int = 5
    same_tool_failure_warn_after: int = 3
    same_tool_failure_halt_after: int = 8
    no_progress_warn_after: int = 2
    no_progress_block_after: int = 5
    idempotent_tools: frozenset[str] = field(default_factory=lambda: IDEMPOTENT_TOOL_NAMES)
    mutating_tools: frozenset[str] = field(default_factory=lambda: MUTATING_TOOL_NAMES)
    loop_caps: "LoopCapConfig" = field(default_factory=lambda: LoopCapConfig())
    cycle_detection: "CycleDetectionConfig" = field(
        default_factory=lambda: CycleDetectionConfig()
    )
    # Ordered list of increasingly capable subagents the steering message
    # points at when a loop is interrupted ("escalate to the next smarter
    # agent"). Populated from config.yaml (tool_loop_guardrails.
    # escalation_ladder); empty means a generic escalation hint.
    escalation_ladder: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "ToolCallGuardrailConfig":
        """Build config from the `tool_loop_guardrails` config.yaml section."""
        if not isinstance(data, Mapping):
            return cls()

        warn_after = data.get("warn_after")
        if not isinstance(warn_after, Mapping):
            warn_after = {}
        hard_stop_after = data.get("hard_stop_after")
        if not isinstance(hard_stop_after, Mapping):
            hard_stop_after = {}

        defaults = cls()
        return cls(
            warnings_enabled=_as_bool(data.get("warnings_enabled"), defaults.warnings_enabled),
            hard_stop_enabled=_as_bool(data.get("hard_stop_enabled"), defaults.hard_stop_enabled),
            exact_failure_warn_after=_positive_int(
                warn_after.get("exact_failure", data.get("exact_failure_warn_after")),
                defaults.exact_failure_warn_after,
            ),
            same_tool_failure_warn_after=_positive_int(
                warn_after.get("same_tool_failure", data.get("same_tool_failure_warn_after")),
                defaults.same_tool_failure_warn_after,
            ),
            no_progress_warn_after=_positive_int(
                warn_after.get("idempotent_no_progress", data.get("no_progress_warn_after")),
                defaults.no_progress_warn_after,
            ),
            exact_failure_block_after=_positive_int(
                hard_stop_after.get("exact_failure", data.get("exact_failure_block_after")),
                defaults.exact_failure_block_after,
            ),
            same_tool_failure_halt_after=_positive_int(
                hard_stop_after.get("same_tool_failure", data.get("same_tool_failure_halt_after")),
                defaults.same_tool_failure_halt_after,
            ),
            no_progress_block_after=_positive_int(
                hard_stop_after.get("idempotent_no_progress", data.get("no_progress_block_after")),
                defaults.no_progress_block_after,
            ),
            loop_caps=LoopCapConfig.from_mapping(data.get("loop_caps")),
            cycle_detection=CycleDetectionConfig.from_mapping(data.get("cycle_detection")),
            escalation_ladder=_str_tuple(data.get("escalation_ladder")),
        )


def _str_tuple(value: Any) -> tuple[str, ...]:
    """Coerce a config list into a tuple of non-empty strings."""
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


# Per-turn (per-agent-loop) caps on *identical* tool calls. Counts reset at
# the start of every agent loop (reset_for_turn), so the limit is "within a
# single turn" rather than cumulative over the whole session.
#
# Deliberately NOT a total-count cap: a turn may legitimately issue 50
# different web searches, create 50 different bookings, or delegate 50
# distinct subtasks. What is never legitimate is repeating the *same* call
# (same tool, same canonical args) dozens of times — that is a runaway loop.
# The caps therefore count per (tool, args_hash) signature.
_DEFAULT_MAX_IDENTICAL_CALLS_PER_TURN = 50
_DEFAULT_MAX_WEB_SEARCHES_PER_TURN = 50
_DEFAULT_MAX_SUBAGENTS_PER_TURN = 50


@dataclass(frozen=True)
class LoopCapConfig:
    """Per-turn caps on *identical* tool calls (same tool + same args).

    Historically these caps counted every call of a runaway-prone tool
    (inspired by Claude Code v2.1.212), which false-positived on legitimate
    long sequences — e.g. deep research issuing 50 distinct queries. The caps
    now count per call *signature* (tool name + canonical args hash): distinct
    calls are unlimited, only literal repetition is bounded.

    ``max_identical_calls`` applies to every tool; ``max_web_searches`` and
    ``max_subagents`` remain as per-tool overrides for ``web_search`` and
    ``delegate_task`` (kept for config compatibility, now with identical-call
    semantics; ``delegate_task`` still counts spawned agents, per signature).

    Semantics differ from the per-turn loop *detector* above (which keys on
    repeated identical/failing calls and needs ``hard_stop_enabled``): these
    caps fire regardless of ``hard_stop_enabled``. A value of ``0`` disables
    the respective cap (unlimited). A cap block refuses only that signature —
    the turn continues and different calls keep working.
    """

    max_identical_calls: int = _DEFAULT_MAX_IDENTICAL_CALLS_PER_TURN
    max_web_searches: int = _DEFAULT_MAX_WEB_SEARCHES_PER_TURN
    max_subagents: int = _DEFAULT_MAX_SUBAGENTS_PER_TURN

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "LoopCapConfig":
        """Build config from the ``tool_loop_guardrails.loop_caps`` section."""
        if not isinstance(data, Mapping):
            return cls()
        defaults = cls()
        return cls(
            max_identical_calls=_non_negative_int(
                data.get("max_identical_calls"), defaults.max_identical_calls
            ),
            max_web_searches=_non_negative_int(
                data.get("max_web_searches"), defaults.max_web_searches
            ),
            max_subagents=_non_negative_int(
                data.get("max_subagents"), defaults.max_subagents
            ),
        )


@dataclass(frozen=True)
class CycleDetectionConfig:
    """Per-turn detection of repeating tool-call cycles (A-B-C-A-B-C-…).

    A cycle is a trailing sequence of call signatures (tool + canonical args)
    that repeats with a fixed period. Legitimate batch work is *not* a cycle
    in this sense: iterating "template → fill → create" over 20 invoices uses
    different args each round, so every round has different signatures and
    never matches. Only literal round-trips — the same calls with the same
    args, over and over — are detected. Those burn context/KV without any
    progress and are always pathological.

    ``min_period`` starts at 2 because period-1 repetition (A-A-A-…) is
    already covered by the identical-call cap and the no-progress detector.
    Detection warns after ``warn_after_cycles`` full repetitions and blocks
    (halting the turn) after ``block_after_cycles``. ``enabled: false``
    switches the detector off entirely.
    """

    enabled: bool = True
    min_period: int = 2
    max_period: int = 12
    warn_after_cycles: int = 2
    block_after_cycles: int = 3

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "CycleDetectionConfig":
        """Build config from the ``tool_loop_guardrails.cycle_detection`` section."""
        if not isinstance(data, Mapping):
            return cls()
        defaults = cls()
        return cls(
            enabled=_as_bool(data.get("enabled"), defaults.enabled),
            min_period=max(2, _positive_int(data.get("min_period"), defaults.min_period)),
            max_period=_positive_int(data.get("max_period"), defaults.max_period),
            warn_after_cycles=_positive_int(
                data.get("warn_after_cycles"), defaults.warn_after_cycles
            ),
            block_after_cycles=_positive_int(
                data.get("block_after_cycles"), defaults.block_after_cycles
            ),
        )


def _trailing_cycle(
    seq: list[tuple[str, str]], min_period: int, max_period: int
) -> tuple[int, int]:
    """Return ``(cycles, period)`` for the longest trailing cycle in ``seq``.

    Scans periods ``min_period..max_period`` and counts how many times the
    final ``period``-tuple repeats consecutively at the tail of ``seq``.
    Returns the repetition count and period maximizing the covered tail
    length (``cycles * period``); ``(1, 0)`` when nothing repeats.
    """
    best_cycles, best_period = 1, 0
    n = len(seq)
    for period in range(min_period, max_period + 1):
        if 2 * period > n:
            break
        # A uniform tail (A-A-A-…) is plain identical repetition, which the
        # identical-call cap and the failure/no-progress detectors own —
        # treating it as a period-2 "cycle" would double-fire on their turf.
        if len(set(seq[n - period : n])) == 1:
            continue
        cycles = 1
        while True:
            start = n - (cycles + 1) * period
            if start < 0 or seq[start : start + period] != seq[n - period : n]:
                break
            cycles += 1
        if cycles >= 2 and cycles * period > best_cycles * best_period:
            best_cycles, best_period = cycles, period
    return best_cycles, best_period


@dataclass(frozen=True)
class ToolCallSignature:
    """Stable, non-reversible identity for a tool name plus canonical args."""

    tool_name: str
    args_hash: str

    @classmethod
    def from_call(cls, tool_name: str, args: Mapping[str, Any] | None) -> "ToolCallSignature":
        canonical = canonical_tool_args(args or {})
        return cls(tool_name=tool_name, args_hash=_sha256(canonical))

    def to_metadata(self) -> dict[str, str]:
        """Return public metadata without raw argument values."""
        return {"tool_name": self.tool_name, "args_hash": self.args_hash}


@dataclass(frozen=True)
class ToolGuardrailDecision:
    """Decision returned by the tool-call guardrail controller."""

    action: str = "allow"  # allow | warn | block | halt
    code: str = "allow"
    message: str = ""
    tool_name: str = ""
    count: int = 0
    signature: ToolCallSignature | None = None

    @property
    def allows_execution(self) -> bool:
        return self.action in {"allow", "warn"}

    @property
    def should_halt(self) -> bool:
        return self.action in {"block", "halt"}

    def to_metadata(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "action": self.action,
            "code": self.code,
            "message": self.message,
            "tool_name": self.tool_name,
            "count": self.count,
        }
        if self.signature is not None:
            data["signature"] = self.signature.to_metadata()
        return data


def canonical_tool_args(args: Mapping[str, Any]) -> str:
    """Return sorted compact JSON for parsed tool arguments."""
    if not isinstance(args, Mapping):
        raise TypeError(f"tool args must be a mapping, got {type(args).__name__}")
    return json.dumps(
        args,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def classify_tool_failure(tool_name: str, result: str | None) -> tuple[bool, str]:
    """Safety-fallback classifier used only when callers don't pass ``failed``.

    Mirrors ``agent.display._detect_tool_failure`` exactly so the guardrail
    never disagrees with the CLI's user-visible ``[error]`` tag. Production
    callers in ``run_agent.py`` always pass an explicit ``failed=`` derived
    from ``_detect_tool_failure``; this function exists so standalone callers
    (tests, tooling) still get consistent behavior.
    """
    if result is None:
        return False, ""
    if file_mutation_result_landed(tool_name, result):
        return False, ""

    if tool_name == "terminal":
        data = safe_json_loads(result)
        if isinstance(data, dict):
            exit_code = data.get("exit_code")
            if exit_code is not None and exit_code != 0:
                return True, f" [exit {exit_code}]"
        return False, ""

    if tool_name == "memory":
        data = safe_json_loads(result)
        if isinstance(data, dict):
            if data.get("success") is False and "exceed the limit" in data.get("error", ""):
                return True, " [full]"

    lower = result[:500].lower()
    if '"error"' in lower or '"failed"' in lower or result.startswith("Error"):
        return True, " [error]"

    return False, ""


class ToolCallGuardrailController:
    """Per-turn controller for repeated failed/non-progressing tool calls."""

    def __init__(self, config: ToolCallGuardrailConfig | None = None):
        self.config = config or ToolCallGuardrailConfig()
        self.reset_for_turn()

    def reset_for_turn(self) -> None:
        self._exact_failure_counts: dict[ToolCallSignature, int] = {}
        self._same_tool_failure_counts: dict[str, int] = {}
        self._no_progress: dict[ToolCallSignature, tuple[str, int]] = {}
        self._halt_decision: ToolGuardrailDecision | None = None
        # Per-turn runaway-loop state. Reset every turn (this method runs at
        # the start of each run_conversation), so the guards bound a single
        # agent loop rather than accumulating across the session.
        #
        # Identical-call counters, keyed by full signature (tool + args hash):
        # distinct calls are never throttled, only literal repetition.
        self._turn_identical_counts: dict[ToolCallSignature, int] = {}
        # delegate_task spawn counts per args hash (control actions exempt).
        self._turn_subagent_spawns: dict[str, int] = {}
        # Executed-call signature sequence for cycle detection (appended in
        # after_call, so blocked/never-executed calls don't pollute it).
        self._turn_call_sequence: list[tuple[str, str]] = []
        # How many times this turn a cycle was soft-blocked with a steering
        # message. Used for the last-resort halt when steering is ignored.
        self._cycle_soft_blocks = 0

    @property
    def halt_decision(self) -> ToolGuardrailDecision | None:
        return self._halt_decision

    def before_call(self, tool_name: str, args: Mapping[str, Any] | None) -> ToolGuardrailDecision:
        signature = ToolCallSignature.from_call(tool_name, _coerce_args(args))

        # ── Per-turn identical-call caps ────────────────────────────────
        # Ceilings on how often the *same* call (tool + args) may repeat in a
        # single turn. Distinct calls are unlimited. Applies regardless of
        # hard_stop_enabled (which only governs the per-turn loop detector).
        # We block BEFORE the call runs once the count is already at the cap,
        # then increment for an allowed call so the (cap+1)-th is refused.
        cap_block = self._check_loop_cap(tool_name, _coerce_args(args), signature)
        if cap_block is not None:
            return cap_block

        # ── Per-turn cycle detection (A-B-C-A-B-C-…) ────────────────────
        # Steering-first: a detected cycle warns, then soft-blocks the call
        # with a "take what you have and move on / escalate" message instead
        # of aborting the turn. Only if the model keeps cycling despite
        # repeated steering does the turn halt as a last resort.
        cycle_warn = self._check_cycles(tool_name, signature)
        if cycle_warn is not None and cycle_warn.should_halt:
            return cycle_warn
        if cycle_warn is not None and cycle_warn.action == "block":
            return cycle_warn

        if not self.config.hard_stop_enabled:
            return cycle_warn or ToolGuardrailDecision(tool_name=tool_name, signature=signature)

        exact_count = self._exact_failure_counts.get(signature, 0)
        if exact_count >= self.config.exact_failure_block_after:
            decision = ToolGuardrailDecision(
                action="block",
                code="repeated_exact_failure_block",
                message=(
                    f"Blocked {tool_name}: the same tool call failed {exact_count} "
                    "times with identical arguments. Stop retrying it unchanged; "
                    "change strategy or explain the blocker."
                ),
                tool_name=tool_name,
                count=exact_count,
                signature=signature,
            )
            self._halt_decision = decision
            return decision

        if self._is_idempotent(tool_name):
            record = self._no_progress.get(signature)
            if record is not None:
                _result_hash, repeat_count = record
                if repeat_count >= self.config.no_progress_block_after:
                    decision = ToolGuardrailDecision(
                        action="block",
                        code="idempotent_no_progress_block",
                        message=(
                            f"Blocked {tool_name}: this read-only call returned the same "
                            f"result {repeat_count} times. Stop repeating it unchanged; "
                            "use the result already provided or try a different query."
                        ),
                        tool_name=tool_name,
                        count=repeat_count,
                        signature=signature,
                    )
                    self._halt_decision = decision
                    return decision

        return cycle_warn or ToolGuardrailDecision(tool_name=tool_name, signature=signature)

    def after_call(
        self,
        tool_name: str,
        args: Mapping[str, Any] | None,
        result: str | None,
        *,
        failed: bool | None = None,
    ) -> ToolGuardrailDecision:
        args = _coerce_args(args)
        signature = ToolCallSignature.from_call(tool_name, args)
        # Record the executed call for cycle detection. after_call only runs
        # for calls that actually executed, so blocked calls never pollute
        # the sequence.
        self._turn_call_sequence.append((signature.tool_name, signature.args_hash))
        if failed is None:
            failed, _ = classify_tool_failure(tool_name, result)

        if failed:
            exact_count = self._exact_failure_counts.get(signature, 0) + 1
            self._exact_failure_counts[signature] = exact_count
            self._no_progress.pop(signature, None)

            same_count = self._same_tool_failure_counts.get(tool_name, 0) + 1
            self._same_tool_failure_counts[tool_name] = same_count

            if self.config.hard_stop_enabled and same_count >= self.config.same_tool_failure_halt_after:
                decision = ToolGuardrailDecision(
                    action="halt",
                    code="same_tool_failure_halt",
                    message=(
                        f"Stopped {tool_name}: it failed {same_count} times this turn. "
                        "Stop retrying the same failing tool path and choose a different approach."
                    ),
                    tool_name=tool_name,
                    count=same_count,
                    signature=signature,
                )
                self._halt_decision = decision
                return decision

            if self.config.warnings_enabled and exact_count >= self.config.exact_failure_warn_after:
                return ToolGuardrailDecision(
                    action="warn",
                    code="repeated_exact_failure_warning",
                    message=(
                        f"{tool_name} has failed {exact_count} times with identical arguments. "
                        "This looks like a loop; inspect the error and change strategy "
                        "instead of retrying it unchanged."
                    ),
                    tool_name=tool_name,
                    count=exact_count,
                    signature=signature,
                )

            if self.config.warnings_enabled and same_count >= self.config.same_tool_failure_warn_after:
                return ToolGuardrailDecision(
                    action="warn",
                    code="same_tool_failure_warning",
                    message=_tool_failure_recovery_hint(tool_name, same_count),
                    tool_name=tool_name,
                    count=same_count,
                    signature=signature,
                )

            return ToolGuardrailDecision(tool_name=tool_name, count=exact_count, signature=signature)

        self._exact_failure_counts.pop(signature, None)
        self._same_tool_failure_counts.pop(tool_name, None)

        if not self._is_idempotent(tool_name):
            self._no_progress.pop(signature, None)
            return ToolGuardrailDecision(tool_name=tool_name, signature=signature)

        result_hash = _result_hash(result)
        previous = self._no_progress.get(signature)
        repeat_count = 1
        if previous is not None and previous[0] == result_hash:
            repeat_count = previous[1] + 1
        self._no_progress[signature] = (result_hash, repeat_count)

        if self.config.warnings_enabled and repeat_count >= self.config.no_progress_warn_after:
            return ToolGuardrailDecision(
                action="warn",
                code="idempotent_no_progress_warning",
                message=(
                    f"{tool_name} returned the same result {repeat_count} times. "
                    "Use the result already provided or change the query instead of "
                    "repeating it unchanged."
                ),
                tool_name=tool_name,
                count=repeat_count,
                signature=signature,
            )

        return ToolGuardrailDecision(tool_name=tool_name, count=repeat_count, signature=signature)

    def _is_idempotent(self, tool_name: str) -> bool:
        if tool_name in self.config.mutating_tools:
            return False
        return tool_name in self.config.idempotent_tools

    def _escalation_hint(self) -> str:
        """Steering suffix: take what you have, move on, escalate if stuck."""
        base = (
            " That's enough of this call. Take the results you already have and "
            "continue the task with them."
        )
        ladder = self.config.escalation_ladder
        if ladder:
            rungs = " -> ".join(ladder)
            return base + (
                " If you have not reached your goal with what you have, do not "
                "repeat this call — escalate to the next more capable subagent "
                f"instead (escalation order: {rungs})."
            )
        return base + (
            " If you have not reached your goal with what you have, do not "
            "repeat this call — escalate to a more capable subagent instead."
        )

    def _check_loop_cap(
        self,
        tool_name: str,
        args: Mapping[str, Any],
        signature: ToolCallSignature,
    ) -> ToolGuardrailDecision | None:
        """Enforce and advance the per-turn *identical-call* counters.

        Counts per (tool, args) signature — distinct calls are unlimited.
        Returns a ``block`` decision (steering message, no turn halt) when the
        signature has already repeated up to its cap, otherwise increments and
        returns ``None``. A cap of 0 disables that limit. ``delegate_task``
        counts spawned agents per signature; control actions are exempt.
        Counters reset each turn via ``reset_for_turn``.
        """
        caps = self.config.loop_caps

        if tool_name == "delegate_task":
            cap = caps.max_subagents
            if not cap:
                return None
            spawn_count = _subagent_spawn_count(args)
            if spawn_count == 0:
                # Control action (list/steer/stop) — spawns nothing. Never
                # block: once the spawn cap is hit, steering/stopping the
                # existing children is exactly what should still work.
                return None
            spawned = self._turn_subagent_spawns.get(signature.args_hash, 0)
            if spawned >= cap:
                return ToolGuardrailDecision(
                    action="block",
                    code="loop_subagent_cap",
                    message=(
                        f"Blocked delegate_task: this exact delegation has already "
                        f"spawned {spawned} subagents this turn (limit {cap})."
                        + self._escalation_hint()
                    ),
                    tool_name=tool_name,
                    count=spawned,
                    signature=signature,
                )
            self._turn_subagent_spawns[signature.args_hash] = spawned + spawn_count
            return None

        if tool_name == "web_search":
            cap = caps.max_web_searches
            code = "loop_web_search_cap"
        else:
            cap = caps.max_identical_calls
            code = "loop_identical_call_cap"
        if not cap:
            return None

        identical = self._turn_identical_counts.get(signature, 0)
        if identical >= cap:
            # Block only this signature — the turn continues, different
            # calls (and different args for the same tool) keep working.
            return ToolGuardrailDecision(
                action="block",
                code=code,
                message=(
                    f"Blocked {tool_name}: this exact call has been repeated "
                    f"{identical} times this turn (limit {cap})."
                    + self._escalation_hint()
                ),
                tool_name=tool_name,
                count=identical,
                signature=signature,
            )
        self._turn_identical_counts[signature] = identical + 1
        return None

    def _check_cycles(
        self, tool_name: str, signature: ToolCallSignature
    ) -> ToolGuardrailDecision | None:
        """Detect repeating trailing call cycles including the pending call.

        Returns ``None`` when no cycle threshold is reached, a ``warn``
        decision at ``warn_after_cycles``, a soft ``block`` with a steering /
        escalation message at ``block_after_cycles`` (the turn continues), and
        a ``halt`` only when the model keeps attempting cycles after several
        soft blocks.
        """
        cfg = self.config.cycle_detection
        if not cfg.enabled:
            return None
        key = (signature.tool_name, signature.args_hash)
        seq = self._turn_call_sequence + [key]
        cycles, period = _trailing_cycle(seq, cfg.min_period, cfg.max_period)
        if period == 0 or cycles < cfg.warn_after_cycles:
            return None

        if cycles >= cfg.block_after_cycles:
            self._cycle_soft_blocks += 1
            if self._cycle_soft_blocks > 3:
                decision = ToolGuardrailDecision(
                    action="halt",
                    code="tool_call_cycle_halt",
                    message=(
                        f"Stopped: tool calls keep cycling (period {period}, "
                        f"{cycles} repetitions) despite repeated steering. "
                        "Summarize the current state for the user."
                    ),
                    tool_name=tool_name,
                    count=cycles,
                    signature=signature,
                )
                self._halt_decision = decision
                return decision
            return ToolGuardrailDecision(
                action="block",
                code="tool_call_cycle_block",
                message=(
                    f"Blocked {tool_name}: the last tool calls form a repeating "
                    f"cycle (period {period}, repeated {cycles}x) with identical "
                    "arguments each round — this burns context without progress."
                    + self._escalation_hint()
                ),
                tool_name=tool_name,
                count=cycles,
                signature=signature,
            )

        if self.config.warnings_enabled:
            return ToolGuardrailDecision(
                action="warn",
                code="tool_call_cycle_warning",
                message=(
                    f"Tool calls are repeating in a cycle (period {period}, "
                    f"{cycles}x, identical arguments each round). If the next "
                    "round adds nothing new, stop and work with the results you "
                    "already have."
                ),
                tool_name=tool_name,
                count=cycles,
                signature=signature,
            )
        return None


def toolguard_synthetic_result(decision: ToolGuardrailDecision) -> str:
    """Build a synthetic role=tool content string for a blocked tool call."""
    return json.dumps(
        {
            "error": decision.message,
            "guardrail": decision.to_metadata(),
        },
        ensure_ascii=False,
    )


def append_toolguard_guidance(result: str, decision: ToolGuardrailDecision) -> str:
    """Append runtime guidance to the current tool result content."""
    if decision.action not in {"warn", "halt"} or not decision.message:
        return result
    label = "Tool loop hard stop" if decision.action == "halt" else "Tool loop warning"
    suffix = (
        f"\n\n[{label}: "
        f"{decision.code}; count={decision.count}; {decision.message}]"
    )
    return (result or "") + suffix


def _tool_failure_recovery_hint(tool_name: str, count: int) -> str:
    """Action-oriented guidance for recovering from repeated tool failures."""
    common = (
        f"{tool_name} has failed {count} times this turn. This looks like a loop. "
        "Do not switch to text-only replies; keep using tools, but diagnose before retrying. "
        "First inspect the latest error/output and verify your assumptions. "
    )
    if tool_name == "terminal":
        return common + (
            "For terminal failures, run a small diagnostic such as `pwd && ls -la` "
            "in the same tool, then try an absolute path, a simpler command, a different "
            "working directory, or a different tool such as read_file/write_file/patch."
        )
    return common + (
        "Try different arguments, a narrower query/path, an absolute path when relevant, "
        "or a different tool that can make progress. If the blocker is external, report "
        "the blocker after one diagnostic attempt instead of repeating the same failing path."
    )


def _coerce_args(args: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return args if isinstance(args, Mapping) else {}


def _result_hash(result: str | None) -> str:
    parsed = safe_json_loads(result or "")
    if parsed is not None:
        try:
            canonical = json.dumps(
                parsed,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
        except TypeError:
            canonical = str(parsed)
    else:
        canonical = result or ""
    return _sha256(canonical)


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on", "enabled"}:
            return True
        if lowered in {"0", "false", "no", "off", "disabled"}:
            return False
    return default


def _positive_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 1 else default


def _non_negative_int(value: Any, default: int) -> int:
    """Parse a session-cap value. 0 is a valid (disable) value; negatives and
    junk fall back to the default."""
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def _subagent_spawn_count(args: Mapping[str, Any]) -> int:
    """How many subagents a single delegate_task call spawns.

    delegate_task runs in one of two modes: a batch (``tasks`` is a non-empty
    list, one child per item) or a single task (``goal``). Count the batch size
    when present, otherwise 1, so the session subagent cap reflects real spawns
    rather than delegate_task invocations. Control actions (list/steer/stop)
    spawn nothing and must not consume the cap.
    """
    if isinstance(args, Mapping):
        action = str(args.get("action") or "").strip().lower()
        if action in ("list", "steer", "stop"):
            return 0
    tasks = args.get("tasks") if isinstance(args, Mapping) else None
    if isinstance(tasks, list) and tasks:
        return len(tasks)
    return 1


def _sha256(value: str) -> str:
    # surrogatepass: tool results scraped from the web can carry unpaired
    # UTF-16 surrogates (e.g. half of a mathematical-bold pair); a strict
    # encode raises and takes down the whole conversation loop. The hash only
    # needs deterministic bytes, not valid UTF-8.
    return hashlib.sha256(value.encode("utf-8", "surrogatepass")).hexdigest()
