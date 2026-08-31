"""Base-skill reconciliation — the CP-5 3-way merge (companion to skills_sync).

WHY: ``hermes update`` (tools/skills_sync.sync_skills) protects locally-edited
base skills by NOT overwriting them. But if upstream ALSO shipped a new version
of the same skill (sync matrix "case 4"), a plain skip silently drops the
upstream improvement. sync_skills now instead PRESERVES ours, STAGES theirs,
and records a pending entry. This module consumes that queue and produces a
merged result that ADOPTS upstream's change while RE-APPLYING our customization.

THREE-WAY: base = the last upstream version we shipped (snapshotted by
sync_skills under ~/.hermes/skill-baseline/), ours = the current on-disk skill,
theirs = the new upstream version (staged under ~/.hermes/skill-reconcile/).

MERGE STRATEGY per file (fail-safe, cheapest-first):
  1. Trivial: if only one side changed vs base, take that side.
  2. Mechanical: ``git merge-file`` (diff3). Clean → use it, no LLM needed.
  3. Semantic: on a diff3 conflict (or SKILL.md prose, or no baseline), ask the
     configured LLM backend for a merged file. Fail-open: if no backend is
     reachable, keep the diff3 output WITH conflict markers so a human can
     finish it — never a silent wrong merge.

REVIEW-GATED: the merge is written to a staging ``merged/`` tree; it only
replaces the live skill on explicit apply (``--apply``), always after backing
up the current copy. Applying re-baselines to theirs and advances the manifest
origin hash, so the next sync sees a clean state.

BACKEND (env, or hermes config): the driver is deliberately node-agnostic —
Tim's pattern is to reconcile on a PEER (RTX update → drive on Spark, and vice
versa) or on the Mac (Codex/Antigravity/Claude).
  HERMES_SKILL_RECONCILE_BACKEND = http | cmd | none   (default: none → stage+diff3 only)
  http:  ..._URL (OpenAI-compatible /v1/chat/completions), ..._MODEL, ..._API_KEY
  cmd:   ..._CMD (shell command; prompt on stdin, merged file on stdout)
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tools.skills_sync import (
    _baseline_root,
    _reconcile_root,
    _read_pending,
    _write_pending,
    _skills_dir,
    _rmtree_writable,
    _rmtree_reconcile,
    _read_manifest,
    _write_manifest,
    _dir_hash,
)

logger = logging.getLogger(__name__)

# Files we attempt a text merge on. Anything else is treated as opaque (take
# theirs when ours is unchanged, else keep ours and flag).
_TEXT_SUFFIXES = frozenset(
    (".md", ".markdown", ".txt", ".py", ".json", ".yaml", ".yml", ".toml",
     ".cfg", ".ini", ".sh", ".js", ".ts", ".html", ".css", "")
)
_MAX_LLM_CHARS = 60_000  # skip the LLM for pathologically large files


# ────────────────────────────── file model ──────────────────────────────

def _list_files(root: Optional[Path]) -> Dict[str, Path]:
    """Map of relative-posix-path -> absolute path for every file under root."""
    out: Dict[str, Path] = {}
    if root and root.exists():
        for p in sorted(root.rglob("*")):
            if p.is_file():
                out[p.relative_to(root).as_posix()] = p
    return out


def _read_text(p: Optional[Path]) -> Optional[str]:
    if p is None:
        return None
    try:
        return p.read_text(encoding="utf-8")
    except (OSError, IOError, UnicodeDecodeError):
        return None


def _read_bytes(p: Optional[Path]) -> Optional[bytes]:
    if p is None:
        return None
    try:
        return p.read_bytes()
    except (OSError, IOError):
        return None


def _is_text(rel: str) -> bool:
    return Path(rel).suffix.lower() in _TEXT_SUFFIXES


# ────────────────────────────── merge core ──────────────────────────────

def _diff3(base: Optional[str], ours: str, theirs: str) -> Tuple[str, bool]:
    """git merge-file diff3. Returns (merged_text, clean). On no baseline or any
    tool failure, returns a conflict-marked block flagged unclean."""
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        o, b, t = tdp / "ours", tdp / "base", tdp / "theirs"
        o.write_text(ours, encoding="utf-8")
        b.write_text(base if base is not None else "", encoding="utf-8")
        t.write_text(theirs, encoding="utf-8")
        try:
            proc = subprocess.run(
                ["git", "merge-file", "-p", "--diff3",
                 str(o), str(b), str(t)],
                capture_output=True, text=True, timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            marker = (
                f"<<<<<<< ours\n{ours}\n||||||| base\n"
                f"{base or ''}\n=======\n{theirs}\n>>>>>>> theirs\n"
            )
            return marker, False
        # returncode 0 = clean; >0 = number of conflicts; <0 = error
        return proc.stdout, proc.returncode == 0


def merge_file_text(
    rel: str, base: Optional[str], ours: str, theirs: str
) -> Tuple[str, str]:
    """Three-way merge one text file. Returns (merged_text, method) where method
    is one of: trivial-theirs, trivial-ours, converged, diff3, llm, conflict."""
    if base is not None and ours == base:
        return theirs, "trivial-theirs"        # only upstream changed
    if base is not None and theirs == base:
        return ours, "trivial-ours"            # only we changed
    if ours == theirs:
        return ours, "converged"

    merged, clean = _diff3(base, ours, theirs)
    if clean:
        return merged, "diff3"

    # Conflict (or no baseline) → try the LLM backend.
    llm = _llm_merge(rel, base, ours, theirs)
    if llm is not None:
        return llm, "llm"
    # Fail-open: hand back the conflict-marked diff3 for human resolution.
    return merged, "conflict"


# ────────────────────────────── LLM backend ─────────────────────────────

def _backend_config() -> dict:
    kind = os.environ.get("HERMES_SKILL_RECONCILE_BACKEND", "none").strip().lower()
    return {
        "kind": kind,
        "url": os.environ.get("HERMES_SKILL_RECONCILE_URL", "").strip(),
        "model": os.environ.get("HERMES_SKILL_RECONCILE_MODEL", "").strip(),
        "api_key": os.environ.get("HERMES_SKILL_RECONCILE_API_KEY", "").strip(),
        "cmd": os.environ.get("HERMES_SKILL_RECONCILE_CMD", "").strip(),
    }


def backend_available() -> bool:
    cfg = _backend_config()
    if cfg["kind"] == "http":
        return bool(cfg["url"])
    if cfg["kind"] == "cmd":
        return bool(cfg["cmd"])
    return False


def _build_prompt(rel: str, base: Optional[str], ours: str, theirs: str) -> str:
    intent = (
        "You are reconciling a base-skill file that was edited BOTH locally and "
        "upstream. Produce ONE merged version of the file that:\n"
        "1. ADOPTS upstream's functional improvements/innovations (BASE -> THEIRS).\n"
        "2. PRESERVES the local customizations (BASE -> OURS).\n"
        "Keep BOTH intents where they don't collide; upstream's new capabilities "
        "must not be dropped and local behavior must not be lost. Preserve YAML "
        "frontmatter keys from both sides. Output ONLY the merged file content — "
        "no commentary, no code fences.\n\n"
    )
    if base is None:
        intent = intent.replace(
            "(BASE -> THEIRS)", "(infer upstream's additions by comparing THEIRS to OURS)"
        ).replace(
            "(BASE -> OURS)", "(infer the local customizations by comparing OURS to THEIRS)"
        )
    parts = [intent, f"# FILE: {rel}\n"]
    if base is not None:
        parts.append(f"===== BASE (last common upstream) =====\n{base}\n")
    parts.append(f"===== OURS (local — keep these customizations) =====\n{ours}\n")
    parts.append(f"===== THEIRS (new upstream — keep these improvements) =====\n{theirs}\n")
    parts.append("===== MERGED (output only this) =====\n")
    return "\n".join(parts)


def _llm_merge(
    rel: str, base: Optional[str], ours: str, theirs: str
) -> Optional[str]:
    """Ask the configured backend for a merged file. Returns None on any failure
    (unconfigured, unreachable, oversized) so the caller can fall back."""
    cfg = _backend_config()
    if cfg["kind"] == "none":
        return None
    if max(len(ours), len(theirs), len(base or "")) > _MAX_LLM_CHARS:
        logger.debug("skipping LLM merge for %s (too large)", rel)
        return None
    prompt = _build_prompt(rel, base, ours, theirs)
    try:
        if cfg["kind"] == "http":
            return _llm_http(cfg, prompt)
        if cfg["kind"] == "cmd":
            return _llm_cmd(cfg, prompt)
    except Exception:
        logger.debug("LLM merge backend failed for %s", rel, exc_info=True)
    return None


def _llm_http(cfg: dict, prompt: str) -> Optional[str]:
    import urllib.request
    if not cfg["url"]:
        return None
    body = json.dumps({
        "model": cfg["model"] or "default",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "stream": False,
    }).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if cfg["api_key"]:
        headers["Authorization"] = f"Bearer {cfg['api_key']}"
    req = urllib.request.Request(cfg["url"], data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    try:
        return _strip_fences(data["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError):
        return None


def _llm_cmd(cfg: dict, prompt: str) -> Optional[str]:
    import shlex
    proc = subprocess.run(
        shlex.split(cfg["cmd"]),
        input=prompt, capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    return _strip_fences(proc.stdout)


def _strip_fences(text: str) -> str:
    """Defensive: drop a single leading/trailing ``` fence if the model added one."""
    t = text.strip("\n")
    lines = t.split("\n")
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
    return "\n".join(lines)


# ────────────────────────────── orchestration ───────────────────────────

def _entry(name: str) -> Optional[dict]:
    return _read_pending().get(name)


def reconcile_status() -> List[dict]:
    """The reconcile queue with whether a merged candidate is already staged."""
    out = []
    for name, meta in sorted(_read_pending().items()):
        merged = _reconcile_root() / name / "merged"
        out.append({
            "name": name,
            "dest": meta.get("dest", ""),
            "baseline_present": meta.get("baseline_present", False),
            "merged_staged": merged.exists(),
        })
    return out


def build_merge(name: str, quiet: bool = False) -> Optional[dict]:
    """Produce the merged candidate for a pending skill under merged/. Does NOT
    touch the live skill. Returns a summary dict, or None if `name` isn't pending."""
    meta = _entry(name)
    if not meta:
        return None
    rel = meta["install_rel"]
    dest = Path(meta["dest"])
    base_root = _baseline_root() / rel if meta.get("baseline_present") else None
    ours_root = dest
    theirs_root = _reconcile_root() / name / "theirs"
    merged_root = _reconcile_root() / name / "merged"

    if merged_root.exists():
        _rmtree_reconcile(merged_root)
    merged_root.mkdir(parents=True, exist_ok=True)

    base_files = _list_files(base_root)
    ours_files = _list_files(ours_root)
    theirs_files = _list_files(theirs_root)
    all_rels = sorted(set(base_files) | set(ours_files) | set(theirs_files))

    methods: Dict[str, str] = {}
    conflicts: List[str] = []
    for r in all_rels:
        decision, content_text, content_bytes = _merge_one(
            r, base_files.get(r), ours_files.get(r), theirs_files.get(r)
        )
        methods[r] = decision
        if decision == "conflict":
            conflicts.append(r)
        if decision == "drop":
            continue
        target = merged_root / r
        target.parent.mkdir(parents=True, exist_ok=True)
        if content_bytes is not None:
            target.write_bytes(content_bytes)
        elif content_text is not None:
            target.write_text(content_text, encoding="utf-8")

    summary = {
        "name": name,
        "files": methods,
        "conflicts": conflicts,
        "used_llm": any(m == "llm" for m in methods.values()),
        "backend": _backend_config()["kind"],
        "merged_root": str(merged_root),
    }
    if not quiet:
        _print_summary(summary)
    return summary


def _merge_one(
    rel: str, base: Optional[Path], ours: Optional[Path], theirs: Optional[Path]
) -> Tuple[str, Optional[str], Optional[bytes]]:
    """Merge a single relative path across the three trees.
    Returns (decision, text_or_None, bytes_or_None)."""
    ob, bb, tb = _read_bytes(ours), _read_bytes(base), _read_bytes(theirs)

    # Presence-based trivial cases first (work for binary too).
    if ours is None and theirs is not None and base is None:
        return "add-theirs", None, tb                    # new upstream file
    if ours is not None and theirs is None and base is None:
        return "keep-ours", None, ob                     # our own new file
    if ours is not None and theirs is None and base is not None:
        # upstream deleted: honor deletion only if we hadn't changed it
        if ob == bb:
            return "drop", None, None
        return "keep-ours-flag", None, ob                # we edited a file upstream removed
    if ours is None and theirs is not None and base is not None:
        # we deleted a file upstream changed → respect our deletion, but flag
        return "drop-flag", None, None

    # Present on at least ours & theirs.
    if ob == tb:
        return "converged", None, ob
    if base is not None and ob == bb:
        return "trivial-theirs", None, tb
    if base is not None and tb == bb:
        return "trivial-ours", None, ob

    if not _is_text(rel):
        # Opaque binary that differs on both sides — can't merge; keep ours, flag.
        return "keep-ours-flag", None, ob

    base_t = _read_text(base)
    ours_t = _read_text(ours) or ""
    theirs_t = _read_text(theirs) or ""
    merged, method = merge_file_text(rel, base_t, ours_t, theirs_t)
    return method, merged, None


def apply_merge(name: str, quiet: bool = False) -> bool:
    """Replace the live skill with the staged merged candidate (review-gated apply).
    Backs up the current copy, re-baselines to theirs, advances the manifest, and
    clears the pending entry. Returns True on success."""
    meta = _entry(name)
    if not meta:
        if not quiet:
            print(f"  ! {name}: not in the reconcile queue")
        return False
    merged_root = _reconcile_root() / name / "merged"
    if not merged_root.exists():
        if not quiet:
            print(f"  ! {name}: no merged candidate — run build first")
        return False
    dest = Path(meta["dest"])
    theirs_root = _reconcile_root() / name / "theirs"
    backup = dest.with_suffix(".reconcile.bak")
    try:
        if backup.exists():
            _rmtree_writable(backup)
        if dest.exists():
            shutil.move(str(dest), str(backup))
        shutil.copytree(merged_root, dest)
    except (OSError, IOError):
        # Restore on failure — never leave the user without their skill.
        if not dest.exists() and backup.exists():
            shutil.move(str(backup), str(dest))
        logger.warning("apply_merge failed for %s", name, exc_info=True)
        if not quiet:
            print(f"  ! {name}: apply failed, restored original")
        return False

    # Re-baseline to theirs (the new common ancestor) and advance the manifest
    # so the next sync sees ours == origin (case 3), not case 4.
    try:
        base = _baseline_root() / meta["install_rel"]
        if base.exists():
            _rmtree_reconcile(base)
        base.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(theirs_root, base)
    except (OSError, IOError):
        logger.debug("re-baseline after apply failed for %s", name, exc_info=True)
    try:
        manifest = _read_manifest()
        manifest[name] = meta.get("bundled_hash") or _dir_hash(dest)
        _write_manifest(manifest)
    except Exception:
        logger.debug("manifest advance after apply failed for %s", name, exc_info=True)

    # Clear the queue entry + staged trees; keep the .bak for one cycle.
    pending = _read_pending()
    pending.pop(name, None)
    _write_pending(pending)
    try:
        _rmtree_reconcile(_reconcile_root() / name)
    except (OSError, IOError):
        pass
    if not quiet:
        print(f"  ✓ {name}: merged version applied (backup at {backup})")
    return True


def _print_summary(summary: dict) -> None:
    n = summary["name"]
    print(f"\n  Reconcile plan for '{n}' (backend: {summary['backend']}):")
    for rel, method in sorted(summary["files"].items()):
        mark = "⚠" if method == "conflict" else " "
        print(f"    {mark} {rel}: {method}")
    if summary["conflicts"]:
        print(
            f"    ⚠ {len(summary['conflicts'])} file(s) have unresolved conflict "
            f"markers — edit the staged copy under {summary['merged_root']} "
            f"before applying."
        )
    print(f"    Staged at: {summary['merged_root']}")


# ────────────────────────────── CLI entry ───────────────────────────────

def do_reconcile(
    name: str = "", apply: bool = False, quiet: bool = False
) -> dict:
    """`hermes skills reconcile [name] [--apply]`.

    No name → list the queue. With a name → build the merged candidate (and
    apply it when --apply and there are no unresolved conflicts)."""
    if not name:
        status = reconcile_status()
        if not status:
            if not quiet:
                print("  No skills need reconciliation.")
            return {"queue": []}
        if not quiet:
            print("  Skills needing reconciliation (local edits + upstream update):")
            for s in status:
                staged = " [merged staged]" if s["merged_staged"] else ""
                base = "" if s["baseline_present"] else " [no baseline → 2-way]"
                print(f"    ⚑ {s['name']}{staged}{base}")
            if not backend_available():
                print(
                    "\n  Note: no LLM backend configured "
                    "(HERMES_SKILL_RECONCILE_BACKEND). Merges will use diff3 only; "
                    "conflicts will need manual resolution.\n"
                    "  Configure a peer node (RTX↔Spark) or a local CLI to enable "
                    "semantic 3-way merges."
                )
            print("\n  Run `hermes skills reconcile <name>` to build a merge.")
        return {"queue": [s["name"] for s in status]}

    summary = build_merge(name, quiet=quiet)
    if summary is None:
        if not quiet:
            print(f"  ! '{name}' is not in the reconcile queue.")
        return {"error": "not-pending", "name": name}

    if apply:
        if summary["conflicts"]:
            if not quiet:
                print(
                    f"  ! Not applying '{name}': {len(summary['conflicts'])} "
                    f"unresolved conflict(s). Resolve the staged copy, then "
                    f"`hermes skills reconcile {name} --apply` again."
                )
            summary["applied"] = False
        else:
            summary["applied"] = apply_merge(name, quiet=quiet)
    else:
        if not quiet:
            print(
                f"\n  Review the merge under {summary['merged_root']}, then "
                f"`hermes skills reconcile {name} --apply` to replace the live skill."
            )
        summary["applied"] = False
    return summary
