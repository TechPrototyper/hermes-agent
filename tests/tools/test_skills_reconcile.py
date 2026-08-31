"""CP-5 base-skill reconciliation: sync case-4 detection + 3-way merge + apply."""
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pytest

import tools.skills_sync as sync
import tools.skills_reconcile as rec


def _write(p: Path, text: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _skill(root: Path, name: str, body: str):
    _write(root / name / "SKILL.md", f"---\nname: {name}\n---\n{body}")


def _patches(tmp_path):
    bundled = tmp_path / "bundled" / "skills"
    skills_dir = tmp_path / "home" / "skills"
    home = tmp_path / "home"
    bundled.mkdir(parents=True, exist_ok=True)
    stack = ExitStack()
    stack.enter_context(patch("tools.skills_sync._get_bundled_dir", return_value=bundled))
    stack.enter_context(patch("tools.skills_sync._get_optional_dir",
                              return_value=bundled.parent / "optional-skills"))
    stack.enter_context(patch("tools.skills_sync.SKILLS_DIR", skills_dir))
    stack.enter_context(patch("tools.skills_sync.MANIFEST_FILE", skills_dir / ".bundled_manifest"))
    stack.enter_context(patch("tools.skills_sync.HERMES_HOME", str(home)))
    return stack, bundled, skills_dir, home


def test_case4_detected_and_full_reconcile_cycle(tmp_path):
    stack, bundled, skills_dir, home = _patches(tmp_path)
    with stack:
        # 1) upstream v1, first sync installs it + snapshots baseline
        _skill(bundled, "foo", "A\nB\nC\n")
        r1 = sync.sync_skills(quiet=True)
        assert "foo" in r1["copied"]
        assert (skills_dir / "foo" / "SKILL.md").exists()
        assert (home / "skill-baseline" / "foo" / "SKILL.md").exists()

        # 2) user edits their copy (change line 1) — non-overlapping with upstream
        _skill(skills_dir, "foo", "A-OURS\nB\nC\n")
        # 3) upstream ships v2 (change line 3)
        _skill(bundled, "foo", "A\nB\nC-THEIRS\n")

        # 4) second sync → CASE 4: staged, ours preserved, pending recorded
        r2 = sync.sync_skills(quiet=True)
        assert r2["reconcile_needed"] == ["foo"]
        assert "foo" not in r2["updated"]                     # ours NOT overwritten
        assert "A-OURS" in (skills_dir / "foo" / "SKILL.md").read_text()
        assert (home / "skill-reconcile" / "foo" / "theirs" / "SKILL.md").exists()
        assert "foo" in sync._read_pending()

        # 5) build merge with NO backend → diff3, clean (non-overlapping)
        summary = rec.build_merge("foo", quiet=True)
        assert summary["files"]["SKILL.md"] == "diff3"
        assert summary["conflicts"] == []
        merged = (home / "skill-reconcile" / "foo" / "merged" / "SKILL.md").read_text()
        assert "A-OURS" in merged and "C-THEIRS" in merged   # BOTH intents kept

        # 6) apply → live skill replaced, re-baselined, pending cleared, backup made
        assert rec.apply_merge("foo", quiet=True) is True
        live = (skills_dir / "foo" / "SKILL.md").read_text()
        assert "A-OURS" in live and "C-THEIRS" in live
        assert "foo" not in sync._read_pending()
        assert (skills_dir / "foo.reconcile.bak" / "SKILL.md").exists()
        # baseline advanced to theirs (v2)
        assert "C-THEIRS" in (home / "skill-baseline" / "foo" / "SKILL.md").read_text()

        # 7) self-healing: next sync no longer flags foo (now case 3, kept)
        r3 = sync.sync_skills(quiet=True)
        assert r3["reconcile_needed"] == []


def test_conflict_falls_back_to_llm_backend(tmp_path):
    stack, bundled, skills_dir, home = _patches(tmp_path)
    with stack:
        _skill(bundled, "bar", "one\nTWO\nthree\n")
        sync.sync_skills(quiet=True)
        # OVERLAPPING edits on the same line → diff3 will conflict
        _skill(skills_dir, "bar", "one\nTWO-OURS\nthree\n")
        _skill(bundled, "bar", "one\nTWO-THEIRS\nthree\n")
        r = sync.sync_skills(quiet=True)
        assert r["reconcile_needed"] == ["bar"]

        # Stub LLM backend via cmd: reads the prompt on stdin, emits a merge
        stub = f'{sys.executable} -c "import sys; sys.stdin.read(); print(\'one\'); print(\'TWO-MERGED-BY-LLM\'); print(\'three\')"'
        with patch.dict("os.environ", {
            "HERMES_SKILL_RECONCILE_BACKEND": "cmd",
            "HERMES_SKILL_RECONCILE_CMD": stub,
        }):
            assert rec.backend_available() is True
            summary = rec.build_merge("bar", quiet=True)
        assert summary["files"]["SKILL.md"] == "llm"
        assert summary["used_llm"] is True
        merged = (home / "skill-reconcile" / "bar" / "merged" / "SKILL.md").read_text()
        assert "TWO-MERGED-BY-LLM" in merged


def test_conflict_without_backend_leaves_markers_and_blocks_apply(tmp_path):
    stack, bundled, skills_dir, home = _patches(tmp_path)
    with stack:
        _skill(bundled, "baz", "x\nMID\ny\n")
        sync.sync_skills(quiet=True)
        _skill(skills_dir, "baz", "x\nMID-OURS\ny\n")
        _skill(bundled, "baz", "x\nMID-THEIRS\ny\n")
        sync.sync_skills(quiet=True)
        # no backend → conflict markers, apply refused
        summary = rec.build_merge("baz", quiet=True)
        assert summary["files"]["SKILL.md"] == "conflict"
        res = rec.do_reconcile("baz", apply=True, quiet=True)
        assert res["applied"] is False
        assert "baz" in sync._read_pending()   # still queued, not clobbered


def test_merge_file_text_trivial_cases():
    assert rec.merge_file_text("f.md", "B", "B", "T") == ("T", "trivial-theirs")
    assert rec.merge_file_text("f.md", "B", "O", "B") == ("O", "trivial-ours")
    assert rec.merge_file_text("f.md", "B", "S", "S") == ("S", "converged")
