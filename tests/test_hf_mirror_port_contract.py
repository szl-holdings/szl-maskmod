# SPDX-License-Identifier: Apache-2.0
"""Contract for the maskmod port of the lambda-gate release mirror.

Standard library and pytest only, so the kernel-smoke job can run it without
huggingface_hub or PyYAML installed.
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "hf-mirror.yml"
CONFIG = ROOT / ".github" / "hf-mirror.json"
PUBLISHER = ROOT / "scripts" / "hf_mirror_release.py"
REPO_ID = "SZLHOLDINGS/szl-maskmod"

spec = importlib.util.spec_from_file_location("hf_mirror_release_port", PUBLISHER)
assert spec and spec.loader
mirror = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mirror)


def target() -> dict:
    targets = json.loads(CONFIG.read_text(encoding="utf-8"))["targets"]
    assert len(targets) == 1
    return targets[0]


def run_blocks(text: str) -> list[str]:
    """Return the shell body of every `run:` key in a workflow file."""
    lines = text.splitlines()
    blocks: list[str] = []
    for index, line in enumerate(lines):
        match = re.match(r"^(\s*)(?:- )?run:\s*(.*)$", line)
        if not match:
            continue
        indent, value = len(match.group(1)), match.group(2).strip()
        if value not in ("|", ">", "|-", ">-"):
            blocks.append(value)
            continue
        body = []
        for follow in lines[index + 1:]:
            if follow.strip() and len(follow) - len(follow.lstrip()) <= indent:
                break
            body.append(follow)
        blocks.append("\n".join(body))
    return blocks


def test_target_is_oidc_ready_and_release_only() -> None:
    item = target()
    assert item["slug"] == "maskmod"
    assert item["hf_repo_id"] == REPO_ID
    assert item["oidc_resource"] == REPO_ID
    assert item["repo_type"] == "model"
    assert item["subdirectory"] == "."
    assert item["mirror_on_push"] is False
    assert item["preserve_hub_card"] is True


def test_required_card_metadata_matches_live_hub_card() -> None:
    item = target()
    assert item["required_card_metadata"] == {"library_name": "kernels", "license": "apache-2.0"}
    # The live Hub card declares no tags, so no DOI tag may be required.
    assert item["required_card_tags"] == []


def test_preserve_and_replace_paths_are_safe_and_disjoint() -> None:
    item = target()
    preserve, replace = item["preserve_hub_paths"], item["replace_hub_paths"]
    for name in preserve + replace:
        mirror.safe_relative(name)
    assert len(set(preserve)) == len(preserve)
    assert len(set(replace)) == len(replace)
    assert not set(preserve) & set(replace)
    assert "README.md" not in preserve + replace
    assert "LICENSE" in preserve


def test_replace_paths_are_stageable_source_files() -> None:
    for name in target()["replace_hub_paths"]:
        assert (ROOT / name).is_file(), name
        assert not name.startswith((".github/", "tests/")), name


def test_workflow_run_scripts_never_interpolate_expressions() -> None:
    blocks = run_blocks(WORKFLOW.read_text(encoding="utf-8"))
    assert len(blocks) >= 10
    for block in blocks:
        assert "${{" not in block, block


def test_workflow_actions_are_pinned_to_full_shas() -> None:
    uses = re.findall(r"uses:\s*(\S+)", WORKFLOW.read_text(encoding="utf-8"))
    assert uses
    for ref in uses:
        assert re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", ref), ref


def test_old_workflow_defects_are_absent() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    publisher = PUBLISHER.read_text(encoding="utf-8")
    assert "only='${{ inputs.only }}'" not in text
    assert 'echo "HF_TOKEN=${{ secrets.HF_TOKEN }}"' not in text
    assert "delete_tag" not in text + publisher
    assert "ModelCard.from_template" not in (ROOT / "scripts" / "render_model_card.py").read_text(encoding="utf-8")


def test_collision_plan_allows_identical_and_new_files() -> None:
    item = {"replace_hub_paths": [], "preserve_hub_paths": []}
    staged = {"README.md": "a", "same.py": "b", "new.py": "c"}
    hub = {"README.md": "z", "same.py": "b"}
    assert mirror.collision_plan(staged, hub, item) == {}


def test_collision_plan_fails_closed_on_unlisted_difference() -> None:
    item = {"replace_hub_paths": ["build.toml"], "preserve_hub_paths": []}
    with pytest.raises(RuntimeError, match="collision differs: maskmod.py"):
        mirror.collision_plan({"maskmod.py": "new"}, {"maskmod.py": "old"}, item)


def test_collision_plan_records_listed_replacement() -> None:
    item = {"replace_hub_paths": ["build.toml"], "preserve_hub_paths": ["LICENSE"]}
    plan = mirror.collision_plan({"build.toml": "new", "LICENSE": "x"}, {"build.toml": "old"}, item)
    assert plan == {"build.toml": {"hub_before": "old", "source": "new"}}


@pytest.mark.parametrize(
    ("replace", "preserve", "message"),
    [
        (["README.md"], [], "never replaced"),
        (["LICENSE"], ["LICENSE"], "both preserved and replaceable"),
        (["build.toml", "build.toml"], [], "duplicate"),
        (["../escape"], [], "unsafe path"),
    ],
)
def test_collision_plan_rejects_bad_config(replace: list[str], preserve: list[str], message: str) -> None:
    item = {"replace_hub_paths": replace, "preserve_hub_paths": preserve}
    with pytest.raises(RuntimeError, match=message):
        mirror.collision_plan({}, {}, item)
