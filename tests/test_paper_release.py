from pathlib import Path

from src.tools import paper_release


def test_build_manifest_freezes_v060() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = paper_release.build_manifest(root)
    assert manifest["best_result_freeze"]["canonical_result"] == "v0.6.0"
    assert manifest["best_result_freeze"]["analysis_only_branches"] == ["v0_7", "v0_8"]
    assert manifest["best_result_freeze"]["bounded_generalization_branch"] == "v0_9"
    assert manifest["best_result_freeze"]["stage_c_started"] is False


def test_build_manifest_contains_generalization_tasks() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = paper_release.build_manifest(root)
    task_names = {task["task_name"] for task in manifest["exact_slice_ids"]["generalization_tasks"]}
    assert {"hellaswag", "piqa", "winogrande", "arc_easy", "arc_challenge", "lambada_openai"} <= task_names


def test_build_note_mentions_commit_hash() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = paper_release.build_manifest(root)
    note = paper_release.build_note(manifest)
    assert "Git commit hash" in note
    assert "Canonical result: `v0.6.0`" in note
