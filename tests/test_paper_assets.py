from pathlib import Path

from src.tools import paper_assets


def test_build_tables_includes_tokenwise_summary() -> None:
    root = Path(__file__).resolve().parents[1]
    tables = {table.slug: table for table in paper_assets.build_tables(root)}
    tokenwise = tables["table_05_tokenwise_summary"]
    rows = [row for row in tokenwise.rows if row["phase_name"] == "tokenwise_confirmation_main" and row["compared_model"] == "tokenwise_mixture"]
    assert len(rows) == 1
    row = rows[0]
    assert row["seed_count"] == 3
    assert row["logit_kl_to_teacher_mean"] < 0.26
    assert row["nll_mean"] < 3.0


def test_build_figure_specs_includes_generalization_tasks() -> None:
    root = Path(__file__).resolve().parents[1]
    figures = {figure.slug: figure for figure in paper_assets.build_figure_specs(root)}
    generalization = figures["figure_06_generalization_summary"]
    task_names = {row["task_name"] for row in generalization.spec["rows"]}
    assert {"hellaswag", "piqa", "winogrande", "arc_easy", "arc_challenge", "lambada_openai"} <= task_names


def test_render_markdown_table_formats_basic_values() -> None:
    markdown = paper_assets._render_markdown_table(  # type: ignore[attr-defined]
        ["name", "value", "flag"],
        [{"name": "row", "value": 0.255739320347238, "flag": True}],
    )
    assert "| name | value | flag |" in markdown
    assert "0.255739" in markdown
    assert "True" in markdown
