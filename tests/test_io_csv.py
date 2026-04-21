from __future__ import annotations

from pathlib import Path

from src.utils.io import save_csv


def test_save_csv_supports_rows_with_different_keys(tmp_path: Path) -> None:
    output_path = tmp_path / "mixed.csv"

    save_csv(
        output_path,
        [
            {"row_type": "variant", "label": "a", "metric_a": 1.0},
            {"row_type": "delta", "label": "b", "metric_b": 2.0},
        ],
    )

    content = output_path.read_text(encoding="utf-8").strip().splitlines()

    assert content[0] == "row_type,label,metric_a,metric_b"
    assert content[1] == "variant,a,1.0,"
    assert content[2] == "delta,b,,2.0"
