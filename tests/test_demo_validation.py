from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pytest

from src.demo_validation import (
    REQUIRED_OUTPUTS,
    STANDARD_COLUMNS,
    validate_demo_outputs,
)
from src.viz import plot_rejuvenation_by_group


def _workspace_tempdir() -> TemporaryDirectory:
    base = Path(__file__).resolve().parents[1] / ".pytest_tmp"
    base.mkdir(exist_ok=True)
    return TemporaryDirectory(dir=base)


def _write_minimal_demo_tables(results_dir: Path) -> None:
    for filename, specific_columns in REQUIRED_OUTPUTS.items():
        values = {column: 0 for column in STANDARD_COLUMNS | specific_columns}
        values.update(
            {
                "available": False,
                "estimable": False,
                "reason": "test stub",
                "method": "test",
                "evidence_level": 0,
            }
        )
        pd.DataFrame([values]).to_csv(results_dir / filename, index=False)


def test_demo_output_contract_accepts_structured_stubs() -> None:
    with _workspace_tempdir() as temp_dir:
        results_dir = Path(temp_dir)
        _write_minimal_demo_tables(results_dir)
        assert validate_demo_outputs(results_dir) == len(REQUIRED_OUTPUTS)


def test_demo_output_contract_reports_missing_files() -> None:
    with _workspace_tempdir() as temp_dir:
        with pytest.raises(ValueError, match="file is missing"):
            validate_demo_outputs(Path(temp_dir))


def test_rejuvenation_boxplot_uses_current_matplotlib_api() -> None:
    with _workspace_tempdir() as temp_dir:
        output = Path(temp_dir) / "rejuvenation.png"
        metadata = pd.DataFrame(
            {
                "group": ["control", "control", "treated", "treated"],
                "rejuvenation_score": [-1.0, 0.0, 1.0, 2.0],
            }
        )

        plot_rejuvenation_by_group(
            metadata,
            group_col="group",
            rejuvenation_col="rejuvenation_score",
            out_path=output,
        )

        assert output.is_file()
        assert output.stat().st_size > 0
