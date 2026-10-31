"""The released run records regenerate the released tables, the figures and the synthetic tensors."""
import pytest

from src import paths

pytestmark = pytest.mark.skipif(not (paths.RECORDS / "mu" / "runs.csv").exists(),
                                reason="run records not present (results/revision)")


def test_tables_match_the_released_tables(tmp_path):
    from src.analysis import supplementary, tables

    argv = ["--records", str(paths.RECORDS), "--tables-dir", str(tmp_path)]
    tables.main(argv)
    supplementary.main(argv)
    released = sorted(p for p in paths.TABLES.iterdir() if p.is_file())
    assert released
    for f in released:
        assert (tmp_path / f.name).read_bytes() == f.read_bytes(), f"{f.name} differs from the released table"


def test_synthetic_tensors_match_the_records():
    from src.data import export

    export.main(["--records", str(paths.RECORDS)])      # exits with an error on any mismatch


def test_figures_build(tmp_path):
    from src.analysis import figures

    figures.main(["--records", str(paths.RECORDS), "--tables-dir", str(paths.TABLES), "--fig-dir", str(tmp_path)])
    names = {p.name for p in tmp_path.glob("*.png")}
    assert {"FigR_rhat_distribution.png", "FigR_covid_convergence.png", "FigR_tradeoff.png", "FigR_scaling.png",
            "FigS1_covid_trajectories.png"} <= names
