import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(name):
    script = ROOT / "experiments" / f"{name}.py"
    out_dir = ROOT / "results" / name
    res = subprocess.run(
        [sys.executable, str(script), "--quick", "--out-dir", str(out_dir)],
        capture_output=True, text=True, timeout=300,
    )
    assert res.returncode == 0, res.stderr
    return out_dir


def test_package_imports():
    import trees
    assert trees.__version__ == "0.2.0"


def test_exp_4_1_runs_quick():
    out = _run("exp_4_1_pvalue_approximation")
    for f in ("figure1.pdf", "figure2.pdf", "table1.csv"):
        assert (out / f).exists(), f


def test_exp_4_2_runs_quick():
    out = _run("exp_4_2_neufeldt_pruning")
    for f in ("figure3.pdf", "figure4.pdf", "figure5.pdf", "figure6.pdf"):
        assert (out / f).exists(), f


def test_exp_4_2_2_runs_quick():
    out = _run("exp_4_2_2_cv_randomness")
    for f in ("figure8.pdf", "table3.csv"):
        assert (out / f).exists(), f


def test_exp_4_3_runs_quick():
    out = _run("exp_4_3_real_data")
    for f in ("table4.csv", "table5.csv", "raw.csv"):
        assert (out / f).exists(), f
