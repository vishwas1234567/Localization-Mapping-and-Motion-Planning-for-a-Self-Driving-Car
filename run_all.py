"""Run every experiment, regenerate tables.tex and the unit tests.   Usage: python run_all.py [--quick-tests-only]"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = ["exp1_vio", "exp2_slam", "exp3_hdmap", "exp4_behavior", "exp5_planning",
         "exp6_hybrid_sweep", "exp7_lattice_failures", "exp8_sensitivity", "exp9_closed_loop"]


def run(cmd):
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    run([sys.executable, "-m", "pytest", "-q", "tests"])
    if "--quick-tests-only" not in sys.argv:
        for s in STEPS:                       # long: E8 (20 seeds) and E6 dominate; use experiments/expN_*.py to run one
            run([sys.executable, f"experiments/{s}.py"])
        run([sys.executable, "make_tables.py"])
    print("done -> compile main.tex (pdflatex x2)")
