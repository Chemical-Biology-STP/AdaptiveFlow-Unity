"""A real staged run on the cluster: AFLP and AFVS through SLURM, with the `advance` jobs chaining the stages.

Marked `cluster`: it needs sbatch and a folder that the compute nodes can read (not /tmp, which is local to a node).
It took 86 seconds with a free queue; a busy queue makes it longer. Run it with `pixi run test-cluster`; set VS_CLUSTER_TEST_DIR to a folder on shared storage; the test is skipped without it.
"""
import csv
import os
import shutil
import time

import pytest

import vs_staged

pytestmark = pytest.mark.cluster

SMILES = ["CC(=O)Nc1ccc(O)cc1 paracetamol", "CC(C)Cc1ccc(cc1)C(C)C(=O)O ibuprofen", "c1ccc2[nH]ccc2c1 indole",
          "CCN(CC)CCNC(=O)c1ccc(N)cc1 procainamide", "Cn1cnc2c1c(=O)n(C)c(=O)n2C caffeine", "OC(=O)c1ccccc1O salicylic",
          "CC(=O)Oc1ccccc1C(=O)O aspirin", "c1ccc(cc1)C(=O)N benzamide", "CCOc1ccc(N)cc1 phenetidine",
          "Nc1ccc(cc1)S(N)(=O)=O sulfanilamide"]


@pytest.fixture
def work_dir():
    root = os.environ.get("VS_CLUSTER_TEST_DIR")
    if shutil.which("sbatch") is None:
        pytest.skip("SLURM (sbatch) is not available here")
    if not root:
        pytest.skip("set VS_CLUSTER_TEST_DIR to a folder on shared storage (not your home folder, not /tmp)")
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, f"run_{int(time.time())}")
    yield path
    shutil.rmtree(path, ignore_errors=True)


def wait_for(run, minutes=40):
    deadline = time.time() + minutes * 60
    while time.time() < deadline:
        state = vs_staged.load_state(run)
        if state["status"] in ("done", "failed"):
            return state
        time.sleep(20)
    raise AssertionError("the run did not finish in time:\n" + vs_staged.status_text(run))


def test_a_small_library_goes_through_every_stage_and_is_ranked(work_dir, pocket, box_centre):
    os.makedirs(work_dir)
    library = os.path.join(work_dir, "lib.smi")
    with open(library, "w") as fh:
        fh.write("\n".join(SMILES) + "\n")
    run = os.path.join(work_dir, "run")
    plan = {"run_dir": run, "receptor": pocket, "add_hydrogens": False, "center": box_centre, "box_size": 22.0,
            "ligands": library, "program": "qvina", "exhaustiveness": 4, "seed": 1, "ph": 7.4,
            "max_stereoisomers": 1, "max_tautomers": 1, "cpus": 32, "partition": "ncpu", "job_name": "vs_test"}
    vs_staged.start(plan, submit=True)
    state = wait_for(run)
    assert state["status"] == "done", state.get("reason") or vs_staged.status_text(run)
    rows = list(csv.DictReader(open(os.path.join(run, "results", "ranking.csv"))))
    assert [r["name"] for r in rows if r["status"] == "docked"] and len(rows) == len(SMILES)
    docked = [r for r in rows if r["status"] == "docked"]
    assert len(docked) >= len(SMILES) - 1                                  # AFLP can fail on a molecule; it must say so
    scores = [float(r["best_score"]) for r in docked]
    assert scores == sorted(scores) and all(-15 < s < 0 for s in scores)
    assert {r["name"] for r in rows} == {line.split()[1] for line in SMILES}
    assert all(r["n_forms"] == "1" for r in docked)                        # one stereoisomer and one tautomer asked for
    assert "paracetamol" in vs_staged.results_text(run, top=20) or len(docked) < len(SMILES)


def test_the_tranche_strategy_goes_through_prescreen_and_primary_screen(work_dir, pocket, box_centre):
    os.makedirs(work_dir)
    library = os.path.join(work_dir, "lib.smi")
    with open(library, "w") as fh:
        fh.write("\n".join(SMILES) + "\n")
    run = os.path.join(work_dir, "run")
    plan = {"run_dir": run, "receptor": pocket, "add_hydrogens": False, "center": box_centre, "box_size": 22.0,
            "ligands": library, "program": "qvina", "exhaustiveness": 4, "seed": 1, "ph": 7.4,
            "max_stereoisomers": 1, "max_tautomers": 1, "cpus": 32, "partition": "ncpu", "job_name": "vs_test_t",
            "strategy": "tranche", "representatives": 1, "primary_size": 4}
    vs_staged.start(plan, submit=True)
    state = wait_for(run, minutes=60)
    assert state["status"] == "done", state.get("reason") or vs_staged.status_text(run)
    counts = state["counts"]
    assert counts["tranches"] >= 2 and counts["tranches_selected"] >= 1 and counts["primary_forms"] >= 4
    rows = list(csv.DictReader(open(os.path.join(run, "results", "ranking.csv"))))
    assert len(rows) == len(SMILES) and "tranche" in rows[0]
    docked = [r for r in rows if r["status"] == "docked"]
    assert len(docked) >= counts["tranches"] - 1                            # every tranche had a representative docked
    assert {r["status"] for r in rows} <= {"docked", "not_docked", "failed"}
    scores = [float(r["best_score"]) for r in docked]
    assert scores == sorted(scores) and all(-15 < s < 0 for s in scores)
    tranche_rows = list(csv.DictReader(open(os.path.join(run, "results", "tranche_scores.csv"))))
    assert len(tranche_rows) == counts["tranches"] and any(r["selected"] == "yes" for r in tranche_rows)


def test_the_filter_on_a_library_too_small_to_train_it_docks_everything_selected(work_dir, pocket, box_centre):
    """Ten molecules give far fewer than the 200 prescreen scores the filter needs: it must say so and carry on."""
    os.makedirs(work_dir)
    library = os.path.join(work_dir, "lib.smi")
    with open(library, "w") as fh:
        fh.write("\n".join(SMILES) + "\n")
    run = os.path.join(work_dir, "run")
    plan = {"run_dir": run, "receptor": pocket, "add_hydrogens": False, "center": box_centre, "box_size": 22.0,
            "ligands": library, "program": "qvina", "exhaustiveness": 4, "seed": 1, "ph": 7.4,
            "max_stereoisomers": 1, "max_tautomers": 1, "cpus": 32, "partition": "ncpu", "job_name": "vs_test_ml",
            "strategy": "tranche", "representatives": 1, "primary_size": 4, "ml_filter": True}
    vs_staged.start(plan, submit=True)
    state = wait_for(run, minutes=60)
    assert state["status"] == "done", state.get("reason") or vs_staged.status_text(run)
    assert "at least 200" in state["counts"]["ml_skipped"] and "ml_kept_forms" not in state["counts"]
    assert "ML filter    not used" in vs_staged.status_text(run)
    assert not os.path.exists(os.path.join(run, "dock_ml"))
    docked = [r for r in csv.DictReader(open(os.path.join(run, "results", "ranking.csv"))) if r["status"] == "docked"]
    assert len(docked) >= state["counts"]["tranches"] - 1
