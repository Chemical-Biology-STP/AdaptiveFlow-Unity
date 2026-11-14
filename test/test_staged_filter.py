"""The neural-network filter inside the staged screen: reading the prescreen, filtering the selected collections, the
safeguards (too little data, a model no better than chance, everything removed), and the stage machine with the filter
on. The network itself is tested in test_mlfilter.py."""
import csv
import glob
import gzip
import json
import os
import tarfile

import numpy as np
import pytest

import vs_mlfilter
import vs_staged
from test_mlfilter import synthetic
from test_staged import FakeCluster, plan, write_names  # noqa: F401  (fixtures and helpers)
from test_staged_tranche import SMILES, read_tar, write_prescreen_csv, write_tranche_tables


def pdbqt(form, smiles):
    return (f"REMARK    Compound: {form}\nREMARK    Small molecule (ligand)\nREMARK    SMILES_orig: {smiles}\n"
            f"REMARK    SMILES_current: {smiles}\nROOT\nATOM      1  C   UNL     1       0.000   0.000   0.000  0.00  0.00"
            "    +0.000 C \nENDROOT\nTORSDOF 0\n")


def write_collection(run, tranche, num, items):
    """items: [(form, smiles)] -> dock/collections/<tranche>/<num>.tar.gz"""
    from vs_staged import _write_tar
    _write_tar(os.path.join(run, "dock", "collections", tranche, f"{num}.tar.gz"), num,
               [(f"{form}.pdbqt", pdbqt(form, smi)) for form, smi in items])


def write_training(run, smiles, scores):
    folder = os.path.join(run, "afvs_prescreen", "output-files", "ds1", "csv", "9")
    os.makedirs(folder, exist_ok=True)
    with gzip.open(os.path.join(folder, "0.csv.gz"), "wt", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["ligand", "collection_key", "scenario", "score_average", "score_min", "attr_smi"])
        for i, (smi, score) in enumerate(zip(smiles, scores)):
            writer.writerow([f"T{i:06d}_S0_T0", "A" * 18 + "_0000001", "ds1", score, score, smi])


class TestTrainingData:
    def test_the_prescreen_tables_give_smiles_and_the_best_score_of_each_form(self, tmp_path):
        run = str(tmp_path)
        write_training(run, ["CCO", "CCN"], [-5.0, -6.0])
        write_stage = os.path.join(run, "afvs_prescreen", "output-files", "ds1", "csv", "2")
        os.makedirs(write_stage)
        with gzip.open(os.path.join(write_stage, "0.csv.gz"), "wt", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["ligand", "collection_key", "scenario", "score_average", "score_min", "attr_smi"])
            writer.writerow(["T000000_S0_T0", "k", "ds1", -7.0, -7.0, "CCO"])          # the same form, a better score
            writer.writerow(["T000009_S0_T0", "k", "ds1", "N/A", "N/A", "CCC"])        # no score
            writer.writerow(["T000010_S0_T0", "k", "ds1", -4.0, -4.0, ""])             # no SMILES
        smiles, scores = vs_staged.prescreen_training_data(run)
        assert dict(zip(smiles, scores)) == {"CCO": -7.0, "CCN": -6.0}


class TestFormSmiles:
    def test_the_smiles_of_the_form_is_read_from_the_remarks(self):
        text = pdbqt("M0_S0_T0", "O=C1N[C@H]2CCC[C@@H]1C[NH2+]C2")
        assert vs_staged._form_smiles(text) == "O=C1N[C@H]2CCC[C@@H]1C[NH2+]C2"

    def test_a_file_without_it_gives_nothing(self):
        assert vs_staged._form_smiles("REMARK    Compound: x\nROOT\n") == ""


class FakeModel:
    """A model with known answers: molecules with a phenyl ring score 0.9, the others 0.1, unreadable ones NaN."""
    report = {}

    def predict(self, smiles, generator=None):
        return np.array([np.nan if s == "garbage" else (0.9 if "c1ccccc1" in s else 0.1) for s in smiles], dtype=np.float32)


@pytest.fixture
def fake_model(monkeypatch):
    monkeypatch.setitem(vs_staged._ML, "model", FakeModel())
    monkeypatch.setitem(vs_staged._ML, "generator", None)


class TestFilterCollection:
    def job(self, tmp_path, items, cutoff=0.5):
        write_collection(str(tmp_path), "T" * 18, "0000001", items)
        src = os.path.join(str(tmp_path), "dock", "collections", "T" * 18, "0000001.tar.gz")
        dst = os.path.join(str(tmp_path), "dock_ml", "collections", "T" * 18, "0000001.tar.gz")
        return (src, dst, "0000001", "T" * 18 + "_0000001", cutoff), dst

    def test_forms_below_the_cutoff_are_dropped(self, tmp_path, fake_model):
        job, dst = self.job(tmp_path, [("M0_S0_T0", "CCOc1ccccc1"), ("M1_S0_T0", "CCO"), ("M2_S0_T0", "CCNc1ccccc1")])
        key, n_in, n_kept, rows = vs_staged._filter_collection(job)
        assert (n_in, n_kept) == (3, 2) and key.endswith("_0000001")
        assert [(f, keep) for f, _p, keep in rows] == [("M0_S0_T0", True), ("M1_S0_T0", False), ("M2_S0_T0", True)]
        members, names = read_tar(dst)
        assert sorted(os.path.basename(n) for n in members) == ["M0_S0_T0.pdbqt", "M2_S0_T0.pdbqt"]
        assert names[0] == "0000001" and all(n.startswith("0000001/") for n in names[1:])

    def test_a_form_that_cannot_be_judged_is_kept(self, tmp_path, fake_model):
        job, dst = self.job(tmp_path, [("M0_S0_T0", "garbage"), ("M1_S0_T0", "CCO")])
        _key, _n, n_kept, rows = vs_staged._filter_collection(job)
        assert n_kept == 1 and rows[0][2] is True and rows[0][1] != rows[0][1]       # NaN

    def test_a_form_without_smiles_is_kept(self, tmp_path, fake_model):
        from vs_staged import _write_tar
        path = os.path.join(str(tmp_path), "src.tar.gz")
        _write_tar(path, "0000001", [("M0_S0_T0.pdbqt", "REMARK    Compound: M0_S0_T0\nROOT\nENDROOT\n")])
        dst = os.path.join(str(tmp_path), "dst.tar.gz")
        assert vs_staged._filter_collection((path, dst, "0000001", "k", 0.5))[2] == 1

    def test_a_collection_with_nothing_left_writes_no_file(self, tmp_path, fake_model):
        job, dst = self.job(tmp_path, [("M0_S0_T0", "CCO")])
        assert vs_staged._filter_collection(job)[2] == 0 and not os.path.exists(dst)

    def test_the_cutoff_decides(self, tmp_path, fake_model):
        job, _dst = self.job(tmp_path, [("M0_S0_T0", "CCOc1ccccc1"), ("M1_S0_T0", "CCO")], cutoff=0.05)
        assert vs_staged._filter_collection(job)[2] == 2

    def test_the_text_of_a_form_is_not_changed(self, tmp_path, fake_model):
        job, dst = self.job(tmp_path, [("M0_S0_T0", "CCOc1ccccc1")])
        vs_staged._filter_collection(job)
        members, _ = read_tar(dst)
        assert members["0000001/M0_S0_T0.pdbqt"] == pdbqt("M0_S0_T0", "CCOc1ccccc1")


def selected_run(tmp_path, n_per_collection=60, collections=3, seed=21, plan_extra=None):
    """A run folder after the prescreen and the choice of tranches: forms with SMILES in collections, dock/todo.all,
    and the prescreen's training data (made-up problem: a phenyl ring docks better)."""
    run = str(tmp_path / "run")
    for sub in ("prep", "results", "dock"):
        os.makedirs(os.path.join(run, sub))
    smiles, scores, phenyl = synthetic(n_per_collection * collections, seed=seed)
    tranche = "AAAAAAAAAAAAAAAAAA"
    todo = []
    for c in range(collections):
        items = [(f"M{c * n_per_collection + i:09d}_S0_T0", smiles[c * n_per_collection + i]) for i in range(n_per_collection)]
        write_collection(run, tranche, f"{c + 1:07d}", items)
        todo.append(f"{tranche}_{c + 1:07d} {n_per_collection}")
    open(os.path.join(run, "dock", "todo.all"), "w").write("\n".join(todo) + "\n")
    train_smiles, train_scores, _ = synthetic(500, seed=seed + 1)
    write_training(run, train_smiles, train_scores)
    state = {"plan": {"strategy": "tranche", "ml_filter": True, "ml_cutoff": 0.5, **(plan_extra or {})},
             "counts": {"molecules": 180}, "events": []}
    return run, state, np.array(phenyl)


class TestApplyFilter:
    def test_the_filter_keeps_what_the_model_likes_and_writes_a_new_todo(self, tmp_path):
        run, state, phenyl = selected_run(tmp_path)
        result = vs_staged.apply_ml_filter(run, state)
        assert result["forms"] == 180 and 0.15 * 180 < result["kept"] < 0.6 * 180
        todo = [l.split() for l in open(os.path.join(run, "dock_ml", "todo.all")).read().split("\n") if l]
        assert sum(int(n) for _k, n in todo) == result["kept"]
        kept_forms = []
        for key, count in todo:
            tranche, num = key.split("_")
            members, _ = read_tar(os.path.join(run, "dock_ml", "collections", tranche, f"{num}.tar.gz"))
            assert len(members) == int(count)
            kept_forms += [os.path.basename(n)[:-6] for n in members]
        index = {f"M{i:09d}_S0_T0": i for i in range(180)}
        kept = np.array([index[f] for f in kept_forms])
        assert phenyl[kept].mean() > 0.8                                    # what is docked is mostly what docks well
        assert phenyl[kept].sum() > 0.8 * phenyl.sum()                      # and most of what docks well is docked
        assert state["counts"]["ml_pool_forms"] == 180 and state["counts"]["ml_kept_forms"] == result["kept"]
        assert state["counts"]["ml_auc"] > 0.9

    def test_the_tables_say_what_was_decided(self, tmp_path):
        run, state, _ = selected_run(tmp_path)
        result = vs_staged.apply_ml_filter(run, state)
        rows = list(csv.DictReader(open(os.path.join(run, "results", "ml_filter.csv"))))
        assert len(rows) == 180 and {r["tranche"] for r in rows} == {"AAAAAAAAAAAAAAAAAA"}
        assert sum(r["kept"] == "yes" for r in rows) == result["kept"]
        assert all(0 <= float(r["probability"]) <= 1 for r in rows)
        report = json.load(open(os.path.join(run, "results", "ml_filter.json")))
        assert report["used"] is True and report["auc"] > 0.9 and report["kept_forms"] == result["kept"]
        assert os.path.isfile(os.path.join(run, "prep", "ml_model.npz"))

    def test_a_higher_cutoff_keeps_fewer(self, tmp_path):
        a, state_a, _ = selected_run(tmp_path / "a", plan_extra={"ml_cutoff": 0.1})
        b, state_b, _ = selected_run(tmp_path / "b", plan_extra={"ml_cutoff": 0.9})
        assert vs_staged.apply_ml_filter(a, state_a)["kept"] >= vs_staged.apply_ml_filter(b, state_b)["kept"]

    def test_too_little_prescreen_data_means_no_filter_and_says_why(self, tmp_path):
        run, state, _ = selected_run(tmp_path)
        write_training(run, ["CCO", "CCN", "CCC"], [-5.0, -6.0, -7.0])
        for f in glob.glob(os.path.join(run, "afvs_prescreen", "output-files", "ds1", "csv", "1", "*")):
            pass
        assert vs_staged.apply_ml_filter(run, state) is None
        assert "at least 200" in state["counts"]["ml_skipped"]
        assert not os.path.exists(os.path.join(run, "dock_ml"))
        assert json.load(open(os.path.join(run, "results", "ml_filter.json")))["used"] is False
        assert "not used" in state["events"][-1]["text"]

    def test_a_model_that_is_no_better_than_chance_is_not_used(self, tmp_path):
        run, state, _ = selected_run(tmp_path)
        smiles, _s, _p = synthetic(500, seed=33)
        write_training(run, smiles, np.random.default_rng(5).normal(-7, 1, 500))
        assert vs_staged.apply_ml_filter(run, state) is None
        assert "hardly better than chance" in state["counts"]["ml_skipped"]
        assert not os.path.exists(os.path.join(run, "dock_ml"))
        assert json.load(open(os.path.join(run, "results", "ml_filter.json")))["auc"] is not None

    def test_a_filter_that_removes_everything_stops_the_run(self, tmp_path, monkeypatch):
        run, state, _ = selected_run(tmp_path)

        class Nothing(vs_mlfilter.Model):
            def predict(self, smiles, generator=None):
                return np.zeros(len(smiles), dtype=np.float32)

        real = vs_mlfilter.train

        def train(*args, **kwargs):
            model = real(*args, **kwargs)
            return Nothing(model.network, model.threshold, model.report)

        monkeypatch.setattr(vs_mlfilter, "train", train)
        monkeypatch.setattr(vs_staged.os, "cpu_count", lambda: 1)           # in this process, where the patch works
        monkeypatch.setattr(vs_staged, "_ml_init", lambda path: vs_staged._ML.update(
            model=Nothing.load(path), generator=vs_mlfilter._generator()))
        with pytest.raises(vs_staged.StagedError, match="removed every form"):
            vs_staged.apply_ml_filter(run, state)

    def test_several_processes_give_the_same_result_as_one(self, tmp_path, monkeypatch):
        a, state_a, _ = selected_run(tmp_path / "a")
        b, state_b, _ = selected_run(tmp_path / "b")
        many = vs_staged.apply_ml_filter(a, state_a)
        monkeypatch.setattr(vs_staged.os, "cpu_count", lambda: 1)
        one = vs_staged.apply_ml_filter(b, state_b)
        assert many == one
        assert open(os.path.join(a, "results", "ml_filter.csv")).read() == open(os.path.join(b, "results", "ml_filter.csv")).read()
        assert open(os.path.join(a, "dock_ml", "todo.all")).read() == open(os.path.join(b, "dock_ml", "todo.all")).read()

    def test_running_it_twice_replaces_the_first_result(self, tmp_path):
        run, state, _ = selected_run(tmp_path)
        first = vs_staged.apply_ml_filter(run, state)
        second = vs_staged.apply_ml_filter(run, state)
        assert first == second
        assert len(glob.glob(os.path.join(run, "dock_ml", "collections", "*", "*.tar.gz"))) == 3


@pytest.fixture
def filter_run(tmp_path, monkeypatch, plan):
    """A tranche run with the filter on, started with the fake cluster (9 molecules in 3 collections)."""
    (tmp_path / "lib.smi").write_text("".join(f"{s} m{i}\n" for i, s in enumerate(SMILES)))
    monkeypatch.setattr(vs_staged.write_collections, "__defaults__", (3,))
    plan.update(strategy="tranche", representatives=1, primary_size=5, ml_filter=True, ml_cutoff=0.5)
    fake = FakeCluster(monkeypatch, plan["run_dir"])
    fake.workunits = {"prepare": 2, "dock": 2}
    state = vs_staged.start(plan, submit=True)
    return fake, state, plan["run_dir"]


def finish_aflp_with_smiles(run):
    from test_staged import make_aflp_tar  # noqa: F401
    names = vs_staged.read_names(run)
    todo = [l.split()[0] for l in open(os.path.join(run, "aflp", "tools", "templates", "todo.all")).read().split("\n") if l]
    ids = sorted(names)
    for number, key in enumerate(todo):
        meta, tranche, coll = key.split("_")
        from vs_staged import _write_tar
        members = []
        for ident in ids[number * 3: number * 3 + 3]:
            for s in range(2):
                members.append((f"{ident}_S{s}_T0.pdbqt", pdbqt(f"{ident}_S{s}_T0", names[ident][1])))
        _write_tar(os.path.join(run, "aflp", "workflow", "complete", "pdbqt", meta, tranche, f"{coll}.tar.gz"), coll, members)


class TestStagesWithTheFilter:
    def test_the_plan_and_the_estimate_know_about_the_filter(self, filter_run):
        _fake, state, _run = filter_run
        assert state["plan"]["ml_filter"] is True and state["plan"]["ml_cutoff"] == 0.5
        assert state["sizes"]["ml_filter"] is True
        plain = vs_staged.plan_sizes(9, "qvina", 16, 1, 1, 64, "tranche", 5, 3)
        assert state["sizes"]["primary_forms"] < plain["primary_forms"]

    def drive_to_the_primary_screen(self, fake, run, monkeypatch, training):
        vs_staged.advance(run)
        finish_aflp_with_smiles(run)
        vs_staged.advance(run)                                          # tranche library, prescreen starts
        vs_staged.advance(run)                                          # prescreen work unit 2
        listed = []
        for path in glob.glob(os.path.join(run, "prescreen", "collections", "SFL", "*.tar.gz")):
            members, _ = read_tar(path)
            listed += [l.split(",") for l in members[os.path.basename(path)[:-7] + "/.listing"].splitlines()]
        write_prescreen_csv(run, [(form, key, -5.0 - n) for n, (key, form, _i) in enumerate(listed)])
        smiles, scores = training
        write_training(run, smiles, scores)
        vs_staged.advance(run)                                          # select, filter, primary screen starts
        return vs_staged.load_state(run)

    def test_the_primary_screen_docks_what_the_filter_kept(self, filter_run, monkeypatch):
        fake, _state, run = filter_run
        smiles, scores, _ = synthetic(400, seed=40)
        state = self.drive_to_the_primary_screen(fake, run, monkeypatch, (smiles, scores))
        c = state["counts"]
        assert state["stage"] == "dock" and "ml_kept_forms" in c and 1 <= c["ml_kept_forms"] <= c["ml_pool_forms"]
        assert c["primary_forms"] == c["ml_pool_forms"] >= c["ml_kept_forms"]
        ctrl = open(os.path.join(run, "afvs", "tools", "templates", "all.ctrl")).read().splitlines()
        assert f"collection_folder={run}/dock_ml/collections" in ctrl and "prescreen_mode=0" in ctrl
        todo = open(os.path.join(run, "afvs", "tools", "templates", "todo.all")).read()
        assert todo == open(os.path.join(run, "dock_ml", "todo.all")).read()
        assert "ML filter" in vs_staged.status_text(run) and "keeps" in vs_staged.status_text(run)

    def test_without_enough_training_data_the_whole_selection_is_docked(self, filter_run, monkeypatch):
        fake, _state, run = filter_run
        state = self.drive_to_the_primary_screen(fake, run, monkeypatch, (["CCO", "CCN"], [-5.0, -6.0]))
        assert state["stage"] == "dock" and "ml_skipped" in state["counts"]
        ctrl = open(os.path.join(run, "afvs", "tools", "templates", "all.ctrl")).read().splitlines()
        assert f"collection_folder={run}/dock/collections" in ctrl
        assert "ML filter    not used" in vs_staged.status_text(run)

    def test_the_run_finishes_and_ranks_with_the_filter_on(self, filter_run, monkeypatch):
        fake, _state, run = filter_run
        smiles, scores, _ = synthetic(400, seed=40)
        self.drive_to_the_primary_screen(fake, run, monkeypatch, (smiles, scores))
        vs_staged.advance(run)                                          # primary work unit 2
        todo = open(os.path.join(run, "dock_ml", "todo.all")).read().split("\n")
        forms = []
        for line in todo:
            if line.strip():
                key = line.split()[0]
                tranche, num = key.split("_")
                members, _ = read_tar(os.path.join(run, "dock_ml", "collections", tranche, f"{num}.tar.gz"))
                forms += [os.path.basename(n)[:-6] for n in members]
        from test_staged_tranche import write_stage_csv
        write_stage_csv(run, "afvs", [(f, -8.0) for f in forms])
        vs_staged.advance(run)
        state = vs_staged.load_state(run)
        assert state["status"] == "done" and state["counts"]["docked"] >= len(set(f.split("_")[0] for f in forms))


class TestCheckPlanFilter:
    @pytest.fixture(autouse=True)
    def limit(self, monkeypatch):
        monkeypatch.setattr(vs_staged, "cluster_cpu_limit", lambda partition="ncpu": 3000)

    @pytest.mark.parametrize("cutoff", [0.04, 0.96, "x", True])
    def test_the_cutoff_range(self, plan, cutoff):
        plan.update(strategy="tranche", ml_filter=True, ml_cutoff=cutoff)
        with pytest.raises(vs_staged.StagedError, match="ml_cutoff"):
            vs_staged.check_plan(plan)

    def test_the_filter_must_be_true_or_false(self, plan):
        plan.update(strategy="tranche", ml_filter="yes")
        with pytest.raises(vs_staged.StagedError, match="ml_filter"):
            vs_staged.check_plan(plan)

    def test_good_values_pass(self, plan):
        plan.update(strategy="tranche", ml_filter=True, ml_cutoff=0.05)
        vs_staged.check_plan(plan)


class TestSizesWithTheFilter:
    def test_the_filter_lowers_the_expected_docking(self):
        without = vs_staged.plan_sizes(100000, "qvina", 16, 1, 1, 500, "tranche", 20000, 1500)
        with_filter = vs_staged.plan_sizes(100000, "qvina", 16, 1, 1, 500, "tranche", 20000, 1500, True)
        assert with_filter["primary_forms"] == 10000 and without["primary_forms"] == 20000
        assert with_filter["cpu_hours"] < without["cpu_hours"]

    def test_the_filter_does_nothing_without_the_tranche_strategy(self):
        a = vs_staged.plan_sizes(1000, "qvina", 16, 1, 1, 64)
        b = vs_staged.plan_sizes(1000, "qvina", 16, 1, 1, 64, "all", None, None, True)
        assert a["dock_forms"] == b["dock_forms"]

    def test_the_advance_job_gets_memory_for_the_filter(self, tmp_path, monkeypatch):
        class Done:
            returncode, stdout, stderr = 0, "1\n", ""

        monkeypatch.setattr(vs_staged.subprocess, "run", lambda *a, **k: Done())
        monkeypatch.setattr(vs_staged.vs_slurm, "mail_address", lambda: None)
        state = {"plan": {"partition": "ncpu", "strategy": "tranche", "ml_filter": True}, "counts": {"molecules": 2_000_000},
                 "sizes": {"n_forms": 2_000_000}}
        vs_staged.submit_advance(str(tmp_path), state, None)
        text = open(tmp_path / "logs" / "advance.sh").read()
        assert "#SBATCH --mem=19G" in text                         # ceil(2.6 + 6.0 + 6) + 4
