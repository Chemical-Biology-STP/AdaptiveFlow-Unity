"""Staged screening (vs_staged): sizes, the library cut into collections, the configuration written for AFLP and AFVS,
the repacking, the ranking, and the stage logic with stand-ins for SLURM and the upstream scripts. Nothing here needs
the cluster. The run on the cluster is tested by test_staged_cluster.py (marked `cluster`)."""
import csv
import gzip
import io
import json
import os
import tarfile

import pytest

import vs_dock
import vs_staged


# --------------------------------------------------------------------------
# Sizes and cost
# --------------------------------------------------------------------------


class TestSizes:
    def test_one_stereoisomer_and_one_tautomer_is_exactly_one_form(self):
        assert vs_staged.forms_per_molecule(1, 1) == 1.0

    def test_the_measured_default_of_five_and_five_gives_the_measured_4_57(self):
        assert vs_staged.forms_per_molecule(5, 5) == pytest.approx(4.57, abs=0.01)

    def test_more_forms_never_cost_less_and_never_exceed_the_measured_figure(self):
        values = [vs_staged.forms_per_molecule(a, b) for a in (1, 2, 3, 5, 10) for b in (1, 2, 3, 5, 10)]
        assert all(1.0 <= v <= vs_staged.FORMS_MEASURED for v in values)
        assert vs_staged.forms_per_molecule(2, 1) <= vs_staged.forms_per_molecule(3, 1) <= vs_staged.forms_per_molecule(3, 2)

    def test_the_cost_is_forms_times_the_seconds_per_form(self):
        sizes = vs_staged.plan_sizes(1000, "qvina", 16, 1, 1, 500)
        assert sizes["n_forms"] == 1000
        assert sizes["seconds_per_form"] == vs_dock.PROGRAMS["qvina"]["seconds_per_ligand"] * 16 / 8
        assert sizes["dock_cpu_hours"] == pytest.approx(1000 * sizes["seconds_per_form"] / 3600)
        assert sizes["cpu_hours"] > sizes["dock_cpu_hours"]            # preparation is counted too

    def test_a_subjob_is_sized_to_take_about_the_target_time(self):
        sizes = vs_staged.plan_sizes(10 ** 6, "qvina", 16, 1, 1, 500)
        minutes = sizes["dockings_per_subjob"] * sizes["seconds_per_form"] / sizes["subjob_cpus"] / 60
        assert minutes == pytest.approx(vs_staged.SUBJOB_MINUTES, rel=0.05)

    def test_cpus_are_used_in_whole_subjobs(self):
        sizes = vs_staged.plan_sizes(10 ** 6, "qvina", 16, 1, 1, 500)
        assert sizes["subjob_cpus"] == 32 and sizes["subjobs_at_once"] == 500 // 32 == 15

    def test_fewer_cpus_than_a_subjob_make_a_smaller_subjob(self):
        sizes = vs_staged.plan_sizes(1000, "qvina", 16, 1, 1, 8)
        assert sizes["subjob_cpus"] == 8 and sizes["subjobs_at_once"] == 1

    def test_a_small_library_never_asks_for_more_jobs_than_it_has(self):
        sizes = vs_staged.plan_sizes(200, "qvina", 16, 1, 1, 3000)
        assert sizes["dock_subjobs"] == 1 and sizes["subjobs_at_once"] == 1

    def test_more_cpus_shorten_the_wall_time_but_not_the_cpu_hours(self):
        few = vs_staged.plan_sizes(10 ** 6, "qvina", 16, 1, 1, 500)
        many = vs_staged.plan_sizes(10 ** 6, "qvina", 16, 1, 1, 3000)
        assert many["cpu_hours"] == pytest.approx(few["cpu_hours"])
        assert many["wall_hours"] < few["wall_hours"] / 4

    def test_the_time_limit_has_a_floor_and_a_ceiling(self):
        assert vs_staged.minutes_for_subjob(1, 32) == vs_staged.MIN_TIME_MINUTES
        assert vs_staged.minutes_for_subjob(10 ** 12, 1) == vs_slurm_max()

    def test_slurm_time_text(self):
        assert vs_staged.vs_slurm_time(75) == "01:15:00"
        assert vs_staged.vs_slurm_time(1500) == "1-01:00:00"

    def test_collection_names_are_letters_and_unique(self):
        names = [vs_staged.base26(n, 2) for n in range(5000)]
        assert len(set(names)) == 5000 and all(n.isalpha() and n.isupper() for n in names)
        assert vs_staged.base26(0, 2) == "AA" and vs_staged.base26(26, 2) == "BA"


def vs_slurm_max():
    import vs_slurm
    return vs_slurm.MAX_TIME_MINUTES


class TestClusterLimit:
    def test_the_limit_is_read_from_the_cluster_qos(self, monkeypatch):
        outputs = iter(["QoS=ncpu SomethingElse=1", "cpu=3000\n"])

        class Result:
            def __init__(self, stdout):
                self.stdout = stdout

        monkeypatch.setattr(vs_staged.subprocess, "run", lambda *a, **k: Result(next(outputs)))
        assert vs_staged.cluster_cpu_limit("ncpu") == 3000

    def test_without_slurm_the_documented_limit_is_used(self, monkeypatch):
        def broken(*a, **k):
            raise FileNotFoundError("scontrol")
        monkeypatch.setattr(vs_staged.subprocess, "run", broken)
        assert vs_staged.cluster_cpu_limit("ncpu") == vs_staged.FALLBACK_MAX_CPUS == 3000


# --------------------------------------------------------------------------
# Library -> collections
# --------------------------------------------------------------------------


@pytest.fixture
def smiles_file(tmp_path):
    path = tmp_path / "lib.smi"
    path.write_text("smiles name\nCCO ethanol\nc1ccccc1 benzene\nnot_a_smiles bad\nCC(=O)O acetic acid_x\n"
                    "CCN ethylamine\nCCC propane\n")
    return str(path)


class TestCollections:
    def test_molecules_are_cut_into_collections_with_safe_ids(self, tmp_path, smiles_file):
        n, todo, unreadable = vs_staged.write_collections(smiles_file, str(tmp_path / "c"), str(tmp_path / "names.tsv"),
                                                          per_collection=2)
        assert n == 5 and len(todo) == 3
        assert [line.split()[1] for line in todo] == ["2", "2", "1"]
        assert [u[1] for u in unreadable] == ["bad"]
        folder = tmp_path / "c" / "AA" / "AAA"
        rows = [l.split("\t") for l in gzip.open(folder / "AA.txt.gz", "rt").read().splitlines()]
        assert [r[1] for r in rows] == ["M000000000", "M000000001"]

    def test_the_original_names_are_kept_in_a_table(self, tmp_path, smiles_file):
        vs_staged.write_collections(smiles_file, str(tmp_path / "c"), str(tmp_path / "names.tsv"), per_collection=2)
        rows = list(csv.DictReader(open(tmp_path / "names.tsv"), delimiter="\t"))
        assert [r["name"] for r in rows] == ["ethanol", "benzene", "acetic", "ethylamine", "propane"]
        assert [r["id"] for r in rows] == [f"M{i:09d}" for i in range(5)]

    def test_collection_keys_have_the_three_parts_aflp_expects(self, tmp_path, smiles_file):
        _, todo, _ = vs_staged.write_collections(smiles_file, str(tmp_path / "c"), str(tmp_path / "n.tsv"),
                                                 per_collection=2)
        assert all(len(line.split()[0].split("_")) == 3 for line in todo)

    def test_a_thousand_collections_share_a_folder_at_most(self, tmp_path):
        path = tmp_path / "big.smi"
        path.write_text("".join(f"C{'C' * (i % 7)}O m{i}\n" for i in range(30)))
        _, todo, _ = vs_staged.write_collections(str(path), str(tmp_path / "c"), str(tmp_path / "n.tsv"),
                                                 per_collection=1)
        assert len(todo) == 30 and len({line.split("_")[1] for line in todo}) == 1

    def test_every_molecule_gets_its_own_id_in_input_order(self, tmp_path):
        path = tmp_path / "many.smi"
        path.write_text("".join(f"C{'C' * (i % 9)}O mol_{i}\n" for i in range(40)))
        vs_staged.write_collections(str(path), str(tmp_path / "c"), str(tmp_path / "n.tsv"), per_collection=7)
        rows = list(csv.DictReader(open(tmp_path / "n.tsv"), delimiter="\t"))
        assert [r["id"] for r in rows] == [f"M{i:09d}" for i in range(40)]
        assert [r["name"] for r in rows] == [f"mol_{i}" for i in range(40)]

    def test_unusual_names_cannot_reach_a_file_name(self, tmp_path):
        path = tmp_path / "odd.smi"
        path.write_text("CCO ../../etc/passwd\nCCN a;b\n")
        vs_staged.write_collections(str(path), str(tmp_path / "c"), str(tmp_path / "n.tsv"))
        text = gzip.open(tmp_path / "c" / "AA" / "AAA" / "AA.txt.gz", "rt").read()
        assert "passwd" not in text and "M000000000" in text


# --------------------------------------------------------------------------
# Configuration written for AFLP and AFVS
# --------------------------------------------------------------------------


@pytest.fixture
def plan(tmp_path, pocket):
    return {"run_dir": str(tmp_path / "run"), "receptor": pocket, "add_hydrogens": False,
            "center": [15.0, -1.9, -1.2], "box_size": 22.0, "ligands": str(tmp_path / "lib.smi"),
            "program": "qvina", "exhaustiveness": 16, "seed": 7, "ph": 7.4, "max_stereoisomers": 1,
            "max_tautomers": 1, "cpus": 64, "partition": "ncpu", "job_name": "t"}


@pytest.fixture
def run_dir(plan):
    os.makedirs(os.path.join(plan["run_dir"], "input"))
    open(os.path.join(plan["run_dir"], "input", "receptor.pdbqt"), "w").write("ATOM\n")
    return plan["run_dir"]


class TestConfiguration:
    def test_set_ctrl_changes_only_the_named_line(self):
        text = "a=1\nb=2\n# a=9\n"
        assert vs_staged.set_ctrl(text, "a", "5") == "a=5\nb=2\n# a=9\n"

    def test_set_ctrl_stops_when_a_key_has_gone_from_the_template(self):
        with pytest.raises(vs_staged.StagedError, match="no 'missing='"):
            vs_staged.set_ctrl("a=1\n", "missing", "x")

    @pytest.mark.parametrize("tool,name", [("AFLP", "aflp"), ("AFVS", "afvs")])
    def test_the_vendored_job_templates_are_patched_for_this_cluster(self, tool, name):
        text = open(os.path.join(vs_staged.THIRD_PARTY, tool, "tools", "templates", "template1.slurm.sh")).read()
        out = vs_staged.patch_slurm_template(text, "/env/bin", 75)
        assert "--account" not in out and "conda activate" not in out and "module load" not in out
        assert "export PATH=/env/bin:$PATH" in out
        assert "#SBATCH --time=01:15:00" in out and "#SBATCH --mem-per-cpu=2G" in out
        assert "#SBATCH --partition={{slurm_partition}}" in out

    def test_a_template_that_changed_upstream_stops_with_a_clear_message(self):
        with pytest.raises(vs_staged.StagedError, match="was probably updated"):
            vs_staged.patch_slurm_template("#!/bin/bash\n#SBATCH --job-name=x\n", "/env/bin", 60)

    def test_aflp_is_configured_from_the_plan(self, plan, run_dir):
        sizes = vs_staged.plan_sizes(2000, "qvina", 16, 3, 2, plan["cpus"])
        plan.update(max_stereoisomers=3, max_tautomers=2)
        work = vs_staged.configure_aflp(run_dir, plan, sizes, ["AA_AAA_AA 500", "AA_AAA_AB 500"])
        ctrl = open(os.path.join(work, "tools", "templates", "all.ctrl")).read()
        for line in ("batchsystem=slurm", "slurm_partition=ncpu", "slurm_cpus=32", "job_storage_mode=sharedfs",
                     "rdkit_stereoisomer_max_count=3", "rdkit_tautomer_max_count=2", "protonation_pH_value=7.4",
                     "conformation_program_1=rdkit", "conformation_program_2=none", "targetformats=pdbqt",
                     f"collection_folder={run_dir}/prep/collections"):
            assert line in ctrl.splitlines(), line
        assert open(os.path.join(work, "tools", "templates", "todo.all")).read() == "AA_AAA_AA 500\nAA_AAA_AB 500\n"

    def test_afvs_is_configured_from_the_plan(self, plan, run_dir):
        sizes = vs_staged.plan_sizes(2000, "qvina", 16, 1, 1, plan["cpus"])
        work = vs_staged.configure_afvs(run_dir, plan, sizes)
        ctrl = open(os.path.join(work, "tools", "templates", "all.ctrl")).read().splitlines()
        for line in ("batchsystem=slurm", "data_storage_mode=sharedfs", "job_storage_mode=sharedfs",
                     "docking_scenario_methods=qvina02", "docking_scenario_names=ds1", "dynamic_tranche_filtering=0",
                     "prescreen_mode=0", "slurm_cpus=32", f"dockings_per_subjob={sizes['dockings_per_subjob']}",
                     f"collection_folder={run_dir}/dock/collections", "program_timeout=900"):
            assert line in ctrl, line
        config = open(os.path.join(work, "input-files", "ds1", "config.txt")).read()
        assert "center_x=15.000" in config and "size_x=22.000" in config and "exhaustiveness=16" in config
        assert "seed=7" in config

    def test_the_docking_program_is_linked_from_this_tools_executables(self, plan, run_dir):
        sizes = vs_staged.plan_sizes(10, "qvina", 16, 1, 1, 8)
        work = vs_staged.configure_afvs(run_dir, plan, sizes)
        link = os.path.join(work, "tools", "bin", "qvina02")
        assert os.path.islink(link) and os.path.realpath(link) == os.path.realpath(
            os.path.join(vs_dock.EXECUTABLES, "qvina"))

    @pytest.mark.parametrize("program", sorted(vs_staged.PROGRAM_METHODS))
    def test_every_offered_program_has_an_afvs_name_and_an_executable(self, program):
        method = vs_staged.PROGRAM_METHODS[program]
        assert os.access(os.path.join(vs_dock.EXECUTABLES, vs_staged.PROGRAM_BINARIES[method]), os.X_OK)

    def test_the_vendored_upstream_files_are_the_pinned_ones(self):
        text = open(os.path.join(vs_staged.THIRD_PARTY, "VERSIONS.md")).read()
        assert "aadf971" in text and "b201c28" in text
        assert not os.path.exists(os.path.join(vs_staged.THIRD_PARTY, "AFVS", "tools", "bin"))


# --------------------------------------------------------------------------
# Plan checks
# --------------------------------------------------------------------------


class TestCheckPlan:
    @pytest.fixture(autouse=True)
    def limit(self, monkeypatch):
        monkeypatch.setattr(vs_staged, "cluster_cpu_limit", lambda partition="ncpu": 3000)

    def test_a_good_plan_passes(self, plan):
        vs_staged.check_plan(plan)

    def test_a_missing_key_is_named(self, plan):
        del plan["box_size"]
        with pytest.raises(vs_staged.StagedError, match="box_size"):
            vs_staged.check_plan(plan)

    def test_an_unsupported_program_is_refused(self, plan):
        plan["program"] = "gold"
        with pytest.raises(vs_staged.StagedError, match="supports"):
            vs_staged.check_plan(plan)

    @pytest.mark.parametrize("cpus", [0, 3001])
    def test_cpus_outside_the_cluster_limit_are_refused(self, plan, cpus):
        plan["cpus"] = cpus
        with pytest.raises(vs_staged.StagedError, match="3000"):
            vs_staged.check_plan(plan)

    def test_the_limit_itself_is_allowed(self, plan):
        plan["cpus"] = 3000
        vs_staged.check_plan(plan)

    @pytest.mark.parametrize("value", [0, 21])
    def test_forms_per_molecule_are_bounded(self, plan, value):
        plan["max_tautomers"] = value
        with pytest.raises(vs_staged.StagedError, match="max_tautomers"):
            vs_staged.check_plan(plan)


# --------------------------------------------------------------------------
# Repacking and ranking
# --------------------------------------------------------------------------

PDBQT = ("REMARK    Compound: {n}\nCOMPND    Compound: {n}\nROOT\nATOM      1  C   UNL     1       0.000   0.000"
         "   0.000  0.00  0.00    +0.000 C \nENDROOT\nTORSDOF 0\n")


def make_aflp_tar(path, folder, forms):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tarfile.open(path, "w:gz") as tf:
        info = tarfile.TarInfo(folder)
        info.type = tarfile.DIRTYPE
        tf.addfile(info)
        for name in forms:
            data = PDBQT.format(n=name).encode()
            info = tarfile.TarInfo(f"{folder}/{name}.pdbqt")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


class TestRepack:
    def test_the_folder_is_renamed_and_the_compnd_line_removed(self, tmp_path):
        src, dst = str(tmp_path / "in.tar.gz"), str(tmp_path / "out" / "AA.tar.gz")
        make_aflp_tar(src, "AA", ["M000000000_S0_T0", "M000000001_S0_T0"])
        key, count = vs_staged._repack(("AA_AAA_AA", src, dst, "AAAAA_AA", "AA"))
        assert (key, count) == ("AAAAA_AA", 2)
        with tarfile.open(dst) as tf:
            assert tf.getnames()[0] == "AA" and all(n.startswith("AA/") for n in tf.getnames()[1:])
            text = tf.extractfile("AA/M000000000_S0_T0.pdbqt").read().decode()
        assert "COMPND" not in text and "REMARK    Compound: M000000000_S0_T0" in text and "TORSDOF 0" in text

    def test_an_empty_collection_counts_zero(self, tmp_path):
        src, dst = str(tmp_path / "in.tar.gz"), str(tmp_path / "AA.tar.gz")
        make_aflp_tar(src, "AA", [])
        assert vs_staged._repack(("k", src, dst, "AAAAA_AA", "AA"))[1] == 0


def write_afvs_csv(run_dir, rows):
    folder = os.path.join(run_dir, "afvs", "output-files", "ds1", "csv", "1")
    os.makedirs(folder, exist_ok=True)
    with gzip.open(os.path.join(folder, "0.csv.gz"), "wt", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["ligand", "collection_key", "scenario", "score_average", "score_min", "attr_smi",
                         "attr_heavy_atom_count", "score_0"])
        for ligand, score in rows:
            writer.writerow([ligand, "AAAAA_AA", "ds1", score, score, "CCO", "N/A", score])


def write_names(run_dir, names):
    os.makedirs(os.path.join(run_dir, "input"), exist_ok=True)
    with open(os.path.join(run_dir, "input", "names.tsv"), "w") as fh:
        fh.write("id\tname\tsmiles\n")
        for i, name in enumerate(names):
            fh.write(f"M{i:09d}\t{name}\tC{'C' * i}O\n")


class TestRank:
    def test_the_best_form_of_every_molecule_is_kept_and_names_restored(self, tmp_path):
        run = str(tmp_path)
        write_names(run, ["aspirin", "caffeine", "lost"])
        write_afvs_csv(run, [("M000000000_S0_T0", -6.0), ("M000000000_S1_T0", -7.5), ("M000000000_S0_T1", -5.0),
                             ("M000000001_S0_T0", -8.2)])
        state = {"counts": {}}
        docked, failed = vs_staged.rank(run, state)
        assert (docked, failed) == (2, 1)
        rows = list(csv.DictReader(open(tmp_path / "results" / "ranking.csv")))
        assert [r["name"] for r in rows] == ["caffeine", "aspirin", "lost"]
        assert rows[1]["best_score"] == "-7.500" and rows[1]["n_forms"] == "3" and rows[1]["best_form"] == "M000000000_S1_T0"
        assert rows[2]["status"] == "failed" and rows[2]["note"]

    def test_results_csv_keeps_the_input_order(self, tmp_path):
        run = str(tmp_path)
        write_names(run, ["b", "a"])
        write_afvs_csv(run, [("M000000000_S0_T0", -1.0), ("M000000001_S0_T0", -9.0)])
        vs_staged.rank(run, {"counts": {}})
        rows = list(csv.DictReader(open(tmp_path / "results" / "results.csv")))
        assert [r["name"] for r in rows] == ["b", "a"] and [r["index"] for r in rows] == ["0", "1"]

    def test_no_result_tables_is_an_error_not_an_empty_ranking(self, tmp_path):
        write_names(str(tmp_path), ["a"])
        with pytest.raises(vs_staged.StagedError, match="no result tables"):
            vs_staged.rank(str(tmp_path), {"counts": {}})

    def test_the_results_text_lists_the_best_first(self, tmp_path):
        run = str(tmp_path)
        write_names(run, ["b", "a"])
        write_afvs_csv(run, [("M000000000_S0_T0", -1.0), ("M000000001_S0_T0", -9.0)])
        vs_staged.rank(run, {"counts": {}})
        lines = vs_staged.results_text(run, top=5).splitlines()
        assert lines[1].split()[-1] == "a" and lines[2].split()[-1] == "b"

    def test_results_before_the_end_say_so(self, tmp_path):
        with pytest.raises(vs_staged.StagedError, match="no ranking yet"):
            vs_staged.results_text(str(tmp_path))


# --------------------------------------------------------------------------
# The stage logic, with stand-ins for SLURM and the upstream scripts
# --------------------------------------------------------------------------


class FakeCluster:
    """Replaces sbatch, squeue/sacct and the upstream scripts. `states` maps job id -> list of task states."""

    def __init__(self, monkeypatch, run_dir):
        self.run_dir = run_dir
        self.next_id = 1000
        self.states = {}
        self.calls = []
        self.workunits = {"prepare": 2, "dock": 2}
        monkeypatch.setattr(vs_staged, "upstream", self.upstream)
        monkeypatch.setattr(vs_staged, "submitted_job_id", self.submitted_job_id)
        monkeypatch.setattr(vs_staged, "job_states", lambda job_id: self.states.get(int(job_id), ["COMPLETED"]))
        monkeypatch.setattr(vs_staged, "submit_advance", self.submit_advance)
        monkeypatch.setattr(vs_staged, "cluster_cpu_limit", lambda partition="ncpu": 3000)
        monkeypatch.setattr(vs_dock, "prepare_receptor", self.prepare_receptor)

    def prepare_receptor(self, workdir, receptor_pdb, adfr_path=None, add_hydrogens=False):
        os.makedirs(workdir, exist_ok=True)
        path = os.path.join(workdir, "receptor.pdbqt")
        open(path, "w").write("ATOM\n")
        return path

    def upstream(self, tool_dir, script, *args):
        self.calls.append((os.path.basename(tool_dir), script, args))
        if script == "aflp_prepare_workunits.py":
            return f"Generated {self.workunits['prepare']} workunits\n"
        if script == "afvs_prepare_workunits.py":
            return f"Generated {self.workunits['dock']} workunits\n"
        return ""

    def submitted_job_id(self, tool_dir, workunit):
        self.next_id += 1
        return self.next_id

    def submit_advance(self, run_dir, state, after_job, final=False):
        self.next_id += 1
        state["advance_job"] = str(self.next_id)
        self.calls.append(("advance", after_job, final))
        return str(self.next_id)

    def aflp_output(self, todo_keys, forms_each=2):
        for key in todo_keys:
            meta, tranche, coll = key.split("_")
            make_aflp_tar(os.path.join(self.run_dir, "aflp", "workflow", "complete", "pdbqt", meta, tranche,
                                       f"{coll}.tar.gz"), coll,
                          [f"M{i:09d}_S0_T{t}" for i in range(2) for t in range(forms_each)])


@pytest.fixture
def started(tmp_path, monkeypatch, plan):
    """A run folder started with the fake cluster: 4 molecules in 2 collections."""
    (tmp_path / "lib.smi").write_text("CCO a\nCCN b\nCCC c\nCCCC d\n")
    monkeypatch.setattr(vs_staged.write_collections, "__defaults__", (2,))       # two molecules per collection
    fake = FakeCluster(monkeypatch, plan["run_dir"])
    state = vs_staged.start(plan, submit=True)
    return fake, state, plan["run_dir"]


class TestStages:
    def test_starting_prepares_the_folder_and_queues_the_first_work_unit(self, started):
        fake, state, run = started
        assert state["stage"] == "prepare" and state["status"] == "running"
        assert state["counts"]["molecules"] == 4 and state["counts"]["collections"] == 2
        assert state["prepare"]["workunits"] == 2 and state["prepare"]["submitted"] == 1
        assert ("aflp", "aflp_submit_jobs.py", ("1", "1")) in fake.calls
        assert os.path.isfile(os.path.join(run, "state.json"))

    def test_the_next_work_unit_waits_until_the_previous_job_is_done(self, started):
        fake, state, run = started
        job = state["prepare"]["jobs"]["1"]
        fake.states[job] = ["RUNNING"]
        vs_staged.advance(run)
        assert vs_staged.load_state(run)["prepare"]["submitted"] == 1          # nothing changed, nothing lost
        fake.states[job] = ["COMPLETED"]
        vs_staged.advance(run)
        assert vs_staged.load_state(run)["prepare"]["submitted"] == 2

    def test_a_failed_job_stops_the_run_with_the_reason(self, started):
        fake, state, run = started
        fake.states[state["prepare"]["jobs"]["1"]] = ["COMPLETED", "TIMEOUT", "OUT_OF_MEMORY"]
        with pytest.raises(vs_staged.StagedError, match="TIMEOUT"):
            vs_staged.advance(run)
        after = vs_staged.load_state(run)
        assert after["status"] == "failed" and "TIMEOUT" in after["reason"]
        assert after["prepare"]["submitted"] == 1                              # no further work unit was queued

    def test_a_stopped_run_is_not_advanced_again(self, started, capsys):
        fake, state, run = started
        fake.states[state["prepare"]["jobs"]["1"]] = ["FAILED"]
        with pytest.raises(vs_staged.StagedError):
            vs_staged.advance(run)
        vs_staged.advance(run)
        assert "failed" in capsys.readouterr().out

    def test_retry_submits_the_failed_work_unit_again(self, started):
        fake, state, run = started
        first = state["prepare"]["jobs"]["1"]
        fake.states[first] = ["FAILED"]
        with pytest.raises(vs_staged.StagedError):
            vs_staged.advance(run)
        vs_staged.retry(run)
        fake.states[first] = ["COMPLETED"]
        vs_staged.advance(run)
        after = vs_staged.load_state(run)
        assert after["status"] == "running" and after["prepare"]["submitted"] == 1
        assert after["prepare"]["jobs"]["1"] != first

    def test_the_whole_pipeline_from_start_to_ranking(self, started):
        fake, state, run = started
        todo = [l.split()[0] for l in open(os.path.join(run, "aflp", "tools", "templates", "todo.all")).read().split("\n") if l]
        vs_staged.advance(run)                                                  # prepare work unit 2
        fake.aflp_output(todo)
        vs_staged.advance(run)                                                  # repack, start docking
        state = vs_staged.load_state(run)
        assert state["stage"] == "dock" and state["counts"]["forms"] == 8
        assert state["dock"]["workunits"] == 2 and state["dock"]["submitted"] == 1
        todo_afvs = open(os.path.join(run, "afvs", "tools", "templates", "todo.all")).read().split()
        assert todo_afvs[1::2] == ["4", "4"]
        vs_staged.advance(run)                                                  # dock work unit 2
        names = vs_staged.read_names(run)
        rows = [(f"{i}_S0_T{t}", -5.0 - int(i[-1]) - t) for i in names for t in range(2)]
        write_afvs_csv(run, rows)
        vs_staged.advance(run)                                                  # rank
        state = vs_staged.load_state(run)
        assert state["status"] == "done" and state["counts"]["docked"] == 4 and state["counts"]["failed"] == 0
        assert os.path.isfile(os.path.join(run, "results", "ranking.csv"))
        finals = [c for c in fake.calls if c[0] == "advance"]
        assert [c[2] for c in finals].count(True) == 1                          # only the last advance asks for an END mail

    def test_a_collection_that_aflp_did_not_deliver_stops_the_run(self, started):
        fake, state, run = started
        vs_staged.advance(run)
        with pytest.raises(vs_staged.StagedError, match="no prepared output"):
            vs_staged.advance(run)
        assert vs_staged.load_state(run)["status"] == "failed"

    def test_two_advances_at_once_do_not_collide(self, started):
        fake, state, run = started
        with vs_staged.Lock(run):
            with pytest.raises(vs_staged.StagedError, match="Another"):
                vs_staged.advance(run)

    def test_a_folder_with_a_run_in_it_cannot_be_started_again(self, started, plan):
        with pytest.raises(vs_staged.StagedError, match="already holds"):
            vs_staged.start(plan, submit=False)

    def test_status_describes_the_run(self, started):
        fake, state, run = started
        text = vs_staged.status_text(run)
        assert "running" in text and "Molecules    4 in 2 collections" in text
        assert "work unit 1 of 2 submitted" in text

    def test_a_folder_that_is_not_a_run_is_explained(self, tmp_path):
        with pytest.raises(vs_staged.StagedError, match="not a staged run folder"):
            vs_staged.load_state(str(tmp_path))

    def test_the_command_line_reports_errors_without_a_traceback(self, tmp_path, capsys):
        assert vs_staged.main(["status", str(tmp_path)]) == 1
        assert "not a staged run folder" in capsys.readouterr().err

    def test_vs_cli_hands_the_follow_up_commands_to_the_staged_module(self, tmp_path, capsys):
        import vs_cli
        assert vs_cli.main(["status", str(tmp_path)]) == 1
        assert "not a staged run folder" in capsys.readouterr().err
