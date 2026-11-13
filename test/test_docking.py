"""End to end: the real docking programs on the small test receptor. Marked `slow`: about 7 minutes in all."""
import csv
import os
import subprocess
import sys

import pytest
from rdkit import Chem

import vs_dock

pytestmark = [pytest.mark.slow, pytest.mark.usefixtures("adfr_available")]

LIBRARY = ("smiles name\n"
           "CC(=O)Nc1ccc(O)cc1 paracetamol\n"
           "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O ibuprofen_S\n"
           "CN1C(=O)C(c2cccc(-c3cccc(Cl)c3)c2)(C2CCCC2)N=C1N cat13f_like\n"
           "COc1cc2c(cc1OC)C13CCN4CC5=CCOC6CC(=O)N2C1C6C5CC43 cage_alkaloid\n"
           "OB(O)c1ccccc1 boronic_acid\n"
           "CCC1=[O+][Cu]2([O+]=C(CC)C1)[O+]=C(CC)CC(CC)=[O+]2 copper_complex\n"
           "C1=C2CCC1CC2 cannot_embed\n"
           "not_a_smiles badrow\n")


@pytest.fixture(scope="module")
def library_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("lib") / "library.smi"
    path.write_text(LIBRARY)
    return str(path)


def engine(pocket, library, outdir, centre, *extra, program="qvina"):
    """Run vs_dock.py as a job would (a separate process)."""
    command = [sys.executable, "-W", "ignore", os.path.join(vs_dock.REPO_DIR, "vs_dock.py"),
               "--receptor", pocket, "--ligands", library, "--outdir", str(outdir), "--center", *map(str, centre),
               "--size", "22", "22", "22", "--program", program, "--exhaustiveness", "4", "--num_modes", "4",
               "--cpus", "4", *extra]
    return subprocess.run(command, capture_output=True, text=True, timeout=600)


def results(outdir):
    return {r["name"]: r for r in csv.DictReader(open(os.path.join(outdir, "results.csv")))}


@pytest.fixture(scope="module")
def finished_run(tmp_path_factory, pocket, library_file, box_centre):
    out = tmp_path_factory.mktemp("run") / "out"
    proc = engine(pocket, library_file, out, box_centre, "--ionisation", "ph")
    return out, proc


class TestAFullRun:
    def test_the_exit_code_says_that_some_molecules_failed(self, finished_run):
        out, proc = finished_run
        assert proc.returncode == 1, proc.stdout[-500:] + proc.stderr[-500:]

    def test_the_good_molecules_were_docked(self, finished_run):
        out, _ = finished_run
        rows = results(out)
        for name in ("paracetamol", "ibuprofen_S", "cat13f_like", "cage_alkaloid"):
            assert rows[name]["status"] == "docked", (name, rows[name]["note"])
            assert -15 < float(rows[name]["best_score"]) < -3

    def test_each_failure_has_its_own_readable_reason(self, finished_run):
        out, _ = finished_run
        rows = results(out)
        assert "B, which QuickVina" in rows["boronic_acid"]["note"]
        assert "Cu, which has no AutoDock atom type" in rows["copper_complex"]["note"]
        assert "could not generate 3D coordinates" in rows["cannot_embed"]["note"]
        assert "not a valid SMILES" in rows["badrow"]["note"]

    def test_a_failure_does_not_stop_the_others(self, finished_run):
        out, _ = finished_run
        statuses = [r["status"] for r in results(out).values()]
        assert statuses.count("docked") == 4 and statuses.count("failed") == 4

    def test_the_table_is_complete_ordered_and_without_duplicates(self, finished_run):
        out, _ = finished_run
        indices = [int(r["index"]) for r in csv.DictReader(open(out / "results.csv"))]
        assert indices == list(range(8))

    def test_a_larger_more_drug_like_molecule_scores_better_than_a_small_one(self, finished_run):
        rows = results(finished_run[0])
        assert float(rows["cat13f_like"]["best_score"]) < float(rows["paracetamol"]["best_score"])

    def test_the_ranking_is_best_first_with_failures_last(self, finished_run):
        ranking = list(csv.DictReader(open(finished_run[0] / "ranking.csv")))
        docked = [r for r in ranking if r["status"] == "docked"]
        scores = [float(r["best_score"]) for r in docked]
        assert scores == sorted(scores) and [r["status"] for r in ranking][-4:] == ["failed"] * 4
        assert [r["rank"] for r in docked] == ["1", "2", "3", "4"]

    def test_poses_are_complete_molecules_with_every_hydrogen(self, finished_run):
        top = [m for m in Chem.SDMolSupplier(str(finished_run[0] / "top_poses.sdf"), removeHs=False) if m]
        assert len(top) == 4
        by_name = {m.GetProp("ligand"): m for m in top}
        paracetamol = by_name["paracetamol"]
        assert sum(1 for a in paracetamol.GetAtoms() if a.GetAtomicNum() == 1) == 9      # C8H9NO2
        for key in ("pose_rank", "score_kcal_mol", "program"):
            assert paracetamol.HasProp(key)

    def test_the_poses_sit_in_the_box(self, finished_run, box_centre):
        import numpy as np

        for mol in Chem.SDMolSupplier(str(finished_run[0] / "top_poses.sdf"), removeHs=False):
            xyz = mol.GetConformer().GetPositions()
            assert np.all(np.abs(xyz - np.array(box_centre)) < 11 + 4), mol.GetProp("ligand")

    def test_all_poses_holds_every_pose_of_every_docked_molecule(self, finished_run):
        every = [m for m in Chem.SDMolSupplier(str(finished_run[0] / "all_poses.sdf"), removeHs=False) if m]
        assert len(every) >= 4 and len({m.GetProp("ligand") for m in every}) == 4
        ranks = [int(m.GetProp("pose_rank")) for m in every if m.GetProp("ligand") == "paracetamol"]
        assert ranks == sorted(ranks) and ranks[0] == 1

    def test_stereochemistry_is_docked_as_given(self, finished_run):
        row = results(finished_run[0])["ibuprofen_S"]
        assert "@" in row["smiles"]

    def test_the_docked_smiles_shows_the_charge_state(self, finished_run):
        """At pH 7.4 the acid is docked as its anion."""
        assert "[O-]" in results(finished_run[0])["ibuprofen_S"]["smiles"]

    def test_the_macrocycle_like_molecule_that_used_to_fail_now_docks(self, finished_run):
        assert results(finished_run[0])["cage_alkaloid"]["status"] == "docked"

    def test_one_folder_per_molecule_grouped_by_thousands(self, finished_run):
        assert os.path.isfile(finished_run[0] / "ligands" / "0000" / "000000" / "poses.sdf")

    def test_the_settings_are_recorded_for_a_later_resume(self, finished_run):
        import json

        saved = json.load(open(finished_run[0] / "run_settings.json"))
        assert saved["program"] == "qvina" and saved["exhaustiveness"] == 4 and saved["ionisation"] == "ph"


class TestResume:
    def test_only_the_failed_molecules_are_redone(self, finished_run, pocket, library_file, box_centre):
        out, _ = finished_run
        before = {k: v["best_score"] for k, v in results(out).items() if v["status"] == "docked"}
        proc = engine(pocket, library_file, out, box_centre, "--ionisation", "ph", "--resume")
        assert "4 already done, 4 to dock" in proc.stdout
        after = {k: v["best_score"] for k, v in results(out).items() if v["status"] == "docked"}
        assert after == before                                        # finished molecules are not touched

    def test_a_changed_setting_prevents_the_resume(self, finished_run, pocket, library_file, box_centre):
        out, _ = finished_run
        proc = engine(pocket, library_file, out, box_centre, "--ionisation", "ph", "--resume", program="vina")
        assert proc.returncode == 2 and "the docking program" in proc.stderr

    def test_a_folder_with_results_is_not_overwritten_by_accident(self, finished_run, pocket, library_file, box_centre):
        out, _ = finished_run
        proc = engine(pocket, library_file, out, box_centre, "--ionisation", "ph")
        assert proc.returncode == 2 and "not empty" in proc.stderr

    def test_force_deletes_only_folders_this_tool_made(self, tmp_path, pocket, library_file, box_centre):
        foreign = tmp_path / "foreign"
        foreign.mkdir()
        (foreign / "precious.txt").write_text("keep me")
        proc = engine(pocket, library_file, foreign, box_centre, "--force")
        assert proc.returncode == 2 and "will not be deleted" in proc.stderr
        assert (foreign / "precious.txt").read_text() == "keep me"


class TestStopAndResume:
    def test_sigterm_stops_cleanly_and_the_resume_completes_the_table(self, tmp_path, pocket, box_centre):
        """What SLURM does at the time limit: it sends SIGTERM. Finished molecules must be kept."""
        import signal
        import time

        library = tmp_path / "many.smi"
        library.write_text("".join(f"{smi} m{i}\n" for i, smi in enumerate(
            ["CC(=O)Nc1ccc(O)cc1", "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "c1ccc2[nH]ccc2c1", "CCN(CC)CCNC(=O)c1ccc(N)cc1"] * 6)))
        out = tmp_path / "out"
        command = [sys.executable, "-W", "ignore", os.path.join(vs_dock.REPO_DIR, "vs_dock.py"), "--receptor", pocket,
                   "--ligands", str(library), "--outdir", str(out), "--center", *map(str, box_centre), "--size", "22",
                   "22", "22", "--program", "qvina", "--exhaustiveness", "4", "--cpus", "2"]
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        deadline = time.time() + 120
        while time.time() < deadline:                       # wait until some molecules are done
            if os.path.isfile(out / "results.csv") and len(vs_dock.read_results(str(out))) >= 3:
                break
            time.sleep(1)
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=60)
        assert proc.returncode == 130
        done = len(vs_dock.read_results(str(out)))
        assert 3 <= done < 24

        resumed = subprocess.run(command + ["--resume"], capture_output=True, text=True, timeout=600)
        assert f"{done} already done" in resumed.stdout
        indices = [int(r["index"]) for r in csv.DictReader(open(out / "results.csv"))]
        assert indices == list(range(24))                   # complete, ordered, no duplicates


class TestWorkerDeath:
    def test_a_killed_worker_ends_the_run_cleanly_and_the_finished_molecules_are_kept(self, tmp_path, pocket, box_centre):
        """A worker killed by the kernel (out of memory) must not lose the molecules that are done, and the message must
        say what to do. The run can then be resumed."""
        import signal
        import time

        library = tmp_path / "many.smi"
        library.write_text("".join(f"{smi} m{i}\n" for i, smi in enumerate(
            ["CC(=O)Nc1ccc(O)cc1", "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "c1ccc2[nH]ccc2c1", "CCN(CC)CCNC(=O)c1ccc(N)cc1"] * 6)))
        out = tmp_path / "out"
        command = [sys.executable, "-W", "ignore", os.path.join(vs_dock.REPO_DIR, "vs_dock.py"), "--receptor", pocket,
                   "--ligands", str(library), "--outdir", str(out), "--center", *map(str, box_centre), "--size", "22",
                   "22", "22", "--program", "qvina", "--exhaustiveness", "4", "--cpus", "2"]
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        deadline = time.time() + 120
        while time.time() < deadline:
            if os.path.isfile(out / "results.csv") and len(vs_dock.read_results(str(out))) >= 2:
                break
            time.sleep(0.5)
        children = subprocess.run(["pgrep", "-P", str(proc.pid)], capture_output=True, text=True).stdout.split()
        assert children, "the engine should have worker processes"
        os.kill(int(children[0]), signal.SIGKILL)
        _, stderr = proc.communicate(timeout=120)
        assert proc.returncode == 1
        assert "worker process died" in stderr and "--resume" in stderr
        done = len(vs_dock.read_results(str(out)))
        assert 2 <= done < 24

        resumed = subprocess.run(command + ["--resume"], capture_output=True, text=True, timeout=600)
        assert resumed.returncode == 0 and f"{done} already done" in resumed.stdout
        assert [int(r["index"]) for r in csv.DictReader(open(out / "results.csv"))] == list(range(24))


class TestPrograms:
    @pytest.mark.parametrize("program", sorted(vs_dock.PROGRAMS))
    def test_every_offered_program_docks_a_molecule_and_gives_scores_and_poses(self, tmp_path, pocket, box_centre, program):
        library = tmp_path / "one.smi"
        library.write_text("CC(C)Cc1ccc(cc1)C(C)C(=O)O ibuprofen\n")
        proc = engine(pocket, str(library), tmp_path / "out", box_centre, program=program)
        assert proc.returncode == 0, proc.stdout[-300:] + proc.stderr[-300:]
        row = results(tmp_path / "out")["ibuprofen"]
        assert row["status"] == "docked" and int(row["n_poses"]) >= 1 and float(row["best_score"]) < 0
        poses = [m for m in Chem.SDMolSupplier(str(tmp_path / "out" / "all_poses.sdf"), removeHs=False) if m]
        assert poses and all(m.HasProp("score_kcal_mol") for m in poses)

    @pytest.mark.parametrize("program", ["qvina", "AutodockVina_1.2", "smina"])
    def test_a_repeated_run_with_the_same_seed_gives_identical_scores(self, tmp_path, pocket, box_centre, program):
        """True of these three; PSOVina ignores the seed (see the README), so it is deliberately not tested here."""
        library = tmp_path / "one.smi"
        library.write_text("CC(C)Cc1ccc(cc1)C(C)C(=O)O ibuprofen\nCC(=O)Nc1ccc(O)cc1 paracetamol\n")
        scores = []
        for run in ("a", "b"):
            engine(pocket, str(library), tmp_path / run, box_centre, "--seed", "5", program=program)
            scores.append({k: v["best_score"] for k, v in results(tmp_path / run).items()})
        assert scores[0] == scores[1] and len(scores[0]) == 2


class TestOptions:
    def test_nnscore2_adds_a_pkd_for_every_pose(self, tmp_path, pocket, box_centre):
        library = tmp_path / "one.smi"
        library.write_text("CC(C)Cc1ccc(cc1)C(C)C(=O)O ibuprofen\n")
        engine(pocket, str(library), tmp_path / "out", box_centre, "--rescore", "nnscore2")
        row = results(tmp_path / "out")["ibuprofen"]
        assert row["status"] == "docked" and 0 < float(row["pkd_best"]) < 15
        poses = [m for m in Chem.SDMolSupplier(str(tmp_path / "out" / "all_poses.sdf"), removeHs=False) if m]
        assert all(m.HasProp("nnscore2_pkd") for m in poses)

    def test_clean_removes_the_intermediate_files_but_keeps_the_poses(self, tmp_path, pocket, box_centre):
        library = tmp_path / "one.smi"
        library.write_text("CC(=O)Nc1ccc(O)cc1 paracetamol\n")
        engine(pocket, str(library), tmp_path / "out", box_centre, "--clean")
        assert sorted(os.listdir(tmp_path / "out" / "ligands" / "0000" / "000000")) == ["poses.sdf"]

    def test_keeping_charges_docks_the_molecule_as_written(self, tmp_path, pocket, box_centre):
        library = tmp_path / "one.smi"
        library.write_text("CC(C)Cc1ccc(cc1)C(C)C(=O)O ibuprofen\n")
        engine(pocket, str(library), tmp_path / "out", box_centre, "--ionisation", "keep")
        smiles = results(tmp_path / "out")["ibuprofen"]["smiles"]
        assert "[O-]" not in smiles and "C(=O)O" in smiles

    def test_hydrogens_can_be_added_to_a_receptor_that_has_none(self, tmp_path, pocket_noh, box_centre):
        library = tmp_path / "one.smi"
        library.write_text("CC(=O)Nc1ccc(O)cc1 paracetamol\n")
        proc = engine(pocket_noh, str(library), tmp_path / "out", box_centre, "--add_hydrogens")
        assert proc.returncode == 0 and results(tmp_path / "out")["paracetamol"]["status"] == "docked"

    def test_a_missing_receptor_is_reported_before_anything_runs(self, tmp_path, box_centre):
        library = tmp_path / "one.smi"
        library.write_text("CCO ethanol\n")
        proc = engine("/nowhere/protein.pdb", str(library), tmp_path / "out", box_centre)
        assert proc.returncode == 2 and "file not found" in proc.stderr
        assert not (tmp_path / "out").exists()


class TestThroughTheCommandLine:
    """vs_cli.py: the same path a user takes with `ddsuite vs run.yaml`."""

    def test_a_run_file_in_run_mode_gives_a_ranked_table(self, tmp_path, pocket, ligand_sdf):
        import shutil

        shutil.copy(pocket, tmp_path / "protein.pdb")
        shutil.copy(ligand_sdf, tmp_path / "ref.sdf")
        (tmp_path / "mols.smi").write_text("CC(=O)Nc1ccc(O)cc1 paracetamol\nCC(C)Cc1ccc(cc1)C(C)C(=O)O ibuprofen\n")
        (tmp_path / "run.yaml").write_text(
            "receptor: protein.pdb\nreference_ligand: ref.sdf\nligands: mols.smi\noutput: run1\n"
            "docking: {program: qvina, exhaustiveness: 4}\nrun: {mode: run, cpus: 2}\n")
        proc = subprocess.run([sys.executable, "-W", "ignore", os.path.join(vs_dock.REPO_DIR, "vs_cli.py"),
                               str(tmp_path / "run.yaml")], capture_output=True, text=True, timeout=600,
                              env=dict(os.environ, DDSUITE_RUN_COMMAND="ddsuite vs"))
        assert proc.returncode == 0, proc.stdout[-600:] + proc.stderr[-600:]
        assert "Best results" in proc.stdout and "ibuprofen" in proc.stdout
        assert (tmp_path / "run1" / "run.yaml").is_file()
        assert (tmp_path / "run1" / "input" / "receptor.pdb").is_file()
        assert "ddsuite vs" in (tmp_path / "run1" / "run.yaml").read_text()
