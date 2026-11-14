"""`redock`: the best molecules of a finished staged run are docked again in full mode so that poses are kept, and the
two sets of scores are compared. The run file it writes is checked with the real run-file reader; nothing is submitted."""
import csv
import os
import shutil

import pytest

import vs_runfile
import vs_staged
from test_staged import plan  # noqa: F401  (fixture)

RANKING = [("a", "docked", "-9.5", "M0_S0_T0", "CCO"), ("b c", "docked", "-9.0", "M1_S1_T0", "C[C@H](N)C(=O)[O-]"),
           ("d;e", "docked", "-8.0", "M2_S0_T0", "c1ccccc1"), ("f", "docked", "-7.0", "M3_S0_T0", "CCN"),
           ("g", "not_docked", "", "", "CCC"), ("h", "failed", "", "", "CCCC")]


@pytest.fixture
def finished(tmp_path, pocket):
    run = tmp_path / "run"
    (run / "input").mkdir(parents=True)
    (run / "results").mkdir()
    shutil.copyfile(pocket, run / "input" / "receptor.pdb")
    plan = {"run_dir": str(run), "receptor": pocket, "add_hydrogens": False, "center": [15.0, -1.9, -1.2],
            "box_size": 22.0, "program": "qvina", "exhaustiveness": 16, "seed": 7, "partition": "ncpu",
            "job_name": "screen", "strategy": "tranche"}
    state = {"status": "done", "stage": "done", "plan": plan, "counts": {}, "events": []}
    vs_staged.save_state(str(run), state)
    with open(run / "results" / "ranking.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=vs_staged.TRANCHE_RESULT_COLUMNS)
        writer.writeheader()
        for index, (name, status, score, form, smiles) in enumerate(RANKING):
            writer.writerow({"index": index, "name": name, "status": status, "best_score": score, "n_forms": 1,
                             "best_form": form, "note": "", "smiles": smiles, "tranche": "T"})
    return str(run)


def table(path, delimiter=","):
    return list(csv.DictReader(open(path), delimiter=delimiter))


class TestRedockFiles:
    def test_the_best_molecules_are_written_best_first_with_safe_names(self, finished):
        path, n = vs_staged.redock_files(finished, top=3)
        assert n == 3
        lines = open(os.path.join(finished, "redock_input", "hits.smi")).read().splitlines()
        assert lines == ["CCO a", "C[C@H](N)C(=O)[O-] b_c", "c1ccccc1 d_e"]

    def test_the_table_remembers_where_each_hit_came_from(self, finished):
        vs_staged.redock_files(finished, top=3)
        rows = table(os.path.join(finished, "redock_input", "hits.tsv"), "\t")
        assert [(r["index"], r["rank"], r["name"], r["staged_score"], r["form"]) for r in rows] == [
            ("0", "1", "a", "-9.5", "M0_S0_T0"), ("1", "2", "b c", "-9.0", "M1_S1_T0"),
            ("2", "3", "d;e", "-8.0", "M2_S0_T0")]

    def test_only_molecules_with_a_score_are_taken(self, finished):
        _path, n = vs_staged.redock_files(finished, top=100)
        assert n == 4

    def test_the_run_file_is_a_valid_full_mode_run_with_the_settings_of_the_staged_run(self, finished):
        path, _n = vs_staged.redock_files(finished, top=3, cpus=64, num_modes=5)
        cfg = vs_runfile.load(path)
        assert cfg["screen"] == "full" and cfg["receptor"] == os.path.join(finished, "input", "receptor.pdb")
        assert cfg["box_center"] == [15.0, -1.9, -1.2] and cfg["output"] == os.path.join(finished, "redock")
        assert cfg["docking"] == {"program": "qvina", "exhaustiveness": 16, "num_modes": 5, "seed": 7, "box_size": 22.0,
                                  "ionisation": "keep"}
        assert cfg["run"]["mode"] == "submit" and cfg["run"]["partition"] == "ncpu"
        assert cfg["run"]["cpus"] == 3                                      # never more CPUs than molecules
        assert cfg["flagged_ligands"] == "include"

    def test_a_higher_exhaustiveness_and_the_save_mode_can_be_asked_for(self, finished):
        path, _n = vs_staged.redock_files(finished, top=3, exhaustiveness=32, mode="save")
        cfg = vs_runfile.load(path)
        assert cfg["docking"]["exhaustiveness"] == 32 and cfg["run"]["mode"] == "save"

    def test_a_receptor_without_hydrogens_gets_the_decision_the_staged_run_took(self, finished, tmp_path):
        bare = tmp_path / "bare.pdb"
        bare.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n")
        shutil.copyfile(bare, os.path.join(finished, "input", "receptor.pdb"))
        path, _ = vs_staged.redock_files(finished, top=2)
        assert vs_runfile.load(path)["protonation"] == "none"
        shutil.rmtree(os.path.join(finished, "redock_input"))
        state = vs_staged.load_state(finished)
        state["plan"]["add_hydrogens"] = True
        vs_staged.save_state(finished, state)
        path, _ = vs_staged.redock_files(finished, top=2)
        assert vs_runfile.load(path)["protonation"] == "adfr"

    def test_a_receptor_with_hydrogens_needs_no_decision(self, finished):
        path, _ = vs_staged.redock_files(finished, top=2)
        assert "protonation" not in vs_runfile.load(path)

    def test_hydrogens_are_found_by_their_element(self, tmp_path):
        with_h = tmp_path / "h.pdb"
        with_h.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
                          "ATOM      2  HA  ALA A   1       1.000   0.000   0.000  1.00  0.00           H\n")
        without = tmp_path / "n.pdb"
        without.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
                           "ATOM      2  CD  PRO A   2       1.000   0.000   0.000  1.00  0.00           C\n"
                           "HETATM    3 CL   CL  A   3       1.000   0.000   0.000  1.00  0.00          CL\n")
        assert vs_staged.receptor_has_hydrogens(str(with_h))
        assert not vs_staged.receptor_has_hydrogens(str(without))

    def test_runs_that_did_not_keep_the_protein_file_use_the_one_in_the_plan(self, finished):
        os.remove(os.path.join(finished, "input", "receptor.pdb"))
        path, _ = vs_staged.redock_files(finished, top=2)
        assert vs_runfile.load(path)["receptor"].endswith(".pdb")

    def test_a_missing_protein_file_is_explained(self, finished):
        os.remove(os.path.join(finished, "input", "receptor.pdb"))
        state = vs_staged.load_state(finished)
        state["plan"]["receptor"] = "/no/such/file.pdb"
        vs_staged.save_state(finished, state)
        with pytest.raises(vs_staged.StagedError, match="no longer available"):
            vs_staged.redock_files(finished, top=2)

    def test_an_unfinished_run_has_nothing_to_take(self, finished):
        state = vs_staged.load_state(finished)
        state["status"] = "running"
        vs_staged.save_state(finished, state)
        with pytest.raises(vs_staged.StagedError, match="not finished"):
            vs_staged.redock_files(finished, top=2)

    def test_a_second_redock_is_refused_not_overwritten(self, finished):
        vs_staged.redock_files(finished, top=2)
        with pytest.raises(vs_staged.StagedError, match="already has"):
            vs_staged.redock_files(finished, top=2)

    @pytest.mark.parametrize("kwargs,message", [({"top": 0}, "--top"), ({"cpus": 0}, "--cpus"), ({"cpus": 257}, "--cpus")])
    def test_silly_numbers_are_refused(self, finished, kwargs, message):
        with pytest.raises(vs_staged.StagedError, match=message):
            vs_staged.redock_files(finished, **{"top": 3, **kwargs})

    def test_a_run_without_any_score_has_nothing_to_dock(self, finished):
        path = os.path.join(finished, "results", "ranking.csv")
        rows = table(path)
        for r in rows:
            r["status"] = "failed"
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with pytest.raises(vs_staged.StagedError, match="nothing to dock"):
            vs_staged.redock_files(finished, top=3)


def write_results(run, rows):
    folder = os.path.join(run, "redock", "results")
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "results.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["index", "name", "status", "best_score", "n_poses", "pkd_best", "note", "smiles"])
        for index, status, score, poses in rows:
            writer.writerow([index, f"m{index}", status, score, poses, "", "", "C"])


class TestRedockReport:
    def test_the_two_scores_are_put_side_by_side(self, finished):
        vs_staged.redock_files(finished, top=4)
        write_results(finished, [(0, "docked", "-9.9", 9), (1, "docked", "-8.5", 9), (2, "failed", "", 0),
                                 (3, "docked", "-7.0", 5)])
        text = vs_staged.redock_report(finished)
        rows = table(os.path.join(finished, "results", "redock_comparison.csv"))
        assert [(r["name"], r["staged_score"], r["redock_score"], r["difference"], r["redock_status"]) for r in rows] == [
            ("a", "-9.5", "-9.9", "-0.400", "docked"), ("b c", "-9.0", "-8.5", "0.500", "docked"),
            ("d;e", "-8.0", "", "", "failed"), ("f", "-7.0", "-7.0", "0.000", "docked")]
        assert "4 molecules were docked again; 3 have a score (0 not finished, 1 failed)" in text
        assert "within 0.5" in text and "3 of 3" in text and "Correlation" in text and "top_poses.sdf" in text

    def test_molecules_that_are_not_finished_are_counted_as_such(self, finished):
        vs_staged.redock_files(finished, top=4)
        write_results(finished, [(0, "docked", "-9.9", 9)])
        assert "1 have a score (3 not finished, 0 failed)" in vs_staged.redock_report(finished)

    def test_before_the_run_it_says_so(self, finished):
        with pytest.raises(vs_staged.StagedError, match="has not been docked again"):
            vs_staged.redock_report(finished)

    def test_before_the_results_it_says_so(self, finished):
        vs_staged.redock_files(finished, top=2)
        with pytest.raises(vs_staged.StagedError, match="not written results yet"):
            vs_staged.redock_report(finished)


class TestRedockCommand:
    def test_the_command_writes_the_files_and_starts_the_run(self, finished, monkeypatch, capsys):
        started = []
        monkeypatch.setattr(vs_staged, "launch_redock", lambda path, check=False: started.append(path))
        assert vs_staged.main(["redock", finished, "--top", "2", "--cpus", "8", "--save"]) == 0
        assert started == [os.path.join(finished, "redock_input", "run.yaml")]
        assert vs_runfile.load(started[0])["run"]["mode"] == "save"
        assert "2 molecules" in capsys.readouterr().out

    def test_a_failed_start_leaves_the_run_ready_to_try_again(self, finished, monkeypatch):
        def refuse(path, check=False):
            raise SystemExit(2)

        monkeypatch.setattr(vs_staged, "launch_redock", refuse)
        with pytest.raises(SystemExit):
            vs_staged.main(["redock", finished, "--top", "2"])
        assert not os.path.exists(os.path.join(finished, "redock_input"))
        vs_staged.redock_files(finished, top=2)                                  # works again

    def test_the_report_flag_reports(self, finished, capsys):
        vs_staged.redock_files(finished, top=2)
        write_results(finished, [(0, "docked", "-9.9", 9), (1, "docked", "-8.9", 9)])
        assert vs_staged.main(["redock", finished, "--report"]) == 0
        assert "2 have a score" in capsys.readouterr().out

    def test_errors_are_explained_without_a_traceback(self, tmp_path, capsys):
        assert vs_staged.main(["redock", str(tmp_path)]) == 1
        assert "not a staged run folder" in capsys.readouterr().err

    def test_vs_cli_hands_redock_to_the_staged_module(self, tmp_path, capsys):
        import vs_cli
        assert vs_cli.main(["redock", str(tmp_path)]) == 1
        assert "not a staged run folder" in capsys.readouterr().err


class TestStartKeepsTheProteinFile:
    def test_a_started_run_keeps_a_copy_of_the_protein(self, tmp_path, monkeypatch, plan):
        from test_staged import FakeCluster
        (tmp_path / "lib.smi").write_text("CCO a\nCCN b\n")
        FakeCluster(monkeypatch, plan["run_dir"])
        vs_staged.start(plan, submit=False)
        copy = os.path.join(plan["run_dir"], "input", "receptor.pdb")
        assert os.path.isfile(copy) and open(copy).read() == open(plan["receptor"]).read()

