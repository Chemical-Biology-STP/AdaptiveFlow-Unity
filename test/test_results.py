"""The results journal, resume rules, output files and program output parsing (vs_dock). No docking programs are run."""
import csv
import json
import os

import pytest

import vs_dock


def row(i, status="docked", score="-7.100", name=None, note=""):
    return {"index": i, "name": name or f"m{i}", "status": status, "best_score": score if status == "docked" else "",
            "n_poses": 9 if status == "docked" else "", "pkd_best": "", "note": note, "smiles": "CCO"}


# --------------------------------------------------------------------------
# The journal: one line appended per finished molecule
# --------------------------------------------------------------------------


class TestJournal:
    def test_each_molecule_is_one_appended_line(self, tmp_path):
        journal = vs_dock.ResultsJournal(str(tmp_path), fresh=True)
        for i in (3, 0, 2):
            journal.add(row(i))
        journal.close()
        lines = (tmp_path / "results.csv").read_text().splitlines()
        assert lines[0].split(",")[:3] == ["index", "name", "status"]
        assert len(lines) == 4

    def test_it_can_be_read_while_the_run_is_going(self, tmp_path):
        journal = vs_dock.ResultsJournal(str(tmp_path), fresh=True)
        journal.add(row(5))
        assert set(vs_dock.read_results(str(tmp_path))) == {5}     # flushed: a killed job loses nothing
        journal.close()

    def test_when_a_molecule_appears_twice_the_later_line_wins(self, tmp_path):
        journal = vs_dock.ResultsJournal(str(tmp_path), fresh=True)
        journal.add(row(1, status="failed", note="first try failed"))
        journal.add(row(1, status="docked", score="-8.200"))
        journal.close()
        assert vs_dock.read_results(str(tmp_path))[1]["status"] == "docked"

    def test_a_line_cut_off_by_a_killed_job_is_ignored(self, tmp_path):
        journal = vs_dock.ResultsJournal(str(tmp_path), fresh=True)
        journal.add(row(0))
        journal.add(row(1))
        journal.close()
        with open(tmp_path / "results.csv", "a") as fh:
            fh.write("2,m2,dock")                     # the job died in the middle of this line
        results = vs_dock.read_results(str(tmp_path))
        assert set(results) == {0, 1}

    def test_a_missing_file_means_no_results(self, tmp_path):
        assert vs_dock.read_results(str(tmp_path)) == {}

    def test_appending_to_an_existing_journal_keeps_the_header_once(self, tmp_path):
        vs_dock.ResultsJournal(str(tmp_path), fresh=True).close()
        journal = vs_dock.ResultsJournal(str(tmp_path), fresh=False)
        journal.add(row(0))
        journal.close()
        assert (tmp_path / "results.csv").read_text().count("index,name") == 1

    def test_adding_a_line_costs_the_same_however_long_the_run_is(self, tmp_path):
        """The first version rewrote the whole table after every molecule: 144 s of bookkeeping at 10,000 molecules."""
        import time

        journal = vs_dock.ResultsJournal(str(tmp_path), fresh=True)
        start = time.time()
        for i in range(20000):
            journal.add(row(i))
        journal.close()
        assert time.time() - start < 20, "appending 20,000 results should take a second or two, not minutes"
        assert len(vs_dock.read_results(str(tmp_path))) == 20000


class TestFinalTable:
    def test_write_results_sorts_by_input_order_and_replaces_the_journal(self, tmp_path):
        vs_dock.write_results(str(tmp_path), {2: row(2), 0: row(0), 1: row(1)})
        assert [r["index"] for r in csv.DictReader(open(tmp_path / "results.csv"))] == ["0", "1", "2"]

    def test_a_row_with_missing_fields_is_written_with_blanks(self, tmp_path):
        vs_dock.write_results(str(tmp_path), {0: {"index": 0, "name": "x", "status": "failed", "note": "why"}})
        out = list(csv.DictReader(open(tmp_path / "results.csv")))
        assert out[0]["best_score"] == "" and out[0]["note"] == "why"

    def test_ranking_orders_docked_molecules_by_score_and_puts_failures_last(self, tmp_path):
        rows = {0: row(0, score="-5.0"), 1: row(1, status="failed"), 2: row(2, score="-9.5"), 3: row(3, score="-7.0")}
        vs_dock.write_ranked_table(str(tmp_path), rows)
        ranking = list(csv.DictReader(open(tmp_path / "ranking.csv")))
        assert [r["name"] for r in ranking] == ["m2", "m3", "m0", "m1"]
        assert [r["rank"] for r in ranking] == ["1", "2", "3", ""]

    def test_sort_key_copes_with_blank_and_missing_scores(self):
        assert vs_dock.sort_key({"best_score": "-9.0"}) < vs_dock.sort_key({"best_score": "-5.0"})
        assert vs_dock.sort_key({"best_score": ""}) > vs_dock.sort_key({"best_score": "-1.0"})
        assert vs_dock.sort_key({}) > vs_dock.sort_key({"best_score": "-1.0"})


class TestPoseFiles:
    def test_folders_are_grouped_by_thousands(self):
        assert vs_dock.ligand_folder("/r", 0) == os.path.join("/r", "ligands", "0000", "000000")
        assert vs_dock.ligand_folder("/r", 999).endswith(os.path.join("0000", "000999"))
        assert vs_dock.ligand_folder("/r", 1000).endswith(os.path.join("0001", "001000"))
        assert vs_dock.ligand_folder("/r", 123456).endswith(os.path.join("0123", "123456"))

    def test_top_and_all_pose_files_are_rebuilt_from_the_folders(self, tmp_path):
        for index, n_poses in ((0, 3), (1, 2)):
            folder = vs_dock.ligand_folder(str(tmp_path), index)
            os.makedirs(folder)
            records = "".join(f"mol{index}_{k}\n\n$$$$\n" for k in range(n_poses))
            open(os.path.join(folder, "poses.sdf"), "w").write(records)
        vs_dock.write_sdf_files(str(tmp_path), {0: row(0), 1: row(1), 2: row(2, status="failed")})
        top = open(tmp_path / "top_poses.sdf").read()
        every = open(tmp_path / "all_poses.sdf").read()
        assert top.count("$$$$") == 2 and every.count("$$$$") == 5
        assert top.startswith("mol0_0") and "mol1_0" in top and "mol0_1" not in top

    def test_a_failed_molecule_contributes_no_poses(self, tmp_path):
        vs_dock.write_sdf_files(str(tmp_path), {0: row(0, status="failed")})
        assert open(tmp_path / "top_poses.sdf").read() == ""


# --------------------------------------------------------------------------
# What the programs print
# --------------------------------------------------------------------------


class TestScoreParsing:
    VINA = ("MODEL 1\nREMARK VINA RESULT:     -9.321      0.000      0.000\nENDMDL\n"
            "MODEL 2\nREMARK VINA RESULT:     -8.800      2.100      4.300\nENDMDL\n")
    SMINA = ("MODEL 1\nREMARK minimizedAffinity -9.24771023\nENDMDL\nMODEL 2\nREMARK minimizedAffinity -8.5\nENDMDL\n")

    def test_vina_family_programs(self):
        assert vs_dock.parse_scores(self.VINA) == [-9.321, -8.8]

    def test_smina_writes_a_different_remark(self):
        assert vs_dock.parse_scores(self.SMINA) == [-9.24771023, -8.5]

    def test_no_scores(self):
        assert vs_dock.parse_scores("") == [] and vs_dock.parse_scores("ATOM  1  C\n") == []

    def test_nnscore2_prints_one_python_2_style_tuple_per_pose(self):
        """NNScore2 is a Python 2 script run under Python 3: its 'print(a, b)' shows up as a tuple."""
        output = ("('\\tBest Score:         ', 9.202, '(Kd = 627.43 pM)')\n"
                  "('\\tBest Score:         ', 9.649, '(Kd = 224.49 pM)')\n")
        assert [float(v) for v in vs_dock.NNSCORE_PATTERN.findall(output)] == [9.202, 9.649]


class TestCommandLines:
    BOX = {"center": [1.0, 2.0, 3.0], "size": [20.0, 20.0, 20.0]}

    @pytest.mark.parametrize("program", sorted(vs_dock.PROGRAMS))
    def test_every_program_gets_the_full_set_of_options(self, program):
        command = vs_dock.program_command(program, "rec.pdbqt", "lig.pdbqt", "out.pdbqt", self.BOX, 16, 9, 7)
        assert command[0].endswith(program)
        for flag, value in (("--exhaustiveness", "16"), ("--num_modes", "9"), ("--cpu", "1"), ("--seed", "7"),
                            ("--center_x", "1.000"), ("--size_z", "20.000")):
            assert command[command.index(flag) + 1] == value, (program, flag)

    def test_smina_uses_short_options_for_the_files(self):
        command = vs_dock.program_command("smina", "rec.pdbqt", "lig.pdbqt", "out.pdbqt", self.BOX, 8, 9, 1)
        assert command[1:7] == ["-r", "rec.pdbqt", "-l", "lig.pdbqt", "-o", "out.pdbqt"]

    def test_the_output_file_ends_in_pdbqt_because_smina_chooses_its_format_from_the_extension(self):
        """Learned the hard way: smina wrote nothing and said 'Not a valid output format' for 'poses.out'."""
        command = vs_dock.program_command("smina", "r", "l", "poses.pdbqt", self.BOX, 8, 9, 1)
        assert command[command.index("-o") + 1].endswith(".pdbqt")

    def test_each_program_is_run_on_one_cpu_so_that_many_molecules_run_at_once(self):
        command = vs_dock.program_command("qvina", "r", "l", "o", self.BOX, 8, 9, 1)
        assert command[command.index("--cpu") + 1] == "1"


# --------------------------------------------------------------------------
# Resuming
# --------------------------------------------------------------------------


class Args:
    """Stands in for the parsed command line."""

    def __init__(self, tmp_path, **overrides):
        # created once: a test that edits a file must not have it reset when the next Args is built
        if not (tmp_path / "rec.pdb").exists():
            (tmp_path / "rec.pdb").write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n")
        if not (tmp_path / "lig.smi").exists():
            (tmp_path / "lig.smi").write_text("CCO ethanol\n")
        self.receptor, self.ligands = str(tmp_path / "rec.pdb"), str(tmp_path / "lig.smi")
        self.program, self.center, self.size = "qvina", [1.0, 2.0, 3.0], [20.0, 20.0, 20.0]
        self.exhaustiveness, self.num_modes, self.seed = 16, 9, 1
        self.rescore, self.add_hydrogens, self.ionisation, self.ph = None, False, "ph", 7.4
        for key, value in overrides.items():
            setattr(self, key, value)


class TestSignature:
    def test_identical_settings_give_identical_signatures(self, tmp_path):
        assert vs_dock.run_signature(Args(tmp_path)) == vs_dock.run_signature(Args(tmp_path))

    @pytest.mark.parametrize("change,name", [
        ({"program": "vina"}, "the docking program"),
        ({"center": [1.0, 2.0, 4.0]}, "the box centre"),
        ({"size": [22.0, 20.0, 20.0]}, "the box size"),
        ({"exhaustiveness": 8}, "the exhaustiveness"),
        ({"num_modes": 5}, "the number of poses"),
        ({"seed": 2}, "the random seed"),
        ({"rescore": "nnscore2"}, "the re-scoring"),
        ({"add_hydrogens": True}, "the hydrogen setting"),
        ({"ionisation": "keep"}, "the ionisation setting"),
        ({"ph": 6.0}, "the ligand pH"),
    ])
    def test_anything_that_changes_the_scores_prevents_a_resume(self, tmp_path, change, name):
        old = vs_dock.run_signature(Args(tmp_path))
        new = vs_dock.run_signature(Args(tmp_path, **change))
        assert name in vs_dock.signature_differences(old, new)

    def test_changed_files_prevent_a_resume(self, tmp_path):
        old = vs_dock.run_signature(Args(tmp_path))
        (tmp_path / "lig.smi").write_text("CCO ethanol\nCCN ethylamine\n")
        assert "the ligand file" in vs_dock.signature_differences(old, vs_dock.run_signature(Args(tmp_path)))
        (tmp_path / "rec.pdb").write_text("ATOM      1  C   ALA A   1       5.000   0.000   0.000  1.00  0.00           C\n")
        assert "the receptor" in vs_dock.signature_differences(old, vs_dock.run_signature(Args(tmp_path)))

    def test_the_ligand_ph_does_not_matter_when_charges_are_kept(self, tmp_path):
        a = vs_dock.run_signature(Args(tmp_path, ionisation="keep", ph=7.4))
        b = vs_dock.run_signature(Args(tmp_path, ionisation="keep", ph=5.0))
        assert vs_dock.signature_differences(a, b) == []


class TestCommandLineChecks:
    def parse(self, tmp_path, *extra):
        a = Args(tmp_path)
        argv = ["--receptor", a.receptor, "--ligands", a.ligands, "--outdir", str(tmp_path / "out"),
                "--center", "1", "2", "3", *extra]
        args = vs_dock.build_parser().parse_args(argv)
        return vs_dock.validate_args(args)

    def test_a_valid_command_line_has_no_problems(self, tmp_path):
        assert self.parse(tmp_path) == []

    def test_defaults(self):
        args = vs_dock.build_parser().parse_args(["--receptor", "r", "--ligands", "l", "--outdir", "o",
                                                  "--center", "0", "0", "0"])
        assert (args.program, args.exhaustiveness, args.num_modes, args.ionisation, args.ph) == \
            ("qvina", 16, 9, "ph", 7.4)

    @pytest.mark.parametrize("extra,message", [
        (["--exhaustiveness", "0"], "exhaustiveness"), (["--num_modes", "0"], "num_modes"),
        (["--size", "0", "10", "10"], "size"), (["--size", "200", "10", "10"], "126"),
        (["--ph", "15"], "ph"), (["--resume", "--force"], "cannot be combined"),
    ])
    def test_invalid_values_are_reported(self, tmp_path, extra, message):
        problems = self.parse(tmp_path, *extra)
        assert problems and any(message in p for p in problems)

    def test_a_missing_file_is_reported(self, tmp_path):
        args = vs_dock.build_parser().parse_args(["--receptor", "/nope.pdb", "--ligands", "/nope.smi", "--outdir", "o",
                                                  "--center", "0", "0", "0"])
        assert len(vs_dock.validate_args(args)) == 2
