"""The tranche strategy of the staged screen (vs_staged): tranche IDs, the estimate of the prescreen, the library
regrouped into tranches, the choice of tranches from the prescreen scores, the ranking over both stages, and the stage
machine with stand-ins for SLURM and the upstream scripts. Nothing here needs the cluster."""
import csv
import glob
import gzip
import os
import math
import tarfile

import pytest
from rdkit import Chem

import vs_staged
from test_staged import FakeCluster, make_aflp_tar, plan, run_dir, started  # noqa: F401  (fixtures)

# varied molecules: several land in the same tranche, some alone
SMILES = ["CCO", "CCCO", "CCCCO", "c1ccccc1", "Cc1ccccc1", "CC(=O)Oc1ccccc1C(=O)O",
          "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
          "CC(C)c1c(C(=O)Nc2ccccc2)c(c(n1CCC(O)CC(O)CC(O)=O)c1ccc(F)cc1)-c1ccccc1"]


# --------------------------------------------------------------------------
# Tranche IDs
# --------------------------------------------------------------------------


class TestTrancheId:
    def test_a_value_on_a_cutoff_belongs_to_the_lower_class(self):
        assert vs_staged.tranche_letter(200, [200, 300]) == "A"
        assert vs_staged.tranche_letter(200.01, [200, 300]) == "B"
        assert vs_staged.tranche_letter(10 ** 6, [200, 300]) == "C"

    def test_the_id_has_eighteen_letters_and_unchosen_properties_are_always_a(self):
        tid = vs_staged.tranche_of(Chem.MolFromSmiles(SMILES[-1]), ("mw", "hbd"))
        assert len(tid) == 18 and set(tid) <= set("ABCDEF")
        names = vs_staged.TRANCHE_PROPERTY_NAMES
        assert len(names) == 18
        for position, name in enumerate(names):
            if name not in ("mw", "hbd"):
                assert tid[position] == "A", name

    def test_a_bigger_molecule_has_a_later_molecular_weight_class(self):
        small = vs_staged.tranche_of(Chem.MolFromSmiles("CCO"), ("mw",))
        big = vs_staged.tranche_of(Chem.MolFromSmiles(SMILES[-1]), ("mw",))
        assert small[0] == "A" and big[0] > "C"

    def test_the_same_molecule_always_gets_the_same_tranche(self):
        mol = Chem.MolFromSmiles(SMILES[5])
        assert vs_staged.tranche_of(mol) == vs_staged.tranche_of(Chem.MolFromSmiles(SMILES[5]))

    def test_fewer_chosen_properties_make_fewer_tranches(self):
        mols = [Chem.MolFromSmiles(s) for s in SMILES]
        one = {vs_staged.tranche_of(m, ("mw",)) for m in mols}
        four = {vs_staged.tranche_of(m, vs_staged.DEFAULT_TRANCHE_PROPERTIES) for m in mols}
        assert len(one) <= len(four)

    def test_every_property_gives_a_letter(self):
        mol = Chem.MolFromSmiles(SMILES[-1])
        dims = vs_staged.tranche_dimensions()
        assert set(dims) == set(vs_staged.TRANCHE_PROPERTY_NAMES)
        for name, (function, cutoffs) in dims.items():
            assert vs_staged.tranche_letter(function(mol), cutoffs) in "ABCDEF", name


class TestEstimate:
    def write_names(self, tmp_path, smiles):
        path = tmp_path / "names.tsv"
        path.write_text("id\tname\tsmiles\n" + "".join(f"M{i:09d}\tn{i}\t{s}\n" for i, s in enumerate(smiles)))
        return str(path)

    def test_a_library_inside_the_sample_is_counted_exactly(self, tmp_path):
        smiles = SMILES + ["CCO"] * 4                                   # one tranche is bigger than the representatives
        path = self.write_names(tmp_path, smiles)
        est = vs_staged.estimate_prescreen(path, len(smiles), 1.0, vs_staged.DEFAULT_TRANCHE_PROPERTIES, 2)
        dims = vs_staged.tranche_dimensions()
        counts = {}
        for s in smiles:
            t = vs_staged.tranche_of(Chem.MolFromSmiles(s), vs_staged.DEFAULT_TRANCHE_PROPERTIES, dims)
            counts[t] = counts.get(t, 0) + 1
        assert est["exact"] and est["tranches"] == len(counts)
        assert max(counts.values()) > 2 and est["prescreen_forms"] == sum(min(c, 2) for c in counts.values())
        assert est["prescreen_forms"] < len(smiles)

    def test_more_representatives_never_shrink_the_prescreen(self, tmp_path):
        path = self.write_names(tmp_path, SMILES * 3)
        sizes = [vs_staged.estimate_prescreen(path, len(SMILES) * 3, 1.0, vs_staged.DEFAULT_TRANCHE_PROPERTIES, r)
                 ["prescreen_forms"] for r in (1, 2, 3, 10)]
        assert sizes == sorted(sizes)

    def test_the_prescreen_never_exceeds_the_library(self, tmp_path):
        path = self.write_names(tmp_path, SMILES)
        est = vs_staged.estimate_prescreen(path, len(SMILES), 1.0, vs_staged.DEFAULT_TRANCHE_PROPERTIES, 10)
        assert est["prescreen_forms"] <= len(SMILES)

    def test_a_library_bigger_than_the_sample_is_extrapolated_from_the_tranches_not_the_fraction(self, tmp_path):
        path = self.write_names(tmp_path, SMILES * 40)                  # 360 molecules stand for a library of 3,600
        est = vs_staged.estimate_prescreen(path, 3600, 1.0, vs_staged.DEFAULT_TRANCHE_PROPERTIES, 3, wanted=100)
        assert not est["exact"] and est["tranches"] >= est["tranches_in_sample"]
        assert est["prescreen_forms"] <= 3 * est["tranches"] and est["prescreen_forms"] < 1000

    def test_the_number_of_tranches_of_a_big_library_is_corrected_for_those_not_yet_seen(self, monkeypatch):
        # a sample with 4 tranches seen once and 2 seen twice: Chao1 adds 4*4 / (2 * (2 + 1)) = 2.67 unseen tranches
        monkeypatch.setattr(vs_staged, "sample_molecules", lambda source, total, wanted=0: list("abcdeeff"))
        monkeypatch.setattr(vs_staged, "tranche_dimensions", lambda: None)
        monkeypatch.setattr(vs_staged, "tranche_of", lambda mol, properties, dimensions: mol)
        est = vs_staged.estimate_prescreen("x", 1000, 1.0, ("mw",), 3)
        assert est["tranches_in_sample"] == 6 and est["tranches"] == 9 and not est["exact"]
        assert est["prescreen_forms"] == 27
        small = vs_staged.estimate_prescreen("x", 20, 1.0, ("mw",), 3)         # capped at the number of forms
        assert small["prescreen_forms"] == 20

    def test_an_unreadable_sample_falls_back_to_the_whole_library(self, tmp_path):
        path = self.write_names(tmp_path, ["not a smiles"])
        est = vs_staged.estimate_prescreen(path, 50, 1.0, vs_staged.DEFAULT_TRANCHE_PROPERTIES, 3)
        assert est["prescreen_forms"] == 50 and est["sample"] == 0


# --------------------------------------------------------------------------
# Sizes and plan checks
# --------------------------------------------------------------------------


class TestSizesByStrategy:
    def test_the_default_strategy_docks_every_form(self):
        s = vs_staged.plan_sizes(10000, "qvina", 16, 1, 1, 500)
        assert s["strategy"] == "all" and s["dock_forms"] == s["n_forms"] == s["primary_forms"] == 10000

    def test_the_tranche_strategy_docks_the_prescreen_and_the_primary_screen(self):
        s = vs_staged.plan_sizes(10000, "qvina", 16, 1, 1, 500, "tranche", 2000, 300)
        assert s["n_forms"] == 10000 and s["primary_forms"] == 2000 and s["dock_forms"] == 2300

    def test_the_docking_is_never_more_than_the_library(self):
        s = vs_staged.plan_sizes(1000, "qvina", 16, 1, 1, 500, "tranche", 900, 500)
        assert s["dock_forms"] == 1000

    def test_a_tranche_screen_costs_less_than_docking_everything(self):
        full = vs_staged.plan_sizes(100000, "qvina", 16, 1, 1, 500)
        part = vs_staged.plan_sizes(100000, "qvina", 16, 1, 1, 500, "tranche", 20000, 1500)
        assert part["cpu_hours"] < 0.4 * full["cpu_hours"]

    def test_forms_per_molecule_scale_the_primary_screen(self):
        s = vs_staged.plan_sizes(1000, "qvina", 16, 5, 5, 500, "tranche", 100, 50)
        assert s["primary_forms"] == math.ceil(100 * s["forms_per_molecule"]) > 400

    def test_stage_sizes_follow_the_forms_of_that_stage(self, plan):
        plan["cpus"] = 64
        small, big = vs_staged.stage_sizes(100, plan), vs_staged.stage_sizes(10 ** 6, plan)
        assert small["dock_subjobs"] == 1 and big["dock_subjobs"] > 100


class TestCheckPlanTranche:
    @pytest.fixture(autouse=True)
    def limit(self, monkeypatch):
        monkeypatch.setattr(vs_staged, "cluster_cpu_limit", lambda partition="ncpu": 3000)

    def test_the_defaults_are_accepted(self, plan):
        plan["strategy"] = "tranche"
        vs_staged.check_plan(plan)

    def test_an_unknown_strategy_is_refused(self, plan):
        plan["strategy"] = "random"
        with pytest.raises(vs_staged.StagedError, match="strategy"):
            vs_staged.check_plan(plan)

    @pytest.mark.parametrize("reps", [0, 11, "x", -1])
    def test_representatives_are_one_to_ten(self, plan, reps):
        plan.update(strategy="tranche", representatives=reps)
        with pytest.raises(vs_staged.StagedError, match="representatives"):
            vs_staged.check_plan(plan)

    @pytest.mark.parametrize("reps", [1, 10])
    def test_the_ends_of_the_range_are_allowed(self, plan, reps):
        plan.update(strategy="tranche", representatives=reps)
        vs_staged.check_plan(plan)

    @pytest.mark.parametrize("size", [0, -5, "many"])
    def test_the_primary_size_is_a_positive_whole_number(self, plan, size):
        plan.update(strategy="tranche", primary_size=size)
        with pytest.raises(vs_staged.StagedError, match="primary_size"):
            vs_staged.check_plan(plan)

    def test_unknown_or_repeated_properties_are_refused(self, plan):
        plan.update(strategy="tranche", tranche_properties=["mw", "colour"])
        with pytest.raises(vs_staged.StagedError, match="colour"):
            vs_staged.check_plan(plan)
        plan["tranche_properties"] = ["mw", "mw"]
        with pytest.raises(vs_staged.StagedError, match="different"):
            vs_staged.check_plan(plan)

    def test_the_tranche_keys_are_ignored_when_every_form_is_docked(self, plan):
        plan.update(representatives=99, primary_size=0)
        vs_staged.check_plan(plan)


# --------------------------------------------------------------------------
# The AFVS working copy of the prescreen
# --------------------------------------------------------------------------


class TestPrescreenConfiguration:
    def test_the_prescreen_has_its_own_folder_and_docks_only_the_representatives(self, plan, run_dir):
        sizes = vs_staged.plan_sizes(2000, "qvina", 16, 1, 1, plan["cpus"])
        coll = os.path.join(run_dir, "prescreen", "collections")
        work = vs_staged.configure_afvs(run_dir, plan, sizes, "afvs_prescreen", coll, representatives=4)
        assert work == os.path.join(run_dir, "afvs_prescreen")
        ctrl = open(os.path.join(work, "tools", "templates", "all.ctrl")).read().splitlines()
        for line in ("prescreen_mode=1", "prescreen_ligands_per_tranche=4", f"collection_folder={coll}",
                     "job_name=vs_prescreen", "data_collection_addressing_mode=metatranche"):
            assert line in ctrl, line

    def test_the_primary_screen_is_not_a_prescreen(self, plan, run_dir):
        sizes = vs_staged.plan_sizes(2000, "qvina", 16, 1, 1, plan["cpus"])
        work = vs_staged.configure_afvs(run_dir, plan, sizes)
        ctrl = open(os.path.join(work, "tools", "templates", "all.ctrl")).read().splitlines()
        assert "prescreen_mode=0" in ctrl and "job_name=vs_dock" in ctrl

    def test_a_second_configuration_replaces_the_first(self, plan, run_dir):
        sizes = vs_staged.plan_sizes(2000, "qvina", 16, 1, 1, plan["cpus"])
        work = vs_staged.configure_afvs(run_dir, plan, sizes)
        stale = os.path.join(work, "workflow", "stale.txt")
        os.makedirs(os.path.dirname(stale))
        open(stale, "w").write("x")
        vs_staged.configure_afvs(run_dir, plan, sizes)
        assert not os.path.exists(stale)

    def test_the_stage_folders(self):
        assert vs_staged.stage_dirs("prepare")[0] == "aflp"
        assert vs_staged.stage_dirs("prescreen")[0] == "afvs_prescreen"
        assert vs_staged.stage_dirs("dock")[0] == "afvs"


# --------------------------------------------------------------------------
# The tranche library
# --------------------------------------------------------------------------


def make_run(tmp_path, smiles, per_collection=3, reps=2, properties=None):
    """A run folder after AFLP: names.tsv, the AFLP todo file and one prepared tar per collection (two forms per
    molecule, the second a different stereoisomer)."""
    run = str(tmp_path / "run")
    os.makedirs(os.path.join(run, "input"))
    os.makedirs(os.path.join(run, "prep"))
    os.makedirs(os.path.join(run, "results"))
    os.makedirs(os.path.join(run, "aflp", "tools", "templates"))
    with open(os.path.join(run, "input", "names.tsv"), "w") as fh:
        fh.write("id\tname\tsmiles\n")
        for i, s in enumerate(smiles):
            fh.write(f"M{i:09d}\tn{i}\t{s}\n")
    todo = []
    for number, start in enumerate(range(0, len(smiles), per_collection)):
        key = f"AA_AAA_{vs_staged.base26(number, 2)}"
        ids = list(range(start, min(start + per_collection, len(smiles))))
        todo.append(f"{key} {len(ids)}")
        meta, tranche, coll = key.split("_")
        make_aflp_tar(os.path.join(run, "aflp", "workflow", "complete", "pdbqt", meta, tranche, f"{coll}.tar.gz"), coll,
                      [f"M{i:09d}_S{s}_T0" for i in ids for s in (0, 1)])
    open(os.path.join(run, "aflp", "tools", "templates", "todo.all"), "w").write("\n".join(todo) + "\n")
    state = {"plan": {"strategy": "tranche", "representatives": reps, "run_dir": run,
                      "tranche_properties": list(properties or vs_staged.DEFAULT_TRANCHE_PROPERTIES)},
             "counts": {"molecules": len(smiles)}, "sizes": {"forms_per_molecule": 2.0}}
    return run, state


def read_tar(path):
    with tarfile.open(path) as tf:
        return {m.name: tf.extractfile(m).read().decode() for m in tf.getmembers() if m.isfile()}, tf.getnames()


def table(path):
    return list(csv.DictReader(open(path), delimiter="\t"))


class TestTrancheLibrary:
    def test_every_form_is_filed_under_the_tranche_of_its_molecule(self, tmp_path):
        run, state = make_run(tmp_path, SMILES)
        counts = vs_staged.build_tranche_library(run, state)
        assert counts["forms"] == 2 * len(SMILES)
        rows = table(os.path.join(run, "prep", "ligand_tranches.tsv"))
        assert len(rows) == counts["forms"]
        dims = vs_staged.tranche_dimensions()
        for row in rows:
            smiles = SMILES[int(row["id"][1:])]
            assert row["tranche"] == vs_staged.tranche_of(Chem.MolFromSmiles(smiles),
                                                          vs_staged.DEFAULT_TRANCHE_PROPERTIES, dims)
        assert counts["tranches"] == len({r["tranche"] for r in rows})

    def test_collections_follow_the_layout_afvs_reads(self, tmp_path):
        run, state = make_run(tmp_path, SMILES)
        vs_staged.build_tranche_library(run, state)
        tars = glob.glob(os.path.join(run, "dock", "collections", "*", "*.tar.gz"))
        assert tars
        for path in tars:
            tranche, num = os.path.basename(os.path.dirname(path)), os.path.basename(path)[:-7]
            assert len(tranche) == 18 and len(num) == 7 and num.isdigit() and "_" not in tranche
            members, names = read_tar(path)
            assert names[0] == num and all(n.startswith(num + "/") for n in names[1:])

    def test_the_compnd_line_is_removed(self, tmp_path):
        run, state = make_run(tmp_path, SMILES)
        vs_staged.build_tranche_library(run, state)
        for path in glob.glob(os.path.join(run, "dock", "collections", "*", "*.tar.gz")):
            members, _ = read_tar(path)
            assert members and all("COMPND" not in text and "TORSDOF" in text for text in members.values())

    def test_a_big_tranche_is_split_into_collections(self, tmp_path, monkeypatch):
        monkeypatch.setattr(vs_staged, "TRANCHE_COLLECTION_SIZE", 2)
        run, state = make_run(tmp_path, ["CCO", "CCO", "CCO"], per_collection=3)
        counts = vs_staged.build_tranche_library(run, state)
        rows = table(os.path.join(run, "prep", "tranches.tsv"))
        assert [r["collection"] for r in rows] == ["0000001", "0000002", "0000003"]
        assert counts["tranches"] == 1 and [int(r["ligands"]) for r in rows] == [2, 2, 2]

    def test_the_tables_agree_with_the_files(self, tmp_path):
        run, state = make_run(tmp_path, SMILES)
        vs_staged.build_tranche_library(run, state)
        for row in table(os.path.join(run, "prep", "tranches.tsv")):
            members, _ = read_tar(os.path.join(run, "dock", "collections", row["tranche"], f"{row['collection']}.tar.gz"))
            assert len(members) == int(row["ligands"])

    def test_the_prescreen_takes_the_asked_number_of_forms_from_every_tranche(self, tmp_path):
        run, state = make_run(tmp_path, SMILES, reps=2)
        counts = vs_staged.build_tranche_library(run, state)
        rows = table(os.path.join(run, "prep", "ligand_tranches.tsv"))
        by_tranche = {}
        for row in rows:
            by_tranche[row["tranche"]] = by_tranche.get(row["tranche"], 0) + 1
        assert counts["prescreen_forms"] == sum(min(2, n) for n in by_tranche.values())
        listed = []
        for path in glob.glob(os.path.join(run, "prescreen", "collections", "SFL", "*.tar.gz")):
            members, _ = read_tar(path)
            lines = members[os.path.basename(path)[:-7] + "/.listing"].splitlines()
            listed += [l.split(",") for l in lines]
            assert len(lines) == len([m for m in members if m.endswith(".pdbqt")])
        assert len(listed) == counts["prescreen_forms"]
        per = {}
        for key, form, index in listed:
            tranche, num = key.split("_")
            per.setdefault(tranche, []).append(int(index))
            assert os.path.isfile(os.path.join(run, "dock", "collections", tranche, f"{num}.tar.gz"))
        assert all(sorted(v) == list(range(len(v))) for v in per.values())
        assert set(per) == set(by_tranche)

    def test_every_listed_representative_is_in_the_collection_it_names(self, tmp_path):
        run, state = make_run(tmp_path, SMILES)
        vs_staged.build_tranche_library(run, state)
        for path in glob.glob(os.path.join(run, "prescreen", "collections", "SFL", "*.tar.gz")):
            members, _ = read_tar(path)
            for line in members[os.path.basename(path)[:-7] + "/.listing"].splitlines():
                key, form, _index = line.split(",")
                tranche, num = key.split("_")
                inside, _ = read_tar(os.path.join(run, "dock", "collections", tranche, f"{num}.tar.gz"))
                assert f"{num}/{form}.pdbqt" in inside

    def test_the_representatives_do_not_depend_on_the_input_order(self, tmp_path):
        run1, state1 = make_run(tmp_path / "a", ["CCCO"] * 6, per_collection=6, reps=1)
        run2, state2 = make_run(tmp_path / "b", ["CCCO"] * 6, per_collection=2, reps=1)

        def chosen(run):
            out = set()
            for path in glob.glob(os.path.join(run, "prescreen", "collections", "SFL", "*.tar.gz")):
                members, _ = read_tar(path)
                out |= {l.split(",")[1] for l in members[os.path.basename(path)[:-7] + "/.listing"].splitlines()}
            return out

        for run, state in ((run1, state1), (run2, state2)):
            vs_staged.build_tranche_library(run, state)
        assert chosen(run1) == chosen(run2) and len(chosen(run1)) == 1

    def test_the_todo_file_of_the_prescreen_counts_the_forms(self, tmp_path):
        run, state = make_run(tmp_path, SMILES)
        counts = vs_staged.build_tranche_library(run, state)
        lines = open(os.path.join(run, "prescreen", "todo.all")).read().split("\n")
        lines = [l for l in lines if l]
        assert all(l.split()[0].startswith("SFL_") and l.split()[0].count("_") == 1 for l in lines)
        assert sum(int(l.split()[1]) for l in lines) == counts["prescreen_forms"]

    def test_building_twice_gives_the_same_library(self, tmp_path):
        run, state = make_run(tmp_path, SMILES)
        vs_staged.build_tranche_library(run, state)
        first = sorted(glob.glob(os.path.join(run, "*", "collections", "**", "*.tar.gz"), recursive=True))
        vs_staged.build_tranche_library(run, state)
        assert sorted(glob.glob(os.path.join(run, "*", "collections", "**", "*.tar.gz"), recursive=True)) == first

    def test_a_missing_collection_stops_with_the_names(self, tmp_path):
        run, state = make_run(tmp_path, SMILES)
        os.remove(glob.glob(os.path.join(run, "aflp", "workflow", "complete", "pdbqt", "*", "*", "AA.tar.gz"))[0])
        with pytest.raises(vs_staged.StagedError, match="no prepared output"):
            vs_staged.build_tranche_library(run, state)

    def test_nothing_prepared_is_an_error(self, tmp_path):
        run, state = make_run(tmp_path, ["CCO"])
        for path in glob.glob(os.path.join(run, "aflp", "workflow", "complete", "pdbqt", "*", "*", "*.tar.gz")):
            make_aflp_tar(path, "AA", [])
        with pytest.raises(vs_staged.StagedError, match="no molecule at all"):
            vs_staged.build_tranche_library(run, state)

    def test_a_molecule_with_a_tranche_but_unreadable_smiles_is_left_out(self, tmp_path):
        run, state = make_run(tmp_path, ["CCO", "this is not smiles"], per_collection=2)
        counts = vs_staged.build_tranche_library(run, state)
        assert counts["forms"] == 2


# --------------------------------------------------------------------------
# Choosing the tranches
# --------------------------------------------------------------------------


def write_prescreen_csv(run, rows):
    folder = os.path.join(run, "afvs_prescreen", "output-files", "ds1", "csv", "1")
    os.makedirs(folder, exist_ok=True)
    with gzip.open(os.path.join(folder, "0.csv.gz"), "wt", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["ligand", "collection_key", "scenario", "score_average", "score_min", "attr_smi"])
        for ligand, key, score in rows:
            writer.writerow([ligand, key, "ds1", score, score, "CCO"])


def write_tranche_tables(run, tranches):
    """tranches: {tranche: [collection sizes]}"""
    with open(os.path.join(run, "prep", "tranches.tsv"), "w") as fh:
        fh.write("tranche\tcollection\tligands\n")
        for tranche, sizes in tranches.items():
            for number, size in enumerate(sizes, 1):
                fh.write(f"{tranche}\t{number:07d}\t{size}\n")
    os.makedirs(os.path.join(run, "dock"), exist_ok=True)


class TestSelectTranches:
    T1, T2, T3, T4 = ("A" * 17 + c for c in "ABCD")

    def setup_run(self, tmp_path, primary_size):
        run = str(tmp_path / "run")
        for sub in ("prep", "results", "dock"):
            os.makedirs(os.path.join(run, sub))
        write_tranche_tables(run, {self.T1: [10], self.T2: [10, 5], self.T3: [10], self.T4: [10]})
        write_prescreen_csv(run, [("M0_S0_T0", f"{self.T1}_0000001", -9.0), ("M1_S0_T0", f"{self.T1}_0000001", -8.0),
                                  ("M2_S0_T0", f"{self.T2}_0000001", -6.0), ("M3_S0_T0", f"{self.T2}_0000001", -7.0),
                                  ("M4_S0_T0", f"{self.T3}_0000001", -5.0), ("M5_S0_T0", f"{self.T3}_0000001", -5.5),
                                  ("M6_S0_T0", f"{self.T4}_0000001", -9.5), ("M7_S0_T0", f"{self.T4}_0000001", -3.0)])
        state = {"plan": {"primary_size": primary_size}, "counts": {"molecules": 100}, "sizes": {"forms_per_molecule": 1.0}}
        return run, state

    def todo(self, run):
        return [l.split() for l in open(os.path.join(run, "dock", "todo.all")).read().split("\n") if l]

    def test_the_tranches_with_the_best_mean_score_come_first(self, tmp_path):
        run, state = self.setup_run(tmp_path, primary_size=10)
        total = vs_staged.select_tranches(run, state)
        # means: T1 -8.5, T2 -6.5, T4 -6.25, T3 -5.25
        assert self.todo(run) == [[f"{self.T1}_0000001", "10"]] and total == 10

    def test_whole_collections_are_taken_until_the_size_is_reached(self, tmp_path):
        run, state = self.setup_run(tmp_path, primary_size=11)
        total = vs_staged.select_tranches(run, state)
        assert [t[0] for t in self.todo(run)] == [f"{self.T1}_0000001", f"{self.T2}_0000001"]
        assert total == 20                                              # whole collections: may overshoot, never undershoot

    def test_the_mean_decides_not_the_single_best_ligand(self, tmp_path):
        run, state = self.setup_run(tmp_path, primary_size=21)
        vs_staged.select_tranches(run, state)
        chosen = {t[0].split("_")[0] for t in self.todo(run)}
        assert self.T4 not in chosen and self.T2 in chosen              # T4 has the best ligand but a worse mean than T2

    def test_a_target_bigger_than_the_library_takes_everything_that_was_scored(self, tmp_path):
        run, state = self.setup_run(tmp_path, primary_size=10 ** 6)
        total = vs_staged.select_tranches(run, state)
        assert total == 45 and len(self.todo(run)) == 5

    def test_the_tranche_table_lists_every_tranche_with_its_choice(self, tmp_path):
        run, state = self.setup_run(tmp_path, primary_size=11)
        vs_staged.select_tranches(run, state)
        rows = list(csv.DictReader(open(os.path.join(run, "results", "tranche_scores.csv"))))
        assert [r["tranche"] for r in rows] == [self.T1, self.T2, self.T4, self.T3]
        assert [r["selected"] for r in rows] == ["yes", "yes", "no", "no"]
        assert rows[0]["mean_score"] == "-8.500" and rows[0]["min_score"] == "-9.000" and rows[1]["forms_in_library"] == "15"
        assert state["counts"]["tranches_scored"] == 4 and state["counts"]["tranches_selected"] == 2

    def test_a_tranche_without_a_score_is_never_chosen(self, tmp_path):
        run, state = self.setup_run(tmp_path, primary_size=10 ** 6)
        write_tranche_tables(run, {self.T1: [10], "B" * 18: [10]})
        vs_staged.select_tranches(run, state)
        assert [t[0].split("_")[0] for t in self.todo(run)] == [self.T1]

    def test_the_forms_per_molecule_scale_the_target(self, tmp_path):
        run, state = self.setup_run(tmp_path, primary_size=5)
        state["sizes"]["forms_per_molecule"] = 4.0
        vs_staged.select_tranches(run, state)
        assert state["counts"]["primary_forms"] >= 20

    def test_the_same_scores_always_give_the_same_choice(self, tmp_path):
        run, state = self.setup_run(tmp_path, primary_size=20)
        vs_staged.select_tranches(run, state)
        first = open(os.path.join(run, "dock", "todo.all")).read()
        vs_staged.select_tranches(run, state)
        assert open(os.path.join(run, "dock", "todo.all")).read() == first

    def test_no_prescreen_output_is_an_error(self, tmp_path):
        run, state = self.setup_run(tmp_path, primary_size=10)
        os.remove(glob.glob(os.path.join(run, "afvs_prescreen", "output-files", "ds1", "csv", "*", "*.csv.gz"))[0])
        with pytest.raises(vs_staged.StagedError, match="no result tables"):
            vs_staged.select_tranches(run, state)

    def test_scores_that_are_not_numbers_are_skipped_and_all_unscored_is_an_error(self, tmp_path):
        run, state = self.setup_run(tmp_path, primary_size=10)
        write_prescreen_csv(run, [("M0_S0_T0", f"{self.T1}_0000001", "N/A")])
        with pytest.raises(vs_staged.StagedError, match="docking score"):
            vs_staged.select_tranches(run, state)


# --------------------------------------------------------------------------
# Ranking over both stages
# --------------------------------------------------------------------------


def write_stage_csv(run, folder, rows, number="0"):
    path = os.path.join(run, folder, "output-files", "ds1", "csv", "1")
    os.makedirs(path, exist_ok=True)
    with gzip.open(os.path.join(path, f"{number}.csv.gz"), "wt", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["ligand", "collection_key", "scenario", "score_average", "score_min", "attr_smi"])
        for ligand, score in rows:
            writer.writerow([ligand, "AAAAAAAAAAAAAAAAAA_0000001", "ds1", score, score, f"S-{ligand}"])


class TestRankTranche:
    def setup_run(self, tmp_path):
        run = str(tmp_path / "run")
        for sub in ("input", "prep", "results"):
            os.makedirs(os.path.join(run, sub))
        with open(os.path.join(run, "input", "names.tsv"), "w") as fh:
            fh.write("id\tname\tsmiles\n" + "".join(f"M{i:09d}\tmol{i}\tC{'C' * i}O\n" for i in range(5)))
        with open(os.path.join(run, "prep", "ligand_tranches.tsv"), "w") as fh:
            fh.write("ligand\tid\ttranche\n")
            for i in range(4):                                          # molecule 4 has no prepared form
                fh.write(f"M{i:09d}_S0_T0\tM{i:09d}\t{'A' * 17}{'AB'[i % 2]}\n")
        state = {"plan": {"strategy": "tranche"}, "counts": {}}
        return run, state

    def test_statuses_are_docked_not_docked_and_failed(self, tmp_path):
        run, state = self.setup_run(tmp_path)
        write_stage_csv(run, "afvs_prescreen", [("M000000000_S0_T0", -6.0), ("M000000001_S0_T0", -5.0)])
        write_stage_csv(run, "afvs", [("M000000000_S0_T0", -6.0), ("M000000002_S0_T0", -7.0)])
        assert vs_staged.rank(run, state) == (3, 2)
        rows = {r["name"]: r for r in csv.DictReader(open(os.path.join(run, "results", "results.csv")))}
        assert [rows[f"mol{i}"]["status"] for i in range(5)] == ["docked", "docked", "docked", "not_docked", "failed"]
        assert rows["mol3"]["tranche"] == "A" * 17 + "B" and rows["mol4"]["tranche"] == ""
        assert state["counts"] == {"docked": 3, "failed": 1, "not_docked": 1}

    def test_a_form_docked_in_both_stages_counts_once_and_the_best_score_wins(self, tmp_path):
        run, state = self.setup_run(tmp_path)
        write_stage_csv(run, "afvs_prescreen", [("M000000000_S0_T0", -6.0)])
        write_stage_csv(run, "afvs", [("M000000000_S0_T0", -6.0), ("M000000000_S1_T0", -8.0)])
        vs_staged.rank(run, state)
        rows = list(csv.DictReader(open(os.path.join(run, "results", "ranking.csv"))))
        assert rows[0]["name"] == "mol0" and rows[0]["n_forms"] == "2" and rows[0]["best_score"] == "-8.000"
        assert rows[0]["best_form"] == "M000000000_S1_T0"

    def test_a_molecule_found_only_in_the_prescreen_is_still_ranked(self, tmp_path):
        run, state = self.setup_run(tmp_path)
        write_stage_csv(run, "afvs_prescreen", [("M000000001_S0_T0", -9.0)])
        write_stage_csv(run, "afvs", [("M000000002_S0_T0", -7.0)])
        vs_staged.rank(run, state)
        rows = list(csv.DictReader(open(os.path.join(run, "results", "ranking.csv"))))
        assert [r["name"] for r in rows[:2]] == ["mol1", "mol2"]

    def test_ranking_puts_docked_first_then_the_rest(self, tmp_path):
        run, state = self.setup_run(tmp_path)
        write_stage_csv(run, "afvs", [("M000000003_S0_T0", -2.0)])
        vs_staged.rank(run, state)
        rows = list(csv.DictReader(open(os.path.join(run, "results", "ranking.csv"))))
        assert rows[0]["status"] == "docked" and {r["status"] for r in rows[1:]} == {"not_docked", "failed"}

    def test_the_tranche_column_exists_only_in_tranche_mode(self, tmp_path):
        run, state = self.setup_run(tmp_path)
        write_stage_csv(run, "afvs", [("M000000003_S0_T0", -2.0)])
        vs_staged.rank(run, state)
        assert "tranche" in open(os.path.join(run, "results", "results.csv")).readline()
        state_all = {"plan": {"strategy": "all"}, "counts": {}}
        vs_staged.rank(run, state_all)
        assert "tranche" not in open(os.path.join(run, "results", "results.csv")).readline()

    def test_without_the_primary_screen_output_the_prescreen_alone_is_not_enough(self, tmp_path):
        run, state = self.setup_run(tmp_path)
        write_stage_csv(run, "afvs_prescreen", [("M000000000_S0_T0", -6.0)])
        with pytest.raises(vs_staged.StagedError, match="no result tables"):
            vs_staged.rank(run, {"plan": {"strategy": "all"}, "counts": {}})


# --------------------------------------------------------------------------
# The stage machine
# --------------------------------------------------------------------------


@pytest.fixture
def tranche_run(tmp_path, monkeypatch, plan):
    """A tranche run started with the fake cluster: 9 molecules in 3 collections."""
    (tmp_path / "lib.smi").write_text("".join(f"{s} m{i}\n" for i, s in enumerate(SMILES)))
    monkeypatch.setattr(vs_staged.write_collections, "__defaults__", (3,))
    plan.update(strategy="tranche", representatives=1, primary_size=4)
    fake = FakeCluster(monkeypatch, plan["run_dir"])
    fake.workunits = {"prepare": 2, "dock": 2}
    state = vs_staged.start(plan, submit=True)
    return fake, state, plan["run_dir"]


def finish_aflp(run, fake, forms=2):
    todo = [l.split()[0] for l in open(os.path.join(run, "aflp", "tools", "templates", "todo.all")).read().split("\n") if l]
    start = 0
    for key in todo:
        meta, tranche, coll = key.split("_")
        ids = range(start, min(start + 3, len(SMILES)))
        start += 3
        make_aflp_tar(os.path.join(run, "aflp", "workflow", "complete", "pdbqt", meta, tranche, f"{coll}.tar.gz"), coll,
                      [f"M{i:09d}_S{s}_T0" for i in ids for s in range(forms)])


class TestTrancheStages:
    def test_starting_estimates_the_prescreen_and_leaves_the_primary_screen_for_later(self, tranche_run):
        fake, state, run = tranche_run
        assert state["plan"]["strategy"] == "tranche" and state["estimate"]["exact"]
        assert state["estimate"]["tranches"] >= 3
        assert not os.path.exists(os.path.join(run, "afvs")) and not os.path.exists(os.path.join(run, "afvs_prescreen"))
        assert state["sizes"]["prescreen_forms"] == state["estimate"]["prescreen_forms"]
        assert state["stage"] == "prepare" and state["prepare"]["submitted"] == 1

    def test_the_defaults_are_filled_in(self, tmp_path, monkeypatch, plan):
        (tmp_path / "lib.smi").write_text("".join(f"{s} m{i}\n" for i, s in enumerate(SMILES)))
        plan["strategy"] = "tranche"
        FakeCluster(monkeypatch, plan["run_dir"])
        p = vs_staged.start(plan, submit=False)["plan"]
        assert p["representatives"] == vs_staged.DEFAULT_REPRESENTATIVES
        assert p["tranche_properties"] == list(vs_staged.DEFAULT_TRANCHE_PROPERTIES)
        assert p["primary_size"] == round(len(SMILES) * vs_staged.DEFAULT_PRIMARY_FRACTION)

    def test_a_primary_size_above_the_library_is_cut_to_the_library(self, tmp_path, monkeypatch, plan):
        (tmp_path / "lib.smi").write_text("".join(f"{s} m{i}\n" for i, s in enumerate(SMILES)))
        plan.update(strategy="tranche", primary_size=10 ** 6)
        FakeCluster(monkeypatch, plan["run_dir"])
        assert vs_staged.start(plan, submit=False)["plan"]["primary_size"] == len(SMILES)

    def test_the_whole_pipeline_from_start_to_ranking(self, tranche_run):
        fake, state, run = tranche_run
        vs_staged.advance(run)                                                  # preparation work unit 2
        finish_aflp(run, fake)
        vs_staged.advance(run)                                                  # tranche library, prescreen starts
        state = vs_staged.load_state(run)
        assert state["stage"] == "prescreen" and state["prescreen"]["submitted"] == 1
        assert state["counts"]["forms"] == 2 * len(SMILES) and state["counts"]["tranches"] >= 3
        assert ("afvs_prescreen", "afvs_submit_jobs.py", ("1", "1")) in fake.calls
        assert os.path.isfile(os.path.join(run, "afvs_prescreen", "tools", "templates", "todo.all"))
        vs_staged.advance(run)                                                  # prescreen work unit 2
        listed = []
        for path in glob.glob(os.path.join(run, "prescreen", "collections", "SFL", "*.tar.gz")):
            members, _ = read_tar(path)
            listed += [l.split(",") for l in members[os.path.basename(path)[:-7] + "/.listing"].splitlines()]
        write_prescreen_csv(run, [(form, key, -5.0 - n) for n, (key, form, _i) in enumerate(listed)])
        vs_staged.advance(run)                                                  # choose tranches, primary screen starts
        state = vs_staged.load_state(run)
        assert state["stage"] == "dock" and state["dock"]["submitted"] == 1
        assert state["counts"]["tranches_selected"] >= 1 and state["counts"]["primary_forms"] >= 4 * 1.0
        assert os.path.isfile(os.path.join(run, "results", "tranche_scores.csv"))
        todo = open(os.path.join(run, "afvs", "tools", "templates", "todo.all")).read().split()
        assert todo and all(k.count("_") == 1 for k in todo[::2])
        vs_staged.advance(run)                                                  # primary work unit 2
        forms = [r["ligand"] for r in table(os.path.join(run, "prep", "ligand_tranches.tsv"))][:4]
        write_stage_csv(run, "afvs", [(f, -8.0) for f in forms])
        vs_staged.advance(run)                                                  # rank
        state = vs_staged.load_state(run)
        assert state["status"] == "done"
        assert state["counts"]["docked"] + state["counts"]["not_docked"] + state["counts"]["failed"] == len(SMILES)
        assert os.path.isfile(os.path.join(run, "results", "ranking.csv"))
        assert [c[2] for c in fake.calls if c[0] == "advance"].count(True) == 1

    def test_a_failed_prescreen_job_stops_the_run_and_retry_submits_it_again(self, tranche_run):
        fake, state, run = tranche_run
        vs_staged.advance(run)
        finish_aflp(run, fake)
        vs_staged.advance(run)
        job = vs_staged.load_state(run)["prescreen"]["jobs"]["1"]
        fake.states[job] = ["COMPLETED", "OUT_OF_MEMORY"]
        with pytest.raises(vs_staged.StagedError, match="OUT_OF_MEMORY"):
            vs_staged.advance(run)
        failed = vs_staged.load_state(run)
        assert failed["status"] == "failed" and "afvs_prescreen" in failed["reason"]
        vs_staged.retry(run)
        fake.states[job] = ["COMPLETED"]
        vs_staged.advance(run)
        after = vs_staged.load_state(run)
        assert after["status"] == "running" and after["prescreen"]["jobs"]["1"] != job

    def test_retry_after_a_failed_selection_does_not_resubmit_the_prescreen(self, tranche_run):
        fake, state, run = tranche_run
        vs_staged.advance(run)
        finish_aflp(run, fake)
        vs_staged.advance(run)
        vs_staged.advance(run)                                                  # prescreen work unit 2
        with pytest.raises(vs_staged.StagedError, match="no result tables"):
            vs_staged.advance(run)                                              # the prescreen wrote nothing
        before = vs_staged.load_state(run)["prescreen"]["submitted"]
        vs_staged.retry(run)
        assert vs_staged.load_state(run)["prescreen"]["submitted"] == before

    def test_the_status_shows_the_tranche_numbers(self, tranche_run):
        fake, state, run = tranche_run
        vs_staged.advance(run)
        finish_aflp(run, fake)
        vs_staged.advance(run)
        text = vs_staged.status_text(run)
        assert "Strategy     tranches: 1 representative(s) per tranche" in text
        assert "Tranches" in text and "Prescreen    work unit 1 of 2 submitted" in text

    def test_a_run_started_before_the_tranche_strategy_still_advances(self, started):
        fake, state, run = started
        data = vs_staged.load_state(run)
        data.pop("prescreen", None)
        vs_staged.save_state(run, data)
        vs_staged.advance(run)
        assert vs_staged.load_state(run)["prepare"]["submitted"] == 2



class TestAdvanceJobMemory:
    def make(self, tmp_path, monkeypatch, state):
        captured = {}

        class Done:
            returncode, stdout, stderr = 0, "4242\n", ""

        monkeypatch.setattr(vs_staged.subprocess, "run", lambda *a, **k: Done())
        monkeypatch.setattr(vs_staged.vs_slurm, "mail_address", lambda: None)
        vs_staged.submit_advance(str(tmp_path), state, None)
        text = open(tmp_path / "logs" / "advance.sh").read()
        captured["mem"] = [l for l in text.splitlines() if l.startswith("#SBATCH --mem=")][0]
        return captured["mem"]

    def test_the_memory_follows_the_library_size(self, tmp_path, monkeypatch):
        state = {"plan": {"partition": "ncpu", "strategy": "all"}, "counts": {"molecules": 2_000_000},
                 "sizes": {"n_forms": 2_000_000}}
        assert self.make(tmp_path, monkeypatch, state) == "#SBATCH --mem=7G"          # ceil(2.6) + 4

    def test_the_tranche_library_adds_its_own_memory(self, tmp_path, monkeypatch):
        state = {"plan": {"partition": "ncpu", "strategy": "tranche"}, "counts": {"molecules": 2_000_000},
                 "sizes": {"n_forms": 2_000_000}}
        assert self.make(tmp_path, monkeypatch, state) == "#SBATCH --mem=13G"         # ceil(2.6 + 6.0) + 4
