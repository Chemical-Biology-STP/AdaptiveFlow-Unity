"""The assistant (vs_wizard): the run-file route and the interactive route, without running the docking itself.

PDB2PQR and the scheduler are replaced by stand-ins, so nothing here needs the cluster or takes long.
"""
import os

import pytest
import yaml
from rdkit import Chem

import vs_dock
import vs_runfile
import vs_wizard


@pytest.fixture(autouse=True)
def fresh_warnings():
    vs_wizard.WARNINGS.clear()


@pytest.fixture
def inputs(tmp_path, pocket, pocket_noh, ligand_sdf, library):
    """Copies of the input files in one folder, as a user would have them."""
    import shutil

    for src, name in ((pocket, "protein.pdb"), (pocket_noh, "protein_noh.pdb"), (ligand_sdf, "ref.sdf"),
                      (library, "mols.smi")):
        shutil.copy(src, tmp_path / name)
    return tmp_path


def run_file(inputs, **extra):
    """Write a run file in the usual shape; extra keys override or add top-level keys."""
    data = {"receptor": "protein.pdb", "reference_ligand": "ref.sdf", "ligands": "mols.smi", "output": "run1",
            "flagged_ligands": "exclude", "docking": {"program": "qvina"}, "run": {"mode": "save", "cpus": 2}}
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key] = {**data[key], **value}
        else:
            data[key] = value
    path = inputs / "run.yaml"
    path.write_text(yaml.safe_dump(data))
    return str(path)


def settings_for(inputs, **extra):
    """Run every step of the assistant on a run file and return the finished Settings."""
    cfg = vs_runfile.load(run_file(inputs, **extra))
    s = vs_wizard.Settings()
    vs_wizard.run_steps(s, cfg)
    return s


# --------------------------------------------------------------------------
# The run-file route
# --------------------------------------------------------------------------


class TestRunFileRoute:
    def test_a_ready_receptor_with_a_reference_ligand(self, inputs, box_centre):
        s = settings_for(inputs)
        assert s.route == "files" and s.receptor_hydrogens > 0
        assert s.center == pytest.approx(box_centre, abs=0.01)
        assert s.protonated is None and not s.add_hydrogens

    def test_the_box_is_the_ligand_size_plus_ten_and_at_least_twenty(self, inputs):
        s = settings_for(inputs)
        assert s.extent == pytest.approx(11.8, abs=0.3)
        assert s.box_size == 22.0 == max(20.0, float(__import__("math").ceil(s.extent + vs_wizard.BOX_MARGIN)))

    def test_an_explicit_box_size_wins(self, inputs):
        assert settings_for(inputs, docking={"box_size": 30}).box_size == 30

    def test_the_box_centre_can_be_given_directly(self, inputs, box_centre):
        data = dict(box_center=box_centre)
        cfg_path = run_file(inputs, **data)
        text = yaml.safe_load(open(cfg_path))
        del text["reference_ligand"]
        open(cfg_path, "w").write(yaml.safe_dump(text))
        s = vs_wizard.Settings()
        vs_wizard.run_steps(s, vs_runfile.load(cfg_path))
        assert s.center == pytest.approx(box_centre) and s.extent == 0.0 and s.box_size == 20.0

    def test_a_centre_far_from_the_protein_is_warned_about(self, inputs):
        cfg_path = run_file(inputs, box_center=[500.0, 500.0, 500.0])
        text = yaml.safe_load(open(cfg_path))
        del text["reference_ligand"]
        open(cfg_path, "w").write(yaml.safe_dump(text))
        vs_wizard.run_steps(vs_wizard.Settings(), vs_runfile.load(cfg_path))
        assert any("nearest protein atom" in w for w in vs_wizard.WARNINGS)

    def test_defaults_are_the_documented_ones(self, inputs):
        s = settings_for(inputs)
        assert (s.program, s.exhaustiveness, s.num_modes, s.seed, s.rescore) == ("qvina", 16, 9, 1, None)
        assert (s.ionisation, s.ligand_ph) == ("ph", 7.4)

    def test_settings_from_the_file_are_used(self, inputs):
        s = settings_for(inputs, docking={"program": "smina", "exhaustiveness": 24, "num_modes": 5, "seed": 3,
                                          "rescore": "nnscore2", "ionisation": "keep"})
        assert (s.program, s.exhaustiveness, s.num_modes, s.seed, s.rescore, s.ionisation) == \
            ("smina", 24, 5, 3, "nnscore2", "keep")

    def test_a_large_box_raises_the_exhaustiveness_unless_the_file_sets_it(self, inputs):
        assert settings_for(inputs, docking={"box_size": 50}).exhaustiveness == 16
        assert settings_for(inputs, docking={"box_size": 50, "exhaustiveness": 8}).exhaustiveness == 8
        assert any("box is large" in w or "box" in w for w in vs_wizard.WARNINGS)

    def test_the_run_folder_and_results_folder(self, inputs):
        s = settings_for(inputs)
        assert s.rundir == str(inputs / "run1") and s.outdir == str(inputs / "run1" / "results")

    def test_cpus_and_time_for_a_job(self, inputs):
        s = settings_for(inputs, run={"mode": "submit", "cpus": 8, "time": "1:00:00"})
        assert (s.mode, s.cpus, s.time_limit, s.partition) == ("submit", 8, "1:00:00", "ncpu")

    def test_the_default_time_comes_from_the_estimate(self, inputs):
        s = settings_for(inputs, run={"mode": "save", "cpus": 2})
        assert vs_ui_minutes(s.time_limit) >= vs_wizard.vs_slurm.MIN_TIME_MINUTES

    def test_a_time_over_seven_days_is_refused(self, inputs):
        with pytest.raises(vs_wizard.Stop, match="7 days"):
            settings_for(inputs, run={"mode": "save", "time": "8-00:00:00"})


def vs_ui_minutes(text):
    from vs_ui import time_to_minutes

    return time_to_minutes(text)


# --------------------------------------------------------------------------
# Molecules and flagged molecules
# --------------------------------------------------------------------------


class TestMolecules:
    def test_the_library_is_checked_in_one_streaming_pass(self, library):
        lib = vs_wizard.check_library(library, "qvina")
        assert lib.total == 6 and lib.usable == 3
        assert sorted(r.name for r in lib.bad) == ["badrow", "boronic_acid", "copper_complex"]
        assert [r.name for r in lib.noted] == ["benzoate_salt"]
        assert lib.first_names == ["paracetamol", "ibuprofen_S", "benzoate_salt"]

    def test_the_library_object_holds_no_molecules(self, library):
        """A library may hold millions of molecules: only the flagged ones are remembered, and without structures."""
        lib = vs_wizard.check_library(library, "qvina")
        assert not any(hasattr(r, "mol") for r in lib.bad + lib.noted)

    def test_each_problem_has_a_readable_reason(self, library):
        reasons = {r.name: r.problems[0] for r in vs_wizard.check_library(library, "qvina").bad}
        assert "B, which QuickVina" in reasons["boronic_acid"]
        assert "Cu, which has no AutoDock atom type" in reasons["copper_complex"]
        assert "not a valid SMILES" in reasons["badrow"]

    def test_flagged_molecules_are_excluded_when_the_file_says_so(self, inputs):
        s = settings_for(inputs)
        assert s.n_kept == 3 and s.exclude == {3, 4, 5}      # boronic acid, copper complex, bad row
        assert s.flagged_choice == "exclude"

    def test_flagged_molecules_can_be_included(self, inputs):
        s = settings_for(inputs, flagged_ligands="include")
        assert s.n_kept == 6 and s.exclude == set()
        assert any("included" in w for w in vs_wizard.WARNINGS)

    def test_the_file_must_say_what_to_do_about_flagged_molecules(self, inputs):
        cfg_path = run_file(inputs)
        text = yaml.safe_load(open(cfg_path))
        del text["flagged_ligands"]
        open(cfg_path, "w").write(yaml.safe_dump(text))
        with pytest.raises(vs_wizard.Stop) as excinfo:
            vs_wizard.run_steps(vs_wizard.Settings(), vs_runfile.load(cfg_path))
        message = str(excinfo.value)
        assert "'flagged_ligands' is missing" in message and "boronic_acid" in message and "exclude" in message

    def test_a_program_that_cannot_read_silicon_flags_it_but_vina_12_does_not(self, tmp_path):
        path = tmp_path / "si.smi"
        path.write_text("C[Si](C)(C)c1ccccc1 silane\n")
        assert vs_wizard.check_library(str(path), "qvina").usable == 0
        assert vs_wizard.check_library(str(path), "AutodockVina_1.2").usable == 1

    def test_the_reduced_file_for_the_job_leaves_out_the_excluded_molecules(self, inputs):
        s = settings_for(inputs)
        staged = vs_wizard.staged_ligands(s)
        names = [r[1] for r in vs_dock.iter_ligands(staged)]
        assert names == ["paracetamol", "ibuprofen_S", "benzoate_salt"]

    def test_the_reduced_file_keeps_stereochemistry(self, inputs):
        staged = vs_wizard.staged_ligands(settings_for(inputs))
        smiles = {r[1]: Chem.MolToSmiles(r[2]) for r in vs_dock.iter_ligands(staged)}
        assert "@" in smiles["ibuprofen_S"]

    def test_the_original_file_is_used_when_nothing_is_excluded(self, inputs):
        s = settings_for(inputs, flagged_ligands="include")
        assert vs_wizard.staged_ligands(s) == s.ligands

    def test_a_reduced_sdf_stays_an_sdf(self, tmp_path):
        source = tmp_path / "lib.sdf"
        writer = Chem.SDWriter(str(source))
        for smi, name in (("CCO", "a"), ("CCN", "b"), ("CCC", "c")):
            mol = Chem.MolFromSmiles(smi)
            mol.SetProp("_Name", name)
            writer.write(mol)
        writer.close()
        out = tmp_path / "reduced.sdf"
        vs_wizard.write_library(str(out), str(source), {1})
        assert [r[1] for r in vs_dock.iter_ligands(str(out))] == ["a", "c"]

    def test_names_with_spaces_become_underscores_in_a_reduced_smiles_file(self, tmp_path):
        source = tmp_path / "lib.smi"
        source.write_text("CCO one\nCCN two\n")
        out = tmp_path / "reduced.smi"
        vs_wizard.write_library(str(out), str(source), {1})
        assert out.read_text().split() == ["CCO", "one"]

    def test_no_usable_molecules_stops_the_run(self, inputs):
        (inputs / "mols.smi").write_text("OB(O)c1ccccc1 only_boron\n")
        with pytest.raises(vs_wizard.Stop, match="None of the molecules can be docked"):
            settings_for(inputs)


# --------------------------------------------------------------------------
# The protein
# --------------------------------------------------------------------------


class TestProtein:
    def test_a_receptor_without_hydrogens_needs_a_decision_in_a_run_file(self, inputs):
        with pytest.raises(vs_wizard.Stop) as excinfo:
            settings_for(inputs, receptor="protein_noh.pdb")
        assert "'protonation' is missing" in str(excinfo.value) and "pdb2pqr" in str(excinfo.value)

    def test_adfr_hydrogens(self, inputs):
        s = settings_for(inputs, receptor="protein_noh.pdb", protonation="adfr")
        assert s.add_hydrogens and s.protonated is None
        assert any("quick guess" in w for w in vs_wizard.WARNINGS)

    def test_no_hydrogens_at_all_is_allowed_but_warned_about(self, inputs):
        s = settings_for(inputs, receptor="protein_noh.pdb", protonation="none")
        assert not s.add_hydrogens
        assert any("results will be poor" in w for w in vs_wizard.WARNINGS)

    def test_pdb2pqr_is_used_at_the_requested_ph(self, inputs, monkeypatch):
        import protein_prep as pp

        seen = {}

        class Prep:
            ph = 6.5
            hydrogens = 1234
            pdb = (inputs / "protein.pdb").read_text()
            pocket, problems, kept_without_hydrogens = [], [], []

            def lines(self):
                return ["protonated"]

            def report_text(self):
                return "report"

        monkeypatch.setattr(pp, "protonate", lambda text, xyz, ph: seen.update(ph=ph) or Prep())
        s = settings_for(inputs, receptor="protein_noh.pdb", protonation="pdb2pqr", ph=6.5)
        assert seen["ph"] == 6.5 and s.protonated is not None
        assert open(s.receptor).read() == Prep.pdb       # the job gets the protonated protein, not the original

    def test_a_pdb2pqr_failure_in_a_run_file_explains_the_alternative(self, inputs, monkeypatch):
        import protein_prep as pp

        def fail(*args):
            raise pp.ProteinPrepError("PDB2PQR failed (exit code 1). missing backbone atoms")

        monkeypatch.setattr(pp, "protonate", fail)
        with pytest.raises(vs_wizard.Stop) as excinfo:
            settings_for(inputs, receptor="protein_noh.pdb", protonation="pdb2pqr")
        message = str(excinfo.value)
        assert "missing backbone atoms" in message and "protonation: adfr" in message

    def test_a_protein_that_is_not_a_pdb_file(self, inputs):
        (inputs / "protein.pdb").write_text("this is not a pdb file\n")
        with pytest.raises(vs_wizard.Stop):
            settings_for(inputs)

    def test_a_reference_ligand_in_2d_is_refused(self, inputs):
        mol = Chem.MolFromSmiles("CCO")
        from rdkit.Chem import AllChem

        AllChem.Compute2DCoords(mol)
        (inputs / "ref.sdf").write_text(Chem.MolToMolBlock(mol) + "$$$$\n")
        with pytest.raises(vs_wizard.Stop, match="reference ligand cannot be used"):
            settings_for(inputs)

    def test_a_reference_ligand_far_from_the_receptor_is_warned_about(self, inputs):
        import numpy as np
        from rdkit.Chem import AllChem

        mol = Chem.SDMolSupplier(str(inputs / "ref.sdf"))[0]
        conf = mol.GetConformer()
        for i in range(mol.GetNumAtoms()):
            p = conf.GetAtomPosition(i)
            conf.SetAtomPosition(i, (p.x + 300, p.y, p.z))
        (inputs / "ref.sdf").write_text(Chem.MolToMolBlock(mol) + "$$$$\n")
        settings_for(inputs)
        assert any("different coordinate systems" in w for w in vs_wizard.WARNINGS)


# --------------------------------------------------------------------------
# A complex to clean up
# --------------------------------------------------------------------------


@pytest.fixture
def complex_pdb(tmp_path, pocket_noh, ligand_sdf):
    """The pocket (without hydrogens, like a deposited structure) plus the crystal ligand as a HETATM residue
    'LIG A 901'."""
    mol = Chem.SDMolSupplier(ligand_sdf)[0]
    mol = Chem.RemoveHs(mol)
    for i, atom in enumerate(mol.GetAtoms(), 1):
        info = Chem.AtomPDBResidueInfo()
        info.SetName(f"{atom.GetSymbol():<2}{i:<2}"[:4])
        info.SetResidueName("LIG")
        info.SetResidueNumber(901)
        info.SetChainId("A")
        info.SetIsHeteroAtom(True)
        atom.SetMonomerInfo(info)
    het = [l for l in Chem.MolToPDBBlock(mol, flavor=2).splitlines() if l.startswith("HETATM")]
    protein = [l for l in open(pocket_noh) if l.startswith("ATOM")]
    path = tmp_path / "complex.pdb"
    path.write_text("".join(protein) + "TER\n" + "\n".join(het) + "\nEND\n")
    return path


class TestComplexRoute:
    def base(self, complex_pdb, inputs, **extra):
        data = {"structure": str(complex_pdb), "ligands": "mols.smi", "output": "run1", "flagged_ligands": "exclude",
                "docking": {"program": "qvina"}, "run": {"mode": "save", "cpus": 2}}
        data.update(extra)
        path = inputs / "complex_run.yaml"
        path.write_text(yaml.safe_dump(data))
        return str(path)

    def test_the_ligand_must_be_named_and_the_candidates_are_listed(self, complex_pdb, inputs):
        with pytest.raises(vs_wizard.Stop) as excinfo:
            vs_wizard.run_steps(vs_wizard.Settings(), vs_runfile.load(self.base(complex_pdb, inputs)))
        message = str(excinfo.value)
        assert "'template_molecule' is missing" in message and "LIG A901" in message

    def test_a_wrong_name_lists_the_real_candidates(self, complex_pdb, inputs):
        with pytest.raises(vs_wizard.Stop) as excinfo:
            vs_wizard.run_steps(vs_wizard.Settings(),
                                vs_runfile.load(self.base(complex_pdb, inputs, template_molecule="XYZ B9")))
        assert "not a molecule that can mark the binding site" in str(excinfo.value) and "LIG A901" in str(excinfo.value)

    def test_the_ligand_is_removed_from_the_protein_and_the_box_is_centred_on_it(self, complex_pdb, inputs, box_centre):
        s = vs_wizard.Settings()
        vs_wizard.run_steps(s, vs_runfile.load(self.base(
            complex_pdb, inputs, template_molecule="LIG A901", chains=["A"], keep_molecules=[], protonation="adfr")))
        assert s.route == "structure" and s.template_label == "LIG A901"
        assert s.center == pytest.approx(box_centre, abs=0.3)
        receptor = open(s.receptor).read()
        assert "LIG" not in receptor and "HETATM" not in receptor
        assert s.prepared.kept_chains == ["A"]

    def test_the_users_structure_file_is_never_changed(self, complex_pdb, inputs):
        before = complex_pdb.read_text()
        vs_wizard.run_steps(vs_wizard.Settings(), vs_runfile.load(self.base(
            complex_pdb, inputs, template_molecule="LIG A901", chains=["A"], keep_molecules=[], protonation="adfr")))
        assert complex_pdb.read_text() == before

    def test_a_complex_without_hydrogens_needs_a_protonation_decision(self, complex_pdb, inputs):
        with pytest.raises(vs_wizard.Stop) as excinfo:
            vs_wizard.run_steps(vs_wizard.Settings(), vs_runfile.load(self.base(
                complex_pdb, inputs, template_molecule="LIG A901", chains=["A"], keep_molecules=[])))
        assert "'protonation' is missing" in str(excinfo.value)

    def test_the_run_file_written_for_a_complex_names_the_choices(self, complex_pdb, inputs):
        s = vs_wizard.Settings()
        vs_wizard.run_steps(s, vs_runfile.load(self.base(
            complex_pdb, inputs, template_molecule="LIG A901", chains=["A"], keep_molecules=[], protonation="adfr")))
        values = vs_wizard.run_file_values(s)
        assert values["structure"] == str(complex_pdb) and values["template_molecule"] == "LIG A901"
        assert values["chains"] == ["A"] and values["keep_molecules"] == [] and values["protonation"] == "adfr"


# --------------------------------------------------------------------------
# Files for the job, the saved run file and resuming
# --------------------------------------------------------------------------


class TestJobFiles:
    def test_engine_arguments(self, inputs, box_centre):
        s = settings_for(inputs, docking={"rescore": "nnscore2", "ionisation": "keep"})
        args = vs_wizard.engine_args(s, final=True)
        text = " ".join(args)
        assert "--program qvina" in text and "--exhaustiveness 16" in text and "--rescore nnscore2" in text
        assert "--ionisation keep" in text and "--ph" not in args       # the pH means nothing when charges are kept
        assert args[args.index("--center") + 1:args.index("--center") + 4] == [f"{c:.3f}" for c in box_centre]
        assert args[args.index("--receptor") + 1] == os.path.join(s.rundir, "input", "receptor.pdb")

    def test_the_ph_is_passed_on(self, inputs):
        s = settings_for(inputs, docking={"ligand_ph": 6.0})
        args = vs_wizard.engine_args(s, final=True)
        assert args[args.index("--ph") + 1] == "6"

    def test_the_arguments_are_accepted_by_the_engines_own_parser(self, inputs):
        s = settings_for(inputs, docking={"rescore": "nnscore2", "ionisation": "ph", "ligand_ph": 6.5})
        args = vs_dock.build_parser().parse_args(vs_wizard.engine_args(s, final=True))
        assert (args.program, args.rescore, args.ionisation, args.ph) == ("qvina", "nnscore2", "ph", 6.5)

    def test_resume_and_force_flags(self, inputs):
        s = settings_for(inputs)
        s.resume = True
        assert "--resume" in vs_wizard.engine_args(s) and "--force" not in vs_wizard.engine_args(s)
        s.resume, s.force = False, True
        assert "--force" in vs_wizard.engine_args(s)

    def test_files_written_for_a_saved_job(self, inputs):
        s = settings_for(inputs)
        os.makedirs(s.rundir)
        vs_wizard.write_files(s)
        assert os.path.isfile(os.path.join(s.rundir, "input", "receptor.pdb"))
        assert os.path.isfile(os.path.join(s.rundir, "input", "molecules.smi"))
        script = open(s.script_path).read()
        assert "#SBATCH --partition=ncpu" in script and "vs_dock.py" in script
        assert os.access(s.script_path, os.X_OK)

    def test_the_job_reads_its_own_copies_so_later_edits_to_the_users_files_do_not_matter(self, inputs):
        s = settings_for(inputs)
        os.makedirs(s.rundir)
        vs_wizard.write_files(s)
        assert str(inputs / "mols.smi") not in open(s.script_path).read()

    def test_only_the_kept_molecules_are_copied(self, inputs):
        s = settings_for(inputs)
        os.makedirs(s.rundir)
        vs_wizard.write_files(s)
        assert len(list(vs_dock.iter_ligands(os.path.join(s.rundir, "input", "molecules.smi")))) == 3

    def test_a_run_mode_job_writes_no_script(self, inputs):
        s = settings_for(inputs, run={"mode": "run", "cpus": 1})
        os.makedirs(s.rundir)
        vs_wizard.write_files(s)
        assert s.script_path == ""

    def test_the_saved_run_file_reproduces_the_run(self, inputs):
        s = settings_for(inputs, docking={"program": "smina", "exhaustiveness": 24, "rescore": "nnscore2",
                                          "ionisation": "keep"}, run={"mode": "submit", "cpus": 8, "time": "2:00:00"})
        os.makedirs(s.rundir)
        saved = vs_wizard.write_run_file(s)
        again = vs_wizard.Settings()
        cfg = vs_runfile.load(saved)
        cfg["output"] = str(inputs / "again")             # a different folder, so that nothing is 'existing'
        vs_wizard.run_steps(again, cfg)
        for attribute in ("program", "exhaustiveness", "num_modes", "seed", "rescore", "ionisation", "box_size", "cpus",
                          "time_limit", "mode", "n_kept"):
            assert getattr(again, attribute) == getattr(s, attribute), attribute
        assert again.center == pytest.approx(s.center)

    def test_a_saved_run_file_never_says_overwrite(self, inputs):
        """Running the file again must ask, not silently delete the earlier results."""
        s = settings_for(inputs)
        s.force = True
        assert "existing_output" not in vs_wizard.run_file_values(s)


class TestExistingResults:
    def prepare_results(self, inputs, **change):
        """Make a results folder that looks like a finished run of the same settings."""
        s = settings_for(inputs)
        os.makedirs(s.outdir)
        os.makedirs(s.rundir, exist_ok=True)
        vs_wizard.write_files(s)
        args = vs_dock.build_parser().parse_args(vs_wizard.engine_args(s))
        with open(os.path.join(s.outdir, vs_dock.SIGNATURE_FILE), "w") as fh:
            import json

            json.dump(vs_dock.run_signature(args), fh)
        vs_dock.write_results(s.outdir, {0: {"index": 0, "name": "paracetamol", "status": "docked", "best_score": "-5.4"}})
        return s

    def test_the_file_must_say_what_to_do(self, inputs):
        self.prepare_results(inputs)
        with pytest.raises(vs_wizard.Stop) as excinfo:
            settings_for(inputs)
        message = str(excinfo.value)
        assert "'existing_output' is missing" in message and "resume" in message and "overwrite" in message

    def test_resume_with_the_same_settings(self, inputs):
        self.prepare_results(inputs)
        s = settings_for(inputs, existing_output="resume")
        assert s.resume and s.remaining == 2

    def test_resume_is_refused_when_a_setting_changed(self, inputs):
        self.prepare_results(inputs)
        with pytest.raises(vs_wizard.Stop, match="the docking program"):
            settings_for(inputs, existing_output="resume", docking={"program": "vina"})

    def test_overwrite(self, inputs):
        self.prepare_results(inputs)
        s = settings_for(inputs, existing_output="overwrite")
        assert s.force and not s.resume

    def test_a_folder_that_this_tool_did_not_make_is_never_touched(self, inputs):
        os.makedirs(inputs / "run1" / "results")
        (inputs / "run1" / "results" / "precious.txt").write_text("keep me")
        with pytest.raises(vs_wizard.Stop, match="not made by this tool"):
            settings_for(inputs, existing_output="overwrite")
        assert (inputs / "run1" / "results" / "precious.txt").read_text() == "keep me"

    def test_a_run_folder_in_an_unwritable_place_is_refused(self, inputs):
        with pytest.raises(vs_wizard.Stop):
            settings_for(inputs, output="/proc/nothing_can_be_made_here/run1")


# --------------------------------------------------------------------------
# The interactive route
# --------------------------------------------------------------------------


class TestInteractive:
    def test_the_whole_conversation_for_a_ready_receptor(self, inputs, answers, monkeypatch):
        monkeypatch.chdir(inputs)
        feed = answers([
            "2",                         # protein only
            "protein.pdb",               # receptor
            "1",                         # a ligand in the site
            "ref.sdf",
            "",                          # box size: accept the suggestion
            "2",                         # program: AutodockVina_1.2
            "yes",                       # re-score with NNScore2
            "no",                        # no advanced settings
            "mols.smi",                  # molecule file
            "yes",                       # leave out the molecules that cannot be docked
            "1",                         # charge states: set for a pH
            "6.5",                       # the pH
            "3",                         # only save the job script
            "run_x",                     # run folder
            "",                          # job name
            "4",                         # CPUs
            "",                          # time limit
            "no",                        # do not change the partition
        ])
        s = vs_wizard.Settings()
        vs_wizard.run_steps(s, None)
        assert not feed.answers, f"the assistant did not ask {len(feed.answers)} of the expected questions"
        assert (s.program, s.rescore, s.ionisation, s.ligand_ph) == ("AutodockVina_1.2", "nnscore2", "ph", 6.5)
        assert s.n_kept == 3 and s.cpus == 4 and s.mode == "save" and s.rundir == str(inputs / "run_x")
        assert s.box_size == 22.0

    def test_enter_accepts_every_default(self, inputs, answers, monkeypatch):
        monkeypatch.chdir(inputs)
        answers(["2", "protein.pdb", "1", "ref.sdf", "", "", "", "", "mols.smi", "", "", "", "3", "", "", "", "", ""])
        s = vs_wizard.Settings()
        vs_wizard.run_steps(s, None)
        assert (s.program, s.exhaustiveness, s.rescore, s.ionisation, s.ligand_ph) == ("qvina", 16, None, "ph", 7.4)
        assert s.n_kept == 3                                         # the default is to leave flagged molecules out

    def test_a_missing_file_is_asked_for_again(self, inputs, answers, monkeypatch):
        monkeypatch.chdir(inputs)
        feed = answers(["2", "nope.pdb", "protein.pdb", "1", "ref.sdf", "", "", "", "", "mols.smi", "", "", "", "3", "", "", "",
                        "", ""])
        vs_wizard.run_steps(vs_wizard.Settings(), None)
        assert any("Receptor PDB file" in q for q in feed.asked)

    def test_keeping_the_charges_asks_no_ph(self, inputs, answers, monkeypatch):
        monkeypatch.chdir(inputs)
        feed = answers(["2", "protein.pdb", "1", "ref.sdf", "", "", "", "", "mols.smi", "", "2", "3", "", "", "", "", ""])
        s = vs_wizard.Settings()
        vs_wizard.run_steps(s, None)
        assert s.ionisation == "keep" and not any(q.strip().startswith("pH") for q in feed.asked)

    def test_the_review_lists_the_charge_setting_and_asks_for_confirmation(self, inputs, answers, capsys):
        s = settings_for(inputs)
        answers(["yes"])
        assert vs_wizard.step_review(s) is True
        out = capsys.readouterr().out
        assert "Charge states" in out and "pH 7.4" in out and "Save the job script?" not in out.splitlines()[0]

    def test_declining_the_review_stops_without_running(self, inputs, answers):
        s = settings_for(inputs)
        answers(["no"])
        assert vs_wizard.step_review(s) is False

    def test_a_cancelled_run_writes_nothing(self, inputs, answers, monkeypatch):
        monkeypatch.chdir(inputs)
        answers(["2", "protein.pdb", "1", "ref.sdf", "", "", "", "", "mols.smi", "", "", "", "3", "", "", "", "", "", "no"])
        vs_wizard.run_interactive()
        assert not (inputs / "vs_mols").exists()


# --------------------------------------------------------------------------
# Checks on the review and results text
# --------------------------------------------------------------------------


class TestReport:
    def test_the_review_shows_the_real_numbers(self, inputs, capsys):
        s = settings_for(inputs)
        vs_wizard.step_review(s, confirm=False)
        out = capsys.readouterr().out
        assert "3 of 6" in out and "QuickVina 2.1, exhaustiveness 16" in out and "22 A edge" in out
        assert "ncpu / 2" in out

    def test_results_are_shown_best_first_with_failures_listed(self, inputs, capsys):
        s = settings_for(inputs)
        os.makedirs(s.outdir)
        vs_dock.write_results(s.outdir, {
            0: {"index": 0, "name": "weak", "status": "docked", "best_score": "-5.0"},
            1: {"index": 1, "name": "strong", "status": "docked", "best_score": "-9.5"},
            2: {"index": 2, "name": "broken", "status": "failed", "note": "RDKit could not generate 3D coordinates"}})
        vs_wizard.show_results(s)
        out = capsys.readouterr().out
        assert out.index("strong") < out.index("weak")
        assert "1 molecule(s) failed" in out and "RDKit could not generate" in out
        assert "favour large, flat, aromatic molecules" in out

    def test_no_results_yet(self, inputs, capsys):
        s = settings_for(inputs)
        vs_wizard.show_results(s)
        assert "No results have been written" in capsys.readouterr().out

    def test_the_default_job_name_comes_from_the_molecule_file(self):
        assert vs_wizard.default_jobname("/data/My Library (v2).smi") == "vs_My_Library__v2_"
        assert len(vs_wizard.default_jobname("/data/" + "x" * 100 + ".smi")) == 40
