"""Reading molecule files, checking molecules, and preparing them (vs_dock). No docking programs are run here."""
import os

import pytest
from rdkit import Chem

import vs_dock


# --------------------------------------------------------------------------
# Reading libraries
# --------------------------------------------------------------------------


class TestReadingSmiles:
    def test_header_is_skipped_and_indices_start_at_zero(self, library):
        rows = list(vs_dock.iter_ligands(library))
        assert [r[0] for r in rows] == [0, 1, 2, 3, 4, 5]
        assert rows[0][1] == "paracetamol"

    def test_unreadable_row_is_reported_not_dropped(self, library):
        rows = {r[1]: r for r in vs_dock.iter_ligands(library)}
        index, name, mol, error = rows["badrow"]
        assert mol is None and "not a valid SMILES" in error

    def test_count_agrees_with_what_is_yielded(self, library):
        assert vs_dock.count_ligands(library) == len(list(vs_dock.iter_ligands(library))) == 6

    def test_no_header_means_the_first_line_is_a_molecule(self, tmp_path):
        path = tmp_path / "plain.smi"
        path.write_text("CCO ethanol\nCCN ethylamine\n")
        assert [r[1] for r in vs_dock.iter_ligands(str(path))] == ["ethanol", "ethylamine"]
        assert vs_dock.count_ligands(str(path)) == 2

    def test_comments_blank_lines_and_separators(self, tmp_path):
        path = tmp_path / "mixed.smi"
        path.write_text("# a comment\n\nCCO,ethanol\nCCN\tethylamine\nCCC;propane\nC\n")
        rows = list(vs_dock.iter_ligands(str(path)))
        assert [r[1] for r in rows] == ["ethanol", "ethylamine", "propane", "molecule_4"]

    def test_stereochemistry_survives_reading(self, library):
        rows = {r[1]: r for r in vs_dock.iter_ligands(library)}
        assert "@" in Chem.MolToSmiles(rows["ibuprofen_S"][2])

    def test_a_large_file_is_not_held_in_memory(self, tmp_path):
        """iter_ligands is a generator: asking for the first molecule must not read the rest."""
        path = tmp_path / "big.smi"
        path.write_text("".join(f"CCO m{i}\n" for i in range(5000)))
        stream = vs_dock.iter_ligands(str(path))
        assert next(stream)[1] == "m0"


class TestReadingSdf:
    def test_names_and_indices(self, tmp_path):
        path = tmp_path / "lib.sdf"
        writer = Chem.SDWriter(str(path))
        for smi, name in (("CCO", "ethanol"), ("CCN", "ethylamine")):
            mol = Chem.MolFromSmiles(smi)
            mol.SetProp("_Name", name)
            writer.write(mol)
        writer.close()
        assert [(r[0], r[1]) for r in vs_dock.iter_ligands(str(path))] == [(0, "ethanol"), (1, "ethylamine")]
        assert vs_dock.count_ligands(str(path)) == 2

    def test_a_record_that_cannot_be_read_keeps_its_place(self, tmp_path):
        good = Chem.MolToMolBlock(Chem.MolFromSmiles("CCO"))
        path = tmp_path / "lib.sdf"
        path.write_text(good + "$$$$\ngarbage that is not a molecule\n$$$$\n" + good + "$$$$\n")
        rows = list(vs_dock.iter_ligands(str(path)))
        assert [r[0] for r in rows] == [0, 1, 2]
        assert rows[1][2] is None and rows[1][3]

    def test_a_malformed_record_does_not_swallow_the_molecule_after_it(self, tmp_path):
        """RDKit's own supplier loses its place after a bad record and drops the next good molecule too."""
        good = Chem.MolToMolBlock(Chem.MolFromSmiles("CCO"))
        for bad in ("garbage that is not a molecule\n", "", good[:60] + "\n"):
            path = tmp_path / "lib.sdf"
            path.write_text(good + "$$$$\n" + bad + "$$$$\n" + good + "$$$$\n")
            rows = list(vs_dock.iter_ligands(str(path)))
            assert [r[2] is not None for r in rows] == [True, False, True]
            assert len(rows) == vs_dock.count_ligands(str(path)) == 3

    def test_properties_and_3d_coordinates_survive(self, tmp_path):
        from rdkit.Chem import AllChem

        path = tmp_path / "lib.sdf"
        writer = Chem.SDWriter(str(path))
        for smi in ("CCO", "CCN"):
            mol = Chem.AddHs(Chem.MolFromSmiles(smi))
            AllChem.EmbedMolecule(mol, randomSeed=1)
            mol.SetProp("_Name", smi)
            mol.SetProp("score", "1.5")
            writer.write(mol)
        writer.close()
        rows = list(vs_dock.iter_ligands(str(path)))
        assert [r[1] for r in rows] == ["CCO", "CCN"] and rows[0][2].GetConformer().Is3D()

    def test_unnamed_molecules_get_a_number(self, tmp_path):
        path = tmp_path / "lib.sdf"
        path.write_text(Chem.MolToMolBlock(Chem.MolFromSmiles("CCO")) + "$$$$\n")
        assert list(vs_dock.iter_ligands(str(path)))[0][1] == "molecule_1"


# --------------------------------------------------------------------------
# Checking molecules before a job
# --------------------------------------------------------------------------


class TestCheckLigand:
    def check(self, smiles, program="qvina"):
        return vs_dock.check_ligand("x", Chem.MolFromSmiles(smiles), program)

    def test_an_ordinary_molecule_passes(self):
        usable, problems = self.check("CC(=O)Nc1ccc(O)cc1")
        assert usable and problems == []

    @pytest.mark.parametrize("smiles,element", [("OB(O)c1ccccc1", "B"), ("C[Se]C", "Se"), ("C[Si](C)(C)C", "Si")])
    def test_elements_a_program_cannot_read(self, smiles, element):
        usable, problems = self.check(smiles)
        assert not usable and element in problems[0] and "cannot read" in problems[0]

    def test_silicon_is_accepted_by_vina_12_only(self):
        assert not self.check("C[Si](C)(C)C", "qvina")[0]
        assert self.check("C[Si](C)(C)C", "AutodockVina_1.2")[0]

    @pytest.mark.parametrize("smiles,element", [
        ("CCC1=[O+][Cu]2([O+]=C(CC)C1)[O+]=C(CC)CC(CC)=[O+]2", "Cu"),
        ("C[Hg]Cl", "Hg"),
        ("C[As](C)C", "As"),
    ])
    def test_metals_without_an_autodock_type_are_flagged_with_the_element(self, smiles, element):
        usable, problems = self.check(smiles)
        assert not usable
        assert element in problems[0] and "no AutoDock atom type" in problems[0]

    @pytest.mark.parametrize("element", ["Mg", "Ca", "Mn", "Fe", "Zn"])
    def test_metals_that_do_have_a_type_are_not_flagged(self, element):
        assert element in vs_dock.TYPED_ELEMENTS

    @pytest.mark.parametrize("smiles", [
        "c1ccccc1C(=O)[O-].[Na+]", "CC(=O)Nc1ccc(O)cc1.[K+].[Cl-]", "CNCCC(Oc1ccc(cc1)C(F)(F)F)c1ccccc1.Cl", "CC(C)Cc1ccc(cc1)C(C)C(=O)[O-].[Li+]",
    ])
    def test_a_salt_is_a_note_not_a_failure_even_with_a_counter_ion_that_has_no_autodock_type(self, smiles):
        """The job docks the largest fragment only, so a sodium or potassium counter-ion must not block a drug."""
        usable, problems = self.check(smiles)
        assert usable and any("several fragments" in p for p in problems)

    def test_a_salt_of_an_unusable_molecule_is_still_unusable(self):
        usable, problems = self.check("OB(O)c1ccccc1C(=O)[O-].[Na+]")
        assert not usable and "B" in problems[0]

    def test_very_small_and_very_large_molecules(self):
        assert not self.check("C")[0]
        usable, problems = self.check("C" * 90)
        assert usable and any("heavy atoms" in p for p in problems)

    def test_an_unreadable_molecule(self):
        assert vs_dock.check_ligand("x", None, "qvina") == (False, ["could not be read"])

    def test_every_listed_program_has_the_fields_the_tool_relies_on(self):
        for name, info in vs_dock.PROGRAMS.items():
            assert {"title", "seconds_per_ligand", "unsupported", "text"} <= set(info), name
            assert os.access(os.path.join(vs_dock.EXECUTABLES, name), os.X_OK), f"{name} is not an executable file"


# --------------------------------------------------------------------------
# Charge states
# --------------------------------------------------------------------------

NET = lambda mol: sum(a.GetFormalCharge() for a in mol.GetAtoms())


class TestIonisation:
    def clean(self, smiles, **kwargs):
        return vs_dock.clean_molecule(Chem.MolFromSmiles(smiles), **kwargs)

    @pytest.mark.parametrize("smiles,charge", [
        ("CC(=O)O", -1),                      # carboxylic acid, pKa 4.8
        ("OC(=O)c1ccccc1", -1),
        ("CN", 1),                            # primary aliphatic amine
        ("CCN(CC)CC", 1),
        ("NC(=N)c1ccccc1", 1),                # amidine
        ("CNC(=N)N", 1),                      # guanidine
        ("c1ccc(cc1)-c1nnn[nH]1", -1),        # tetrazole
        ("COP(=O)(O)O", -2),                  # phosphate monoester
        ("CS(=O)(=O)O", -1),                  # sulfonic acid: Open Babel misses it, the extra rule catches it
        ("CC(=O)NS(C)(=O)=O", -1),            # acylsulfonamide: same
        ("NCC(=O)O", 0),                      # zwitterion
        ("NCCCCC(N)C(=O)O", 1),               # lysine
    ])
    def test_groups_with_an_unambiguous_state_at_ph_74(self, smiles, charge):
        assert NET(self.clean(smiles)[0]) == charge

    @pytest.mark.parametrize("smiles", [
        "CC(=O)Nc1ccc(O)cc1", "c1ccncc1", "Nc1ccccc1", "CC(N)=O", "CS(N)(=O)=O", "Oc1ccccc1",
        "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "CCO", "CC(=O)OC", "CS",
    ])
    def test_neutral_groups_stay_neutral(self, smiles):
        mol, note = self.clean(smiles)
        assert NET(mol) == 0 and "ionised" not in note

    @pytest.mark.parametrize("smiles", ["[O-][N+](=O)c1ccccc1", "C[N+]([O-])(C)C"])
    def test_nitro_groups_and_n_oxides_are_net_neutral_and_untouched(self, smiles):
        assert NET(self.clean(smiles)[0]) == 0

    def test_a_group_that_is_already_ionised_correctly_stays(self):
        assert NET(self.clean("[O-]C(=O)c1ccccc1")[0]) == -1
        assert NET(self.clean("C[NH3+]")[0]) == 1
        assert NET(self.clean("CS(=O)(=O)[O-]")[0]) == -1     # neutralising first would turn this into the acid

    def test_a_permanent_cation_is_kept(self):
        assert NET(self.clean("C[N+](C)(C)C")[0]) == 1

    @pytest.mark.parametrize("smiles", [
        "C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O", "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O", "OC(=O)[C@H](O)c1ccccc1",
    ])
    def test_stereochemistry_survives_ionisation(self, smiles):
        before = Chem.MolFromSmiles(smiles)
        after = self.clean(smiles)[0]
        # atom numbers change when Open Babel rewrites the SMILES, so compare the R/S labels, not the positions
        labels = lambda mol: sorted(label for _, label in Chem.FindMolChiralCenters(mol, useLegacyImplementation=False))
        assert labels(before) == labels(after) != []

    def test_ionisation_never_changes_the_structure(self):
        for smiles in ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl", "CNCCC(Oc1ccc(cc1)C(F)(F)F)c1ccccc1"):
            before = Chem.MolFromSmiles(smiles)
            after = self.clean(smiles)[0]
            assert vs_dock._skeleton(before) == vs_dock._skeleton(after)

    def test_keep_mode_uses_the_charges_as_written(self):
        for smiles in ("C[NH3+]", "[O-]C(=O)c1ccccc1", "CC(=O)O", "CCN"):
            mol, note = self.clean(smiles, ionisation="keep")
            assert Chem.MolToSmiles(mol) == Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
            assert "ionised" not in note

    def test_the_ph_is_respected(self):
        """A carboxylic acid (pKa about 4.8) is neutral in strong acid and an anion in water."""
        assert NET(self.clean("CC(=O)O", ph=1.0)[0]) == 0
        assert NET(self.clean("CC(=O)O", ph=7.4)[0]) == -1

    def test_the_note_says_what_was_done(self):
        assert "ionised for pH 7.4" in self.clean("CC(=O)O")[1]

    def test_only_the_largest_fragment_is_kept(self):
        mol, note = self.clean("c1ccccc1C(=O)[O-].[Na+]")
        assert len(Chem.GetMolFrags(mol)) == 1 and "largest of 2 fragments" in note

    def test_a_failure_in_open_babel_falls_back_to_the_input(self, monkeypatch):
        import openbabel.openbabel as ob

        def boom(*args, **kwargs):
            raise RuntimeError("Open Babel failed")

        monkeypatch.setattr(ob, "OBConversion", boom)
        mol = Chem.MolFromSmiles("CC(=O)O")
        result, changed = vs_dock.ionise(mol)
        assert result is mol and not changed


# --------------------------------------------------------------------------
# 3D coordinates and PDBQT
# --------------------------------------------------------------------------


class TestPreparation:
    def test_an_ordinary_molecule_becomes_a_pdbqt(self):
        text, note, smiles = vs_dock.ligand_to_pdbqt(Chem.MolFromSmiles("CC(=O)Nc1ccc(O)cc1"))
        assert text.startswith("REMARK") and "ROOT" in text and "TORSDOF" in text
        types = {line[77:79].strip() for line in text.splitlines() if line.startswith(("ATOM", "HETATM"))}
        assert types <= {"A", "C", "HD", "N", "NA", "OA"}

    def test_macrocycles_and_sugars_use_only_standard_atom_types(self):
        """Meeko's default writes CG/G0 ring-closure types that Vina-family programs reject: 23 of 4,961 NCI molecules."""
        for smiles in ("CC1OCOC(CO)C(O)C1O", "COc1cc2c(cc1OC)C13CCN4CC5=CCOC6CC(=O)N2C1C6C5CC43"):
            text = vs_dock.ligand_to_pdbqt(Chem.MolFromSmiles(smiles))[0]
            types = {line[77:79].strip() for line in text.splitlines() if line.startswith(("ATOM", "HETATM"))}
            assert not types & {"CG", "G0", "CG0"}, smiles

    def test_the_smiles_reported_is_what_was_docked(self):
        _, _, smiles = vs_dock.ligand_to_pdbqt(Chem.MolFromSmiles("c1ccccc1C(=O)[O-].[Na+]"))
        assert "Na" not in smiles and "." not in smiles

    def test_stereochemistry_is_kept_in_the_docked_smiles(self):
        _, _, smiles = vs_dock.ligand_to_pdbqt(Chem.MolFromSmiles("CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O"))
        assert "@" in smiles

    @pytest.mark.parametrize("smiles", ["C[C@]12C[C@@]1(C)C2", "C1=C2CCC1CC2"])
    def test_a_molecule_that_cannot_be_embedded_gives_a_clear_error(self, smiles):
        with pytest.raises(vs_dock.LigandError, match="could not generate 3D coordinates"):
            vs_dock.ligand_to_pdbqt(Chem.MolFromSmiles(smiles))

    def test_an_rdkit_exception_during_embedding_is_also_a_ligand_error(self):
        """A zinc complex makes RDKit raise (Invariant Violation) instead of returning a failure code."""
        with pytest.raises(vs_dock.LigandError):
            vs_dock.ligand_to_pdbqt(Chem.MolFromSmiles("C1=[N+]2CC[N+]3=Cc4ccccc4[O][Zn]23[O]c2ccccc21"))

    def test_the_same_seed_gives_the_same_coordinates(self):
        mol = Chem.MolFromSmiles("CC(C)Cc1ccc(cc1)C(C)C(=O)O")
        assert vs_dock.ligand_to_pdbqt(mol, seed=3)[0] == vs_dock.ligand_to_pdbqt(mol, seed=3)[0]
