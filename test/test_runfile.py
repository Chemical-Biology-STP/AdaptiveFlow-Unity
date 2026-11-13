"""The YAML run file (vs_runfile): reading, checking and writing."""
import textwrap

import pytest
import yaml

import vs_runfile


@pytest.fixture
def files(tmp_path):
    """The input files a valid run file names, in the folder the run file lives in."""
    for name in ("protein.pdb", "complex.pdb", "ref.sdf", "mols.smi"):
        (tmp_path / name).write_text("x\n")
    return tmp_path


def write(files, body):
    path = files / "run.yaml"
    path.write_text(textwrap.dedent(body))
    return str(path)


# no indentation: tests append and replace text, and the result must still be valid YAML
MINIMAL = "receptor: protein.pdb\nreference_ligand: ref.sdf\nligands: mols.smi\noutput: run1\nrun:\n  mode: save\n"


class TestValidFiles:
    def test_the_smallest_valid_file(self, files):
        cfg = vs_runfile.load(write(files, MINIMAL))
        assert cfg["route"] == "files"
        assert cfg["receptor"] == str(files / "protein.pdb")
        assert cfg["output"] == str(files / "run1")
        assert cfg["run"] == {"mode": "save"} and cfg["docking"] == {}

    def test_paths_are_relative_to_the_run_file_not_to_the_current_folder(self, files, tmp_path_factory, monkeypatch):
        monkeypatch.chdir(tmp_path_factory.mktemp("elsewhere"))
        assert vs_runfile.load(write(files, MINIMAL))["ligands"] == str(files / "mols.smi")

    def test_a_complex_to_clean_up(self, files):
        cfg = vs_runfile.load(write(files, """\
            structure: complex.pdb
            template_molecule: B7Z A401
            chains: [A, B]
            keep_molecules: [ZN A601]
            ligands: mols.smi
            output: run1
            run: {mode: submit}
        """))
        assert cfg["route"] == "structure"
        assert (cfg["template_molecule"], cfg["chains"], cfg["keep_molecules"]) == ("B7Z A401", ["A", "B"], ["ZN A601"])

    def test_chains_may_be_written_as_one_text(self, files):
        cfg = vs_runfile.load(write(files, "structure: complex.pdb\nchains: A, B\nligands: mols.smi\noutput: r\nrun: {mode: save}\n"))
        assert cfg["chains"] == ["A", "B"]

    def test_the_box_centre_instead_of_a_reference_ligand(self, files):
        cfg = vs_runfile.load(write(files, """\
            receptor: protein.pdb
            box_center: [12.5, -3, 40.2]
            ligands: mols.smi
            output: r
            run: {mode: save}
        """))
        assert cfg["box_center"] == [12.5, -3.0, 40.2]

    def test_all_the_docking_and_run_settings(self, files):
        cfg = vs_runfile.load(write(files, "receptor: protein.pdb\nreference_ligand: ref.sdf\nligands: mols.smi\n"
                                           "output: run1\n"
                                           "docking:\n  program: AutodockVina_1.2\n  exhaustiveness: 24\n  num_modes: 5\n"
                                           "  seed: 3\n  box_size: 25\n  rescore: nnscore2\n  ionisation: ph\n"
                                           "  ligand_ph: 6.5\n"
                                           "run:\n  mode: submit\n  job_name: my_run\n  cpus: 64\n  time: \"3:30:00\"\n"
                                           "  partition: ncpu\n  watch: true\n"))
        assert cfg["docking"] == {"program": "AutodockVina_1.2", "exhaustiveness": 24, "num_modes": 5, "seed": 3,
                                  "box_size": 25.0, "rescore": "nnscore2", "ionisation": "ph", "ligand_ph": 6.5}
        assert cfg["run"] == {"mode": "submit", "job_name": "my_run", "cpus": 64, "time": "3:30:00",
                              "partition": "ncpu", "watch": True}

    def test_a_program_name_in_any_case_is_matched_to_the_real_name(self, files):
        cfg = vs_runfile.load(write(files, MINIMAL.replace("run:", "docking: {program: autodockvina_1.2}\nrun:")))
        assert cfg["docking"]["program"] == "AutodockVina_1.2"

    def test_rescore_can_be_switched_off_in_several_ways(self, files):
        for value in ("none", "false"):
            cfg = vs_runfile.load(write(files, MINIMAL.replace("run:", f"docking: {{rescore: {value}}}\nrun:")))
            assert cfg["docking"]["rescore"] is None


class TestProblems:
    def problems(self, files, body):
        with pytest.raises(vs_runfile.RunFileError) as excinfo:
            vs_runfile.load(write(files, body))
        return str(excinfo.value)

    def test_every_problem_is_listed_not_only_the_first(self, files):
        text = self.problems(files, """\
            receptor: missing.pdb
            ligands: nothing.smi
            docking: {exhaustiveness: 0, program: nonsense}
            run: {mode: fly}
        """)
        for expected in ("receptor", "ligands", "output", "reference_ligand", "exhaustiveness", "program", "mode"):
            assert expected in text

    def test_a_misspelt_key_gets_a_suggestion(self, files):
        text = self.problems(files, MINIMAL + "ligand_file: x\nexhaustivness: 4\n")
        assert "unknown key" in text and "Did you mean" in text

    def test_a_misspelt_key_inside_a_section(self, files):
        text = self.problems(files, MINIMAL.replace("run:", "docking: {exhaustivenes: 4}\nrun:"))
        assert "docking.exhaustivenes" in text and "exhaustiveness" in text

    def test_both_a_structure_and_a_receptor(self, files):
        assert "not both" in self.problems(files, MINIMAL + "structure: complex.pdb\n")

    def test_neither_a_structure_nor_a_receptor(self, files):
        assert "either 'structure'" in self.problems(files, "ligands: mols.smi\noutput: r\nrun: {mode: save}\n")

    def test_a_receptor_needs_a_place_for_the_box(self, files):
        assert "reference_ligand" in self.problems(files, "receptor: protein.pdb\nligands: mols.smi\noutput: r\nrun: {mode: save}\n")

    def test_not_both_a_reference_ligand_and_a_centre(self, files):
        assert "not both" in self.problems(files, MINIMAL + "box_center: [0, 0, 0]\n")

    def test_the_box_centre_needs_three_numbers(self, files):
        text = self.problems(files, "receptor: protein.pdb\nbox_center: [1, 2]\nligands: mols.smi\noutput: r\nrun: {mode: save}\n")
        assert "three numbers" in text

    def test_complex_options_do_not_apply_to_a_ready_receptor(self, files):
        assert "only applies together with 'structure'" in self.problems(files, MINIMAL + "chains: [A]\n")

    def test_the_run_mode_is_required(self, files):
        assert "run.mode" in self.problems(files, MINIMAL.replace("run:\n  mode: save\n", ""))

    def test_an_unquoted_time_is_explained(self, files):
        """YAML reads an unquoted 2:00:00 as the number 7200, which SLURM would take as 7200 minutes."""
        text = self.problems(files, MINIMAL.replace("mode: save", "mode: save\n  time: 2:00:00"))
        assert "quotes" in text

    def test_a_malformed_time(self, files):
        assert "hours:minutes:seconds" in self.problems(files, MINIMAL.replace("mode: save", 'mode: save\n  time: "soon"'))

    def test_the_ligand_ph_needs_ph_mode(self, files):
        text = self.problems(files, MINIMAL.replace("run:", "docking: {ionisation: keep, ligand_ph: 6}\nrun:"))
        assert "only applies with 'ionisation: ph'" in text

    def test_ph_values_are_range_checked(self, files):
        assert "at most 14" in self.problems(files, MINIMAL.replace("run:", "docking: {ligand_ph: 15}\nrun:"))

    def test_wrong_file_types(self, files):
        (files / "protein.txt").write_text("x")
        text = self.problems(files, MINIMAL.replace("protein.pdb", "protein.txt"))
        assert ".pdb" in text

    def test_an_empty_input_file(self, files):
        (files / "mols.smi").write_text("")
        assert "empty" in self.problems(files, MINIMAL)

    def test_invalid_yaml(self, files):
        assert "not valid YAML" in self.problems(files, "receptor: [unclosed\n")

    def test_a_file_that_is_not_key_value_pairs(self, files):
        assert "key: value" in self.problems(files, "- just\n- a\n- list\n")

    def test_a_missing_run_file(self, tmp_path):
        with pytest.raises(vs_runfile.RunFileError, match="Cannot read"):
            vs_runfile.load(str(tmp_path / "nope.yaml"))

    def test_job_names_with_odd_characters(self, files):
        assert "letters, digits" in self.problems(files, MINIMAL.replace("mode: save", "mode: save\n  job_name: 'my run!'"))


class TestWriting:
    VALUES = {"receptor": "/data/p.pdb", "reference_ligand": "/data/l.sdf", "ligands": "/data/m.smi", "output": "/runs/a",
              "docking": {"program": "qvina", "exhaustiveness": 16, "box_size": 22.0, "rescore": "none",
                          "ionisation": "ph", "ligand_ph": 7.4},
              "run": {"mode": "submit", "job_name": "j", "cpus": 8, "time": "2:00:00", "partition": "ncpu"}}

    def test_what_is_written_can_be_read_back(self, files):
        values = dict(self.VALUES, receptor=str(files / "protein.pdb"), reference_ligand=str(files / "ref.sdf"),
                      ligands=str(files / "mols.smi"), output=str(files / "out"))
        path = files / "written.yaml"
        path.write_text(vs_runfile.render(values, "ddsuite vs"))
        cfg = vs_runfile.load(str(path))
        assert cfg["docking"]["exhaustiveness"] == 16 and cfg["run"]["cpus"] == 8
        assert cfg["run"]["time"] == "2:00:00"

    def test_times_are_always_quoted(self):
        text = vs_runfile.render(self.VALUES, "vs")
        assert 'time: "2:00:00"' in text
        assert yaml.safe_load(text)["run"]["time"] == "2:00:00"

    def test_unset_values_are_left_out(self):
        values = dict(self.VALUES, docking={"program": "qvina", "ligand_ph": None})
        assert "ligand_ph" not in vs_runfile.render(values, "vs")

    def test_the_command_to_repeat_the_run_is_in_the_header(self):
        assert "ddsuite vs <this file>" in vs_runfile.render(self.VALUES, "ddsuite vs")

    def test_the_structure_route_is_written_with_its_choices(self):
        values = {"structure": "/d/c.pdb", "template_molecule": "AQ4 A999", "chains": ["A"], "keep_molecules": [],
                  "ligands": "/d/m.smi", "output": "/r", "protonation": "pdb2pqr", "ph": 7.4,
                  "docking": {}, "run": {"mode": "save"}}
        data = yaml.safe_load(vs_runfile.render(values, "vs"))
        assert data["template_molecule"] == "AQ4 A999" and data["keep_molecules"] == [] and data["protonation"] == "pdb2pqr"


class TestExample:
    def test_it_is_valid_yaml_that_uses_only_known_keys(self):
        data = yaml.safe_load(vs_runfile.EXAMPLE)
        assert set(data) <= set(vs_runfile.TOP_KEYS)
        assert set(data["docking"]) <= set(vs_runfile.DOCKING_KEYS)
        assert set(data["run"]) <= set(vs_runfile.RUN_KEYS)

    def test_every_documented_key_appears_in_the_example(self):
        """A key that exists but is not in the example is a key nobody will find."""
        for key in vs_runfile.DOCKING_KEYS + vs_runfile.RUN_KEYS:
            assert f"{key}:" in vs_runfile.EXAMPLE, key

    def test_the_example_states_the_real_default_exhaustiveness(self):
        import vs_dock

        assert f"exhaustiveness: {vs_dock.DEFAULT_EXHAUSTIVENESS}" in vs_runfile.EXAMPLE
