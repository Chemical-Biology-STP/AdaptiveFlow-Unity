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


# --------------------------------------------------------------------------
# Staged screening
# --------------------------------------------------------------------------

STAGED = MINIMAL.replace("run:\n  mode: save\n", "screen: staged\nrun:\n  mode: save\n  cpus: 500\n")


def problems_of(files, body):
    with pytest.raises(vs_runfile.RunFileError) as info:
        vs_runfile.load(write(files, body))
    return str(info.value)


class TestStaged:
    def test_a_file_without_screen_is_a_full_screen(self, files):
        cfg = vs_runfile.load(write(files, MINIMAL))
        assert cfg["screen"] == "full" and cfg["staged"] == {}

    def test_screen_staged_with_its_settings(self, files):
        cfg = vs_runfile.load(write(files, STAGED + "staged:\n  max_stereoisomers: 3\n  max_tautomers: 2\n"))
        assert cfg["screen"] == "staged" and cfg["staged"] == {"max_stereoisomers": 3, "max_tautomers": 2}
        assert cfg["run"]["cpus"] == 500

    def test_a_staged_run_may_use_up_to_the_cluster_limit_of_cpus(self, files):
        assert vs_runfile.load(write(files, STAGED.replace("cpus: 500", "cpus: 3000")))["run"]["cpus"] == 3000
        assert "at most 3000" in problems_of(files, STAGED.replace("cpus: 500", "cpus: 3001"))

    def test_a_full_screen_may_not(self, files):
        assert "at most 256" in problems_of(files, MINIMAL + "  cpus: 300\n")

    def test_the_staged_section_needs_screen_staged(self, files):
        assert "only applies with 'screen: staged'" in problems_of(files, MINIMAL + "staged:\n  max_tautomers: 2\n")

    @pytest.mark.parametrize("key", ["num_modes: 9", "rescore: nnscore2", "ionisation: keep"])
    def test_docking_settings_that_a_staged_screen_does_not_use_are_refused(self, files, key):
        assert "does not apply to the staged screen" in problems_of(files, STAGED + f"docking:\n  {key}\n")

    def test_a_staged_screen_cannot_run_in_the_terminal(self, files):
        assert "runs on the cluster" in problems_of(files, STAGED.replace("mode: save", "mode: run"))

    @pytest.mark.parametrize("line", ['time: "2:00:00"', "watch: true"])
    def test_run_settings_that_a_staged_screen_sets_itself_are_refused(self, files, line):
        assert "does not apply to the staged screen" in problems_of(files, STAGED + f"  {line}\n")

    @pytest.mark.parametrize("value", [0, 21, 1.5, "many"])
    def test_forms_per_molecule_must_be_a_whole_number_from_1_to_20(self, files, value):
        assert "max_stereoisomers" in problems_of(files, STAGED + f"staged:\n  max_stereoisomers: {value}\n")

    def test_a_misspelt_staged_key_is_caught_with_a_hint(self, files):
        assert "Did you mean 'max_tautomers'" in problems_of(files, STAGED + "staged:\n  max_tautomer: 2\n")

    def test_screen_must_be_full_or_staged(self, files):
        assert "must be one of: full, staged" in problems_of(files, MINIMAL.replace("run:", "screen: partial\nrun:"))

    def test_the_example_file_describes_the_staged_mode(self):
        assert "screen: staged" in vs_runfile.EXAMPLE and "max_stereoisomers" in vs_runfile.EXAMPLE

    def test_render_writes_screen_only_when_it_is_staged(self):
        base = {"ligands": "l.smi", "output": "o", "receptor": "r.pdb", "box_center": [1, 2, 3],
                "run": {"mode": "save"}}
        assert "\nscreen:" not in vs_runfile.render(base)
        assert "screen: staged" in vs_runfile.render({**base, "screen": "staged", "staged": {"max_tautomers": 2}})


class TestStagedTranche:
    TRANCHE = STAGED + "staged:\n  strategy: tranche\n"

    def test_the_tranche_settings_are_read(self, files):
        body = self.TRANCHE + "  representatives: 4\n  primary_size: 50000\n  tranche_properties: [mw, tpsa]\n"
        cfg = vs_runfile.load(write(files, body))
        assert cfg["staged"] == {"strategy": "tranche", "representatives": 4, "primary_size": 50000,
                                 "tranche_properties": ["mw", "tpsa"]}

    def test_the_tranche_strategy_alone_uses_the_defaults(self, files):
        assert vs_runfile.load(write(files, self.TRANCHE))["staged"] == {"strategy": "tranche"}

    def test_strategy_all_is_the_default_and_is_accepted(self, files):
        assert vs_runfile.load(write(files, STAGED + "staged:\n  strategy: all\n"))["staged"] == {"strategy": "all"}

    @pytest.mark.parametrize("line", ["representatives: 3", "primary_size: 100", "tranche_properties: [mw]"])
    def test_tranche_settings_need_the_tranche_strategy(self, files, line):
        assert "only applies with 'strategy: tranche'" in problems_of(files, STAGED + f"staged:\n  {line}\n")

    def test_the_strategy_must_be_known(self, files):
        assert "must be one of: all, tranche" in problems_of(files, STAGED + "staged:\n  strategy: best\n")

    @pytest.mark.parametrize("value", [0, 11, 1.5, "x"])
    def test_representatives_are_one_to_ten(self, files, value):
        assert "staged.representatives" in problems_of(files, self.TRANCHE + f"  representatives: {value}\n")

    @pytest.mark.parametrize("value", [0, -1, 2.5, "all"])
    def test_the_primary_size_is_a_positive_whole_number(self, files, value):
        assert "staged.primary_size" in problems_of(files, self.TRANCHE + f"  primary_size: {value}\n")

    @pytest.mark.parametrize("value", ["[mw, colour]", "[mw, mw]", "[]", "5"])
    def test_the_tranche_properties_must_be_known_and_different(self, files, value):
        assert "staged.tranche_properties" in problems_of(files, self.TRANCHE + f"  tranche_properties: {value}\n")

    def test_the_properties_may_be_written_as_one_text(self, files):
        cfg = vs_runfile.load(write(files, self.TRANCHE + "  tranche_properties: mw logp\n"))
        assert cfg["staged"]["tranche_properties"] == ["mw", "logp"]

    def test_the_example_file_describes_the_tranche_strategy(self):
        assert "strategy: all" in vs_runfile.EXAMPLE and "representatives" in vs_runfile.EXAMPLE
        assert "primary_size" in vs_runfile.EXAMPLE

    def test_render_writes_the_tranche_settings_and_they_load_again(self, files):
        values = {"ligands": "l.smi", "output": "o", "receptor": "r.pdb", "box_center": [1, 2, 3], "screen": "staged",
                  "staged": {"strategy": "tranche", "representatives": 2, "primary_size": 10,
                             "tranche_properties": ["mw", "hbd"]},
                  "run": {"mode": "save", "cpus": 64}}
        text = vs_runfile.render(values)
        assert "strategy: tranche" in text and "representatives: 2" in text and "tranche_properties" in text


class TestStagedFilter:
    BASE = STAGED + "staged:\n  strategy: tranche\n"

    def test_the_filter_settings_are_read(self, files):
        cfg = vs_runfile.load(write(files, self.BASE + "  ml_filter: true\n  ml_cutoff: 0.7\n"))
        assert cfg["staged"]["ml_filter"] is True and cfg["staged"]["ml_cutoff"] == 0.7

    def test_the_filter_may_be_switched_off(self, files):
        assert vs_runfile.load(write(files, self.BASE + "  ml_filter: false\n"))["staged"]["ml_filter"] is False

    @pytest.mark.parametrize("value", ["yes please", "1", "maybe"])
    def test_the_filter_must_be_true_or_false(self, files, value):
        assert "staged.ml_filter" in problems_of(files, self.BASE + f"  ml_filter: {value}\n")

    @pytest.mark.parametrize("value", [0.01, 0.96, 1, "high"])
    def test_the_cutoff_must_be_between_005_and_095(self, files, value):
        assert "staged.ml_cutoff" in problems_of(files, self.BASE + f"  ml_filter: true\n  ml_cutoff: {value}\n")

    def test_a_cutoff_needs_the_filter(self, files):
        assert "only applies with 'ml_filter: true'" in problems_of(files, self.BASE + "  ml_cutoff: 0.6\n")

    @pytest.mark.parametrize("line", ["ml_filter: true", "ml_cutoff: 0.6"])
    def test_the_filter_needs_the_tranche_strategy(self, files, line):
        assert "only applies with 'strategy: tranche'" in problems_of(files, STAGED + f"staged:\n  {line}\n")

    def test_the_example_file_describes_the_filter(self):
        assert "ml_filter" in vs_runfile.EXAMPLE and "ml_cutoff" in vs_runfile.EXAMPLE
