"""The command-line entry point (vs_cli) and classic mode (vs_classic)."""
import os

import pytest
import yaml

import vs_cli
import vs_classic
import vs_dock
import vs_runfile


# --------------------------------------------------------------------------
# The dispatcher
# --------------------------------------------------------------------------


class TestDispatcher:
    def test_example_prints_a_run_file_that_the_run_file_reader_understands(self, capsys):
        assert vs_cli.main(["--example"]) == 0
        data = yaml.safe_load(capsys.readouterr().out)
        assert set(data) <= set(vs_runfile.TOP_KEYS)

    def test_example_config_prints_the_classic_config_file(self, capsys):
        assert vs_cli.main(["--example-config"]) == 0
        out = capsys.readouterr().out
        assert "program_choice=" in out and "receptor=" in out

    def test_check_needs_a_run_file(self, capsys):
        assert vs_cli.main(["--check"]) == 2
        assert "--check needs a run file" in capsys.readouterr().err

    def test_output_and_force_belong_to_classic_mode(self, capsys, tmp_path):
        run = tmp_path / "run.yaml"
        run.write_text("x: 1\n")
        assert vs_cli.main([str(run), "-o", "x"]) == 2
        assert "classic mode" in capsys.readouterr().err
        assert vs_cli.main([str(run), "--force"]) == 2

    def test_without_a_terminal_the_assistant_explains_what_to_do(self, monkeypatch, capsys):
        import vs_wizard

        monkeypatch.setattr("sys.stdin", open(os.devnull))
        monkeypatch.setattr(vs_wizard, "check_environment", lambda: None)
        assert vs_cli.main([]) == 2
        err = capsys.readouterr().err
        assert "needs a terminal" in err and "run.yaml" in err

    def test_a_bad_run_file_is_reported_and_nothing_runs(self, tmp_path, capsys):
        run = tmp_path / "run.yaml"
        run.write_text("receptor: missing.pdb\n")
        with pytest.raises(SystemExit) as excinfo:
            vs_cli.main([str(run), "--check"])
        assert excinfo.value.code == 2
        out = capsys.readouterr().out
        assert "Problems in" in out and "Nothing was run" in out

    def test_a_yaml_extension_selects_run_file_mode_and_anything_else_classic_mode(self, monkeypatch):
        calls = []
        import vs_wizard

        monkeypatch.setattr(vs_wizard, "check_environment", lambda: None)
        monkeypatch.setattr(vs_wizard, "run_from_file", lambda path, check: calls.append(("run file", path, check)))
        monkeypatch.setattr(vs_classic, "main", lambda argv: calls.append(("classic", argv)) or 0)
        vs_cli.main(["a.yaml", "--check"])
        vs_cli.main(["b.yml"])
        vs_cli.main(["config.txt"])
        assert calls[0] == ("run file", "a.yaml", True) and calls[1] == ("run file", "b.yml", False)
        assert calls[2][0] == "classic" and calls[2][1][0] == "config.txt"

    def test_the_help_names_the_command_the_user_typed(self, capsys, monkeypatch):
        with pytest.raises(SystemExit):
            vs_cli.main(["--help"])
        assert "run" in capsys.readouterr().out.lower()


# --------------------------------------------------------------------------
# Classic mode
# --------------------------------------------------------------------------

CONFIG = """\
# a comment
program_choice=qvina
center_x=1.5
center_y=-2
center_z=3
size_x=20
size_y=20
size_z=20
exhaustiveness=4
smi=CCO
is_selfies=False
is_peptide=False
receptor=./config/5wiu_test.pdbqt
"""


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "config.txt"
    path.write_text(CONFIG)
    return path


class TestClassicConfig:
    def test_a_valid_file_has_no_problems(self, config):
        lines, settings = vs_classic.read_config(str(config))
        assert vs_classic.check_settings(settings) == []
        assert settings["program_choice"] == "qvina" and settings["smi"] == "CCO"

    def test_comments_and_blank_lines_are_ignored(self, config):
        _, settings = vs_classic.read_config(str(config))
        assert "# a comment" not in settings

    @pytest.mark.parametrize("old,new,message", [
        ("size_x=20", "size_x=abc", "'size_x' must be a number"),
        ("is_selfies=False", "is_selfies=maybe", "must be True or False"),
        ("exhaustiveness=4", "exhaustiveness=4.5", "whole number"),
        ("program_choice=qvina", "", "'program_choice' is missing"),
    ])
    def test_problems_are_named(self, config, old, new, message):
        config.write_text(CONFIG.replace(old, new))
        _, settings = vs_classic.read_config(str(config))
        assert any(message in p for p in vs_classic.check_settings(settings))

    def test_an_unreadable_file(self, tmp_path):
        with pytest.raises(vs_classic.SetupError, match="Cannot read"):
            vs_classic.read_config(str(tmp_path / "nope.txt"))

    def test_a_program_that_is_not_shipped_is_noted_not_refused(self):
        assert vs_classic.program_notes("qvina") == []
        assert vs_classic.program_notes("qvina+nnscore2") == []
        assert any("GlideSP" in n for n in vs_classic.program_notes("GlideSP"))


class TestClassicRunFolder:
    def test_the_shipped_receptor_path_resolves(self):
        path = vs_classic.resolve_receptor("./config/5wiu_test.pdbqt", "/somewhere/else")
        assert os.path.isfile(path)

    def test_a_receptor_next_to_the_config_file(self, tmp_path):
        (tmp_path / "mine.pdbqt").write_text("ATOM\n")
        assert vs_classic.resolve_receptor("mine.pdbqt", str(tmp_path)) == str(tmp_path / "mine.pdbqt")

    def test_a_missing_receptor(self, tmp_path):
        with pytest.raises(vs_classic.SetupError, match="not found"):
            vs_classic.resolve_receptor("nope.pdbqt", str(tmp_path))

    def test_an_empty_receptor(self, tmp_path):
        (tmp_path / "e.pdbqt").write_text("")
        with pytest.raises(vs_classic.SetupError, match="empty"):
            vs_classic.resolve_receptor("e.pdbqt", str(tmp_path))

    def test_spaces_in_a_path_are_refused_because_the_workflow_does_not_quote(self):
        with pytest.raises(vs_classic.SetupError, match="space"):
            vs_classic.check_path_text("The run folder path", "/my runs/a")

    def test_the_run_folder_links_the_shipped_files_and_rewrites_the_receptor_line(self, tmp_path, config):
        receptor = tmp_path / "mine.pdbqt"
        receptor.write_text("ATOM\n")
        lines, _ = vs_classic.read_config(str(config))
        run = tmp_path / "run"
        vs_classic.prepare_run_dir(str(run), lines, str(receptor), force=False)
        assert os.path.islink(run / "executables") and os.path.isfile(run / "executables" / "qvina")
        assert os.path.islink(run / "config" / "mine.pdbqt")
        assert "receptor=./config/mine.pdbqt" in (run / "config.txt").read_text()
        assert (run / vs_classic.MARKER).is_file()

    def test_a_run_folder_is_not_replaced_without_force(self, tmp_path, config):
        receptor = tmp_path / "r.pdbqt"
        receptor.write_text("ATOM\n")
        lines, _ = vs_classic.read_config(str(config))
        vs_classic.prepare_run_dir(str(tmp_path / "run"), lines, str(receptor), force=False)
        with pytest.raises(vs_classic.SetupError, match="already exists"):
            vs_classic.prepare_run_dir(str(tmp_path / "run"), lines, str(receptor), force=False)

    def test_force_replaces_a_folder_this_tool_made(self, tmp_path, config):
        receptor = tmp_path / "r.pdbqt"
        receptor.write_text("ATOM\n")
        lines, _ = vs_classic.read_config(str(config))
        run = tmp_path / "run"
        vs_classic.prepare_run_dir(str(run), lines, str(receptor), force=False)
        (run / "old_result.txt").write_text("x")
        vs_classic.prepare_run_dir(str(run), lines, str(receptor), force=True)
        assert not (run / "old_result.txt").exists()

    def test_force_never_deletes_a_folder_this_tool_did_not_make(self, tmp_path, config):
        receptor = tmp_path / "r.pdbqt"
        receptor.write_text("ATOM\n")
        lines, _ = vs_classic.read_config(str(config))
        foreign = tmp_path / "foreign"
        foreign.mkdir()
        (foreign / "precious.txt").write_text("keep me")
        with pytest.raises(vs_classic.SetupError, match="not made by this tool"):
            vs_classic.prepare_run_dir(str(foreign), lines, str(receptor), force=True)
        assert (foreign / "precious.txt").read_text() == "keep me"

    def test_a_receptor_named_like_a_helper_file_is_refused(self, tmp_path, config):
        clash = tmp_path / "seed.inp"
        clash.write_text("x")
        lines, _ = vs_classic.read_config(str(config))
        with pytest.raises(vs_classic.SetupError, match="helper file"):
            vs_classic.prepare_run_dir(str(tmp_path / "run"), lines, str(clash), force=False)

    def test_check_stops_before_creating_anything(self, tmp_path, config, capsys):
        os.chdir(tmp_path)
        assert vs_classic.main([str(config), "-o", "newrun", "--check"]) == 0
        assert "Checks passed" in capsys.readouterr().out and not (tmp_path / "newrun").exists()


# --------------------------------------------------------------------------
# Things the whole tool relies on
# --------------------------------------------------------------------------


class TestShippedFiles:
    def test_every_program_the_assistant_offers_is_in_the_executables_folder(self):
        for name in vs_dock.PROGRAMS:
            assert os.access(os.path.join(vs_dock.EXECUTABLES, name), os.X_OK), name

    def test_nnscore2_and_the_vina_it_needs_are_shipped(self):
        assert os.path.isfile(vs_dock.NNSCORE2) and os.access(vs_dock.NNSCORE2_VINA, os.X_OK)

    def test_the_example_config_is_the_one_classic_mode_prints(self):
        text = open(vs_classic.SHIPPED_CONFIG).read()
        assert "program_choice=qvina" in text

    def test_the_licence_notice_for_the_copied_modules_is_present(self):
        notice = open(os.path.join(vs_dock.REPO_DIR, "NOTICE.md")).read()
        assert "MIT License" in notice and "structure_prep.py" in notice and "protein_prep.py" in notice

    def test_the_task_names_in_pixi_toml_match_the_documented_ones(self):
        text = open(os.path.join(vs_dock.REPO_DIR, "pixi.toml")).read()
        for task in ("vs =", "test =", "test-all ="):
            assert task in text
