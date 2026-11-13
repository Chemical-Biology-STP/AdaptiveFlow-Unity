"""The SLURM layer (vs_slurm) and the terminal helpers (vs_ui). The scheduler is replaced by a stand-in."""
import os
import subprocess
import sys

import pytest

import vs_slurm
import vs_ui


# --------------------------------------------------------------------------
# Time limits
# --------------------------------------------------------------------------


class TestTimes:
    @pytest.mark.parametrize("text,minutes", [
        ("30", 30), ("1:30", 1.5), ("2:00:00", 120), ("0:30:00", 30), ("1-00:00:00", 1440), ("1-12", 2160),
        ("7-00:00:00", 10080), ("0:00:30", 0.5),
    ])
    def test_slurm_time_formats_are_converted_to_minutes(self, text, minutes):
        assert vs_ui.time_to_minutes(text) == pytest.approx(minutes)

    @pytest.mark.parametrize("minutes,text", [(30, "0:30:00"), (90, "1:30:00"), (1440, "1-00:00:00"), (1501, "1-01:01:00"),
                                              (29.2, "0:30:00")])
    def test_minutes_are_formatted_for_slurm_rounding_up(self, minutes, text):
        assert vs_ui.fmt_minutes(minutes) == text

    @pytest.mark.parametrize("good", ["2:00:00", "30", "1-00:00:00", "0:30"])
    def test_valid_times_are_accepted(self, good):
        assert vs_ui.time_validator(good) == good

    @pytest.mark.parametrize("bad", ["soon", "2h", "", "1:2:3:4", "-5", "2:0:00"])
    def test_invalid_times_are_rejected(self, bad):
        with pytest.raises(ValueError):
            vs_ui.time_validator(bad)


class TestEstimates:
    def test_there_is_a_minimum(self):
        assert vs_slurm.estimate_minutes(1, 20, 16, 32) == vs_slurm.MIN_TIME_MINUTES

    def test_there_is_a_maximum_of_seven_days(self):
        assert vs_slurm.estimate_minutes(10 ** 9, 20, 16, 32) == vs_slurm.MAX_TIME_MINUTES == 7 * 24 * 60

    def test_more_cpus_finish_sooner(self):
        assert vs_slurm.estimate_minutes(100000, 20, 8, 256) < vs_slurm.estimate_minutes(100000, 20, 8, 32)

    def test_higher_exhaustiveness_takes_proportionally_longer(self):
        slow = vs_slurm.estimate_minutes(100000, 20, 16, 32)
        fast = vs_slurm.estimate_minutes(100000, 20, 8, 32)
        assert 1.8 < (slow - vs_slurm.STARTUP_MINUTES) / (fast - vs_slurm.STARTUP_MINUTES) < 2.2

    def test_rescoring_adds_time(self):
        assert vs_slurm.estimate_minutes(100000, 20, 8, 32, rescore=True) > vs_slurm.estimate_minutes(100000, 20, 8, 32)

    def test_cpus_beyond_the_number_of_molecules_do_not_help(self):
        assert vs_slurm.estimate_minutes(10, 20, 8, 10) == vs_slurm.estimate_minutes(10, 20, 8, 256)

    def test_the_estimate_includes_a_safety_margin(self):
        """100 molecules of 20 s on 10 CPUs is 200 s of work; the limit must be well above that."""
        assert vs_slurm.estimate_minutes(10000, 20, 8, 10) > 2 * (10000 * 22 / 10 / 60)


# --------------------------------------------------------------------------
# The job script
# --------------------------------------------------------------------------


class TestJobScript:
    ARGS = ["--receptor", "/r/input/receptor.pdb", "--ligands", "/r/input/m.smi", "--outdir", "/r/results",
            "--center", "1.000", "2.000", "3.000", "--program", "qvina"]

    def script(self, **kw):
        defaults = dict(jobname="vs_test", logdir="/r", partition="ncpu", time_limit="2:00:00", cpus=32,
                        engine_args=self.ARGS, repo_dir="/tool")
        defaults.update(kw)
        return vs_slurm.build_script(**defaults)

    def test_the_resources_requested(self):
        text = self.script()
        for line in ("#SBATCH --partition=ncpu", "#SBATCH --job-name=vs_test", "#SBATCH --time=2:00:00",
                     "#SBATCH --nodes=1", "#SBATCH --cpus-per-task=32", f"#SBATCH --mem-per-cpu={vs_slurm.MEM_PER_CPU}",
                     "#SBATCH --output=/r/vs_test_%j.log"):
            assert line in text.splitlines()

    def test_the_job_calls_the_engine_with_the_pythons_of_this_environment(self):
        """No pixi and no activation are needed on the compute node."""
        text = self.script()
        assert f"{sys.executable} -W ignore /tool/vs_dock.py" in text

    def test_the_cpu_count_is_passed_to_the_engine(self):
        assert "--cpus 32" in self.script(cpus=32)

    def test_every_argument_reaches_the_engine(self):
        text = self.script()
        for token in self.ARGS:
            assert token in text

    def test_it_starts_in_the_run_folder(self):
        assert "\ncd /r\n" in self.script()

    def test_paths_with_spaces_are_quoted(self):
        text = self.script(logdir="/my runs/a", engine_args=["--outdir", "/my runs/a/results"])
        assert "cd '/my runs/a'" in text and "'/my runs/a/results'" in text

    def test_mail_goes_to_the_user_when_the_job_ends_or_fails(self):
        text = self.script()
        assert "#SBATCH --mail-type=END,FAIL" in text and f"@{vs_slurm.MAIL_DOMAIN}" in text

    def test_it_is_a_bash_script(self):
        assert self.script().startswith("#!/bin/bash\n")

    def test_options_stay_together_with_their_values_when_the_command_is_wrapped(self):
        wrapped = vs_slurm.multiline_command("python x.py", ["--center", "1", "2", "3", "--size", "20"])
        assert "    --center 1 2 3 \\\n    --size 20" in wrapped


# --------------------------------------------------------------------------
# Whether the account may use a partition
# --------------------------------------------------------------------------


class FakeSbatch:
    """Stands in for `sbatch --test-only`: returns what the real one printed in each situation."""

    def __init__(self, returncode=0, output=""):
        self.returncode, self.output, self.calls = returncode, output, []

    def __call__(self, command, **kwargs):
        self.calls.append(command)
        return subprocess.CompletedProcess(command, self.returncode, stdout="", stderr=self.output)


@pytest.fixture
def real_check(monkeypatch):
    """The module with the original check_submission (conftest replaces it for every other test)."""
    from conftest import REAL_CHECK_SUBMISSION

    monkeypatch.setattr(vs_slurm, "check_submission", REAL_CHECK_SUBMISSION)
    monkeypatch.setattr(vs_slurm.shutil, "which", lambda name: "/usr/bin/sbatch")
    return vs_slurm


class TestAccessCheck:
    def test_accepted(self, real_check, monkeypatch):
        fake = FakeSbatch(0, "sbatch: Job 1 to start at ...")
        monkeypatch.setattr(real_check.subprocess, "run", fake)
        assert real_check.check_submission("ncpu", 32, "2:00:00").ok is True
        command = fake.calls[0]
        assert "--test-only" in command and "--partition=ncpu" in command and "--cpus-per-task=32" in command

    def test_nothing_is_queued_by_the_check(self, real_check, monkeypatch):
        fake = FakeSbatch(0)
        monkeypatch.setattr(real_check.subprocess, "run", fake)
        real_check.check_submission("ncpu", 4)
        assert "--test-only" in fake.calls[0]

    @pytest.mark.parametrize("message,expected", [
        ("sbatch: error: invalid partition specified: nope", "does not exist"),
        ("sbatch: error: Invalid account or account/partition combination specified", "cannot use the partition"),
        ("sbatch: error: Batch job submission failed: Requested time limit is invalid", "time limit"),
        ("sbatch: error: Batch job submission failed: Requested node configuration is not available", "CPUs or memory"),
    ])
    def test_refusals_are_explained_in_plain_words(self, real_check, monkeypatch, message, expected):
        monkeypatch.setattr(real_check.subprocess, "run", FakeSbatch(1, message))
        result = real_check.check_submission("nope", 4)
        assert result.ok is False and expected in result.reason

    def test_an_unknown_error_shows_the_schedulers_first_line(self, real_check, monkeypatch):
        monkeypatch.setattr(real_check.subprocess, "run", FakeSbatch(1, "sbatch: error: something odd happened\nmore"))
        assert real_check.check_submission("ncpu", 4).reason == "something odd happened"

    @pytest.mark.parametrize("message", [
        "sbatch: error: Unable to contact slurm controller (connect failure)",
        "slurm_load_jobs error: Socket timed out on send/recv operation",
    ])
    def test_a_scheduler_that_cannot_be_reached_is_not_mistaken_for_missing_access(self, real_check, monkeypatch, message):
        monkeypatch.setattr(real_check.subprocess, "run", FakeSbatch(1, message))
        assert real_check.check_submission("ncpu", 4).ok is None

    def test_a_timeout_means_unknown(self, real_check, monkeypatch):
        def slow(command, **kwargs):
            raise subprocess.TimeoutExpired(command, 30)

        monkeypatch.setattr(real_check.subprocess, "run", slow)
        assert real_check.check_submission("ncpu", 4).ok is None

    def test_no_sbatch_means_unknown(self, real_check, monkeypatch):
        monkeypatch.setattr(real_check.shutil, "which", lambda name: None)
        assert real_check.check_submission("ncpu", 4).ok is None


# --------------------------------------------------------------------------
# Submitting and watching
# --------------------------------------------------------------------------


class TestSubmit:
    def test_the_job_id_is_read_from_parsable_output(self, monkeypatch):
        monkeypatch.setattr(vs_slurm.shutil, "which", lambda name: "/usr/bin/sbatch")
        monkeypatch.setattr(vs_slurm.subprocess, "run",
                            lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="12345;ncpu\n", stderr=""))
        assert vs_slurm.submit("s.sh", "j") == "12345"

    def test_a_refusal_returns_none(self, monkeypatch, capsys):
        monkeypatch.setattr(vs_slurm.shutil, "which", lambda name: "/usr/bin/sbatch")
        monkeypatch.setattr(vs_slurm.subprocess, "run",
                            lambda *a, **k: subprocess.CompletedProcess(a, 1, stdout="", stderr="sbatch: error: no\n"))
        assert vs_slurm.submit("s.sh", "j") is None
        assert "no" in capsys.readouterr().out


class TestJobState:
    def fake_squeue(self, monkeypatch, returncode, stdout="", stderr=""):
        monkeypatch.setattr(vs_slurm.subprocess, "run",
                            lambda *a, **k: subprocess.CompletedProcess(a, returncode, stdout=stdout, stderr=stderr))

    def test_a_running_job(self, monkeypatch):
        self.fake_squeue(monkeypatch, 0, "RUNNING|3:21\n")
        assert vs_slurm.job_state("1") == ("state", ["RUNNING", "3:21"])

    def test_a_job_that_has_left_the_queue(self, monkeypatch):
        self.fake_squeue(monkeypatch, 0, "")
        assert vs_slurm.job_state("1") == ("gone", None)

    def test_an_unknown_job_id_means_gone(self, monkeypatch):
        self.fake_squeue(monkeypatch, 1, "", "slurm_load_jobs error: Invalid job id specified")
        assert vs_slurm.job_state("1")[0] == "gone"

    def test_a_scheduler_hiccup_is_an_error_never_gone(self, monkeypatch):
        """Otherwise a network blip would be reported as 'job finished'."""
        self.fake_squeue(monkeypatch, 1, "", "slurm_load_jobs error: Socket timed out")
        assert vs_slurm.job_state("1")[0] == "error"

    def test_the_final_state_comes_from_the_accounting_database(self, monkeypatch):
        self.fake_squeue(monkeypatch, 0, "FAILED|1:0\n")
        assert vs_slurm.job_final_state("1") == ("FAILED", "1:0")

    def test_no_accounting_means_none(self, monkeypatch):
        self.fake_squeue(monkeypatch, 1)
        assert vs_slurm.job_final_state("1") is None


class TestWatch:
    def test_it_reports_the_states_and_the_end(self, monkeypatch):
        states = iter([("state", ["PENDING", "0:00"]), ("state", ["RUNNING", "0:10"]), ("gone", None)])
        monkeypatch.setattr(vs_slurm, "job_state", lambda job_id: next(states))
        monkeypatch.setattr(vs_slurm, "job_final_state", lambda job_id: ("COMPLETED", "0:0"))
        monkeypatch.setattr(vs_slurm.time, "sleep", lambda s: None)
        said = []
        assert vs_slurm.watch("1", said.append, lambda *a, **k: None) is True
        text = "\n".join(said)
        assert "waiting in the queue" in text and "running" in text and "COMPLETED" in text

    def test_it_waits_while_accounting_still_shows_the_job_active(self, monkeypatch):
        finals = iter([("RUNNING", "0:0"), ("COMPLETING", "0:0"), ("COMPLETED", "0:0")])
        monkeypatch.setattr(vs_slurm, "job_state", lambda job_id: ("gone", None))
        monkeypatch.setattr(vs_slurm, "job_final_state", lambda job_id: next(finals))
        monkeypatch.setattr(vs_slurm.time, "sleep", lambda s: None)
        said = []
        assert vs_slurm.watch("1", said.append, lambda *a, **k: None) is True
        assert "COMPLETED" in "\n".join(said)

    def test_it_gives_up_after_repeated_scheduler_errors_without_touching_the_job(self, monkeypatch):
        monkeypatch.setattr(vs_slurm, "job_state", lambda job_id: ("error", "unreachable"))
        monkeypatch.setattr(vs_slurm.time, "sleep", lambda s: None)
        warned = []
        assert vs_slurm.watch("1", lambda *a: None, lambda msg, **k: warned.append(msg)) is False
        assert warned and "not affected" in warned[0]

    def test_ctrl_c_stops_watching_but_not_the_job(self, monkeypatch):
        def interrupt(job_id):
            raise KeyboardInterrupt

        monkeypatch.setattr(vs_slurm, "job_state", interrupt)
        said = []
        assert vs_slurm.watch("1", said.append, lambda *a, **k: None) is False
        assert "still in the queue or running" in "\n".join(said)

    def test_follow_up_commands(self):
        text = "\n".join(vs_slurm.follow_up_lines("99", "/r/log.log"))
        assert "scancel 99" in text and "tail -f /r/log.log" in text


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------


class TestPrompts:
    def test_enter_accepts_the_default(self, answers):
        answers([""])
        assert vs_ui.ask("Name", "abc") == "abc"

    def test_a_bad_answer_is_asked_again(self, answers):
        feed = answers(["x", "5"])
        assert vs_ui.ask_number("Number", None, int, 1, True, 9) == 5
        assert len(feed.asked) == 2

    def test_numbers_are_range_checked(self, answers):
        answers(["0", "99", "4"])
        assert vs_ui.ask_number("n", None, int, 1, True, 10) == 4

    def test_yes_and_no(self, answers):
        answers(["maybe", "y"])
        assert vs_ui.ask_yes_no("ok?", False) is True
        answers(["no"])
        assert vs_ui.ask_yes_no("ok?", True) is False

    def test_a_menu_returns_the_key_of_the_choice(self, answers):
        answers(["9", "2"])
        assert vs_ui.ask_menu([("a", "first"), ("b", "second")], 1) == "b"

    def test_a_required_answer_cannot_be_empty(self, answers):
        feed = answers(["", "name"])
        assert vs_ui.ask("Name") == "name" and len(feed.asked) == 2

    def test_file_validation(self, tmp_path):
        good = tmp_path / "a.pdb"
        good.write_text("x")
        empty = tmp_path / "e.pdb"
        empty.write_text("")
        validate = vs_ui.file_validator((".pdb",))
        assert validate(str(good)) == str(good)
        with pytest.raises(ValueError, match="not found"):
            validate(str(tmp_path / "missing.pdb"))
        with pytest.raises(ValueError, match="empty"):
            validate(str(empty))
        (tmp_path / "a.txt").write_text("x")
        with pytest.raises(ValueError, match="Expected"):
            validate(str(tmp_path / "a.txt"))

    def test_quotes_around_a_dragged_in_path_are_removed(self, tmp_path):
        path = tmp_path / "a.pdb"
        path.write_text("x")
        assert vs_ui.file_validator((".pdb",))(f"'{path}'") == str(path)
