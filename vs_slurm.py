"""SLURM side of the assistant: the job script, a check that the account may use the partition, submitting a job,
and watching it until it ends. Nothing here asks the user anything.

Adapted from dock_cli.py of the anchored_docking_workflow repository (MIT License, see NOTICE.md), changed from GPU
to CPU jobs.
"""

import getpass
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass

DEFAULT_PARTITION = "ncpu"
MAX_CPUS_DEFAULT = 32               # ligands docked at the same time unless the user asks for more
MEM_PER_CPU = "2G"                  # each worker holds one ligand; measured use is far below this
MAX_TIME_MINUTES = 7 * 24 * 60      # the ncpu partition allows 7 days
MIN_TIME_MINUTES = 30
STARTUP_MINUTES = 10                # receptor preparation, reading the library, starting the workers
SAFETY_FACTOR = 2.0                 # estimates are from short test runs; busy nodes are slower
MAIL_DOMAIN = "crick.ac.uk"
SUBMIT_CHECK_SECONDS = 30
POLL_SECONDS = 10
MAX_WATCH_ERRORS = 6                # consecutive failed squeue calls before giving up watching
ACTIVE_STATES = {"PENDING", "RUNNING", "REQUEUED", "RESIZING", "SUSPENDED", "COMPLETING", "CONFIGURING"}

# Replies of the scheduler that mean "try again later" and say nothing about access.
SCHEDULER_UNAVAILABLE = ("unable to contact", "socket timed out", "connection refused", "slurm controller",
                         "temporarily")


def mail_address():
    """Where SLURM sends the notification: the current user name plus the lab's mail domain (None if unknown)."""
    try:
        user = getpass.getuser()
    except Exception:
        return None
    return f"{user}@{MAIL_DOMAIN}" if re.fullmatch(r"[A-Za-z0-9._-]+", user) else None


# seconds_per_ligand in vs_dock.PROGRAMS were measured at exhaustiveness 8 and scale linearly with it
MEASURED_AT_EXHAUSTIVENESS = 8


def estimate_minutes(n_ligands, seconds_per_ligand, exhaustiveness, cpus, rescore=False,
                     default_exhaustiveness=MEASURED_AT_EXHAUSTIVENESS):
    """A generous time limit, in minutes, for docking n_ligands on `cpus` CPUs at the same time."""
    per_ligand = seconds_per_ligand * exhaustiveness / default_exhaustiveness + (5 if rescore else 0) + 2
    work = n_ligands * per_ligand / max(1, min(cpus, n_ligands)) / 60
    return min(MAX_TIME_MINUTES, max(MIN_TIME_MINUTES, STARTUP_MINUTES + SAFETY_FACTOR * work))


def multiline_command(prefix, args):
    groups, current = [], []
    for token in args:
        if token.startswith("--") and current:
            groups.append(current)
            current = []
        current.append(token)
    groups.append(current)
    lines = [prefix] + ["    " + " ".join(shlex.quote(t) for t in g) for g in groups]
    return " \\\n".join(lines)


def build_script(jobname, logdir, partition, time_limit, cpus, engine_args, repo_dir):
    """The text of the job script. It calls the Python of the environment this assistant runs in directly, so the
    compute node needs neither pixi nor an activated environment."""
    mail = mail_address()
    lines = ["#!/bin/bash",
             f"#SBATCH --partition={partition}",
             f"#SBATCH --job-name={jobname}",
             f"#SBATCH --output={os.path.join(logdir, jobname)}_%j.log",
             f"#SBATCH --time={time_limit}",
             "#SBATCH --nodes=1",
             "#SBATCH --ntasks=1",
             f"#SBATCH --cpus-per-task={cpus}",
             f"#SBATCH --mem-per-cpu={MEM_PER_CPU}",
             *(["#SBATCH --mail-type=END,FAIL", f"#SBATCH --mail-user={mail}"] if mail else []),
             "",
             f"cd {shlex.quote(logdir)}",
             "",
             multiline_command(f"{shlex.quote(sys.executable)} -W ignore {shlex.quote(os.path.join(repo_dir, 'vs_dock.py'))}",
                               engine_args + ["--cpus", str(cpus)]),
             ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Access check
# --------------------------------------------------------------------------


@dataclass
class SubmitCheck:
    ok: object          # True, False, or None when it could not be found out
    reason: str = ""


def explain_submit_error(text, partition):
    """Say in plain words why Slurm refused a job, from its error text."""
    low = text.lower()
    if "invalid partition" in low:
        return f"the partition {partition} does not exist"
    if "invalid account" in low or "account/partition" in low:
        return f"your account cannot use the partition {partition}"
    if "invalid qos" in low or ("qos" in low and "not permitted" in low):
        return f"your account has no access to the quality of service that the partition {partition} needs"
    if "time limit" in low:
        return f"the time limit is longer than the partition {partition} allows"
    if "node configuration is not available" in low:
        return f"no node of the partition {partition} has the requested CPUs or memory"
    lines = [re.sub(r"^(sbatch: )?(error|allocation failure): ?", "", l.strip(), flags=re.I)
             for l in text.splitlines() if l.strip()]
    return lines[0] if lines else "the scheduler refused the job"


def check_submission(partition, cpus=1, time_limit="0:10:00"):
    """Ask Slurm whether this account could submit such a job (sbatch --test-only: nothing is queued)."""
    sbatch = shutil.which("sbatch")     # the real program; the interactive shell wrapper is not visible from here
    if sbatch is None:
        return SubmitCheck(None, "sbatch was not found")
    command = [sbatch, "--test-only", f"--partition={partition}", f"--cpus-per-task={cpus}",
               f"--mem-per-cpu={MEM_PER_CPU}", f"--time={time_limit}", "--wrap=true"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=SUBMIT_CHECK_SECONDS,
                                stdin=subprocess.DEVNULL, check=False)
    except (subprocess.TimeoutExpired, OSError):
        return SubmitCheck(None, "the scheduler did not answer")
    if result.returncode == 0:
        return SubmitCheck(True)
    text = result.stdout + result.stderr
    if any(word in text.lower() for word in SCHEDULER_UNAVAILABLE):
        return SubmitCheck(None, "the scheduler could not be reached")
    return SubmitCheck(False, explain_submit_error(text, partition))


# --------------------------------------------------------------------------
# Submitting and watching
# --------------------------------------------------------------------------


def submit(script_path, jobname):
    """Submit with sbatch. Returns the job id, or None if it was not submitted (the reason is printed)."""
    sbatch = shutil.which("sbatch")
    result = subprocess.run([sbatch, "--parsable", script_path], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"  sbatch reported an error: {(result.stderr or result.stdout).strip()}")
        return None
    job_id = result.stdout.strip().split(";")[0]
    return job_id if job_id.isdigit() else None


def job_state(job_id):
    """Ask SLURM about a job. Returns ("state", [state, run time]), ("gone", None) or ("error", message).

    "gone" means SLURM answered and the job is no longer in the queue. A failure to reach SLURM is an "error",
    never "gone", so a hiccup is not mistaken for a finished job.
    """
    try:
        result = subprocess.run(["squeue", "-h", "-j", job_id, "-o", "%T|%M"],
                                capture_output=True, text=True, check=False)
    except OSError as exc:
        return "error", str(exc)
    if result.returncode == 0:
        out = result.stdout.strip()
        return ("state", out.split("|")) if out else ("gone", None)
    if "invalid job id" in result.stderr.lower():
        return "gone", None
    return "error", result.stderr.strip() or f"squeue exited with code {result.returncode}"


def job_final_state(job_id):
    """(state, exit code) from the accounting database, or None if it is unavailable."""
    try:
        result = subprocess.run(["sacct", "-n", "-X", "-P", "-j", job_id, "-o", "State,ExitCode"],
                                capture_output=True, text=True, check=False)
    except OSError:
        return None
    lines = result.stdout.strip().splitlines() if result.returncode == 0 else []
    return tuple(lines[0].split("|")[:2]) if lines and "|" in lines[0] else None


def watch(job_id, say, warn):
    """Print the job's state until it ends. Returns True if it ended, False if watching stopped early."""
    say(f"\n  Watching job {job_id}. Press Ctrl+C to stop watching (the job keeps running).\n")
    last_state, last_print, errors = None, 0.0, 0
    final = None
    try:
        while True:
            kind, data = job_state(job_id)
            if kind == "error":
                errors += 1
                if errors >= MAX_WATCH_ERRORS:
                    warn(f"Could not reach SLURM to check the job ({data}). Stopped watching; "
                         "your job is not affected.", record=False)
                    return False
                time.sleep(POLL_SECONDS)
                continue
            errors = 0
            if kind == "state":
                now = time.time()
                if data[0] != last_state or now - last_print > 60:
                    label = {"PENDING": "waiting in the queue", "RUNNING": "running"}.get(data[0], data[0].lower())
                    say(f"  {time.strftime('%H:%M:%S')}  {label} (run time {data[1]})")
                    last_state, last_print = data[0], now
                time.sleep(POLL_SECONDS)
                continue
            final = job_final_state(job_id)       # "gone" from the queue; make sure it is really over
            if final and final[0].split()[0] in ACTIVE_STATES:
                time.sleep(POLL_SECONDS)
                continue
            break
    except KeyboardInterrupt:
        say("\n  Stopped watching. The job is still in the queue or running.")
        return False
    say(f"\n  Job finished: {final[0]} (exit code {final[1]})" if final else "\n  Job finished.")
    return True


def follow_up_lines(job_id, log):
    lines = ["Useful commands:",
             f"  squeue -u {getpass.getuser()}            show your queued and running jobs",
             f"  tail -f {log}   follow the job's log (Ctrl+C to stop)",
             f"  scancel {job_id}                         cancel the job"]
    if mail_address():
        lines.append(f"An email goes to {mail_address()} when the job ends or fails.")
    return lines
