#!/usr/bin/env python3
"""Launcher for the virtual screening workflow (VF Unity).

The workflow code expects to run inside a folder that holds ./config, ./executables and ./config.txt, and it writes
./ligands, ./outputs and the result tables into that same folder. A tool installed in the Drug Discovery Suite is
shared and started from the user's own folder, so this launcher builds a separate run folder for every run:

    <run folder>/config.txt       copy of the user's config file, with the receptor line pointing at ./config/
    <run folder>/config/          links to the helper files shipped here, plus a link to the user's receptor
    <run folder>/executables      link to the docking programs shipped here
    <run folder>/ligands, outputs, docking_output.csv ...   written by the workflow

and then runs the unchanged workflow (run_vf_unity.py) inside it. The user's files and the installed tool are never
written to.

Start it with:  ddsuite vs config.txt   (or, on its own:  pixi run python vs_cli.py config.txt)
"""

import argparse
import os
import shutil
import subprocess
import sys

REPO_DIR = os.path.dirname(os.path.abspath(__file__))

# The lab's launcher (ddsuite) sets DDSUITE_RUN_COMMAND, e.g. "ddsuite vs". It is used in messages that tell the
# user how to start this tool; on its own the tool is started with `pixi run python vs_cli.py`.
RUN_COMMAND = os.environ.get("DDSUITE_RUN_COMMAND") or "pixi run python vs_cli.py"

SHIPPED_CONFIG = os.path.join(REPO_DIR, "config.txt")   # annotated example, also used by --example
SHIPPED_CONFIG_DIR = os.path.join(REPO_DIR, "config")
SHIPPED_EXECUTABLES = os.path.join(REPO_DIR, "executables")
WORKFLOW_SCRIPT = os.path.join(REPO_DIR, "run_vf_unity.py")

DEFAULT_RUN_DIR = "vs_run"
MARKER = ".vs_run"          # written into every run folder this launcher makes; --force only deletes folders with it
IGNORED_CONFIG_FILES = {".DS_Store"}
REQUIRED_KEYS = ("program_choice", "center_x", "center_y", "center_z", "size_x", "size_y", "size_z",
                 "exhaustiveness", "smi", "is_selfies", "is_peptide", "receptor")
NUMERIC_KEYS = ("center_x", "center_y", "center_z", "size_x", "size_y", "size_z")


class SetupError(Exception):
    """A problem with the user's input, reported without a traceback."""


def read_config(path):
    """Return (lines, settings) of a config file, read the way run_vf_unity.read_config_file reads it."""
    try:
        with open(path) as fh:
            lines = fh.read().splitlines()
    except OSError as exc:
        raise SetupError(f"Cannot read the config file {path}: {exc.strerror}.")
    settings = {}
    for line in lines:
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, value = text.split("=", 1)
        settings[key.strip()] = value.strip()
    return lines, settings


def check_settings(settings):
    """Problems with the settings, as a list of messages (empty if fine)."""
    problems = [f"'{key}' is missing." for key in REQUIRED_KEYS if not settings.get(key)]
    for key in NUMERIC_KEYS:
        if settings.get(key):
            try:
                float(settings[key])
            except ValueError:
                problems.append(f"'{key}' must be a number, not '{settings[key]}'.")
    if settings.get("exhaustiveness"):
        try:
            int(settings["exhaustiveness"])
        except ValueError:
            problems.append(f"'exhaustiveness' must be a whole number, not '{settings['exhaustiveness']}'.")
    for key in ("is_selfies", "is_peptide"):
        if settings.get(key) and settings[key] not in ("True", "False"):
            problems.append(f"'{key}' must be True or False, not '{settings[key]}'.")
    return problems


def program_notes(program_choice):
    """Warnings about the chosen docking program (and scoring function, written as program+scoring)."""
    notes = []
    program = program_choice.split("+")[0]
    if program and not os.path.isfile(os.path.join(SHIPPED_EXECUTABLES, program)):
        notes.append(f"No program named '{program}' is shipped with this tool. That is expected for programs that "
                     "are installed separately (for example Glide or GOLD); otherwise the choice is misspelled "
                     "(see the list at the top of config.txt).")
    return notes


def resolve_receptor(value, config_dir):
    """Find the receptor file named in the config. Relative paths are tried against the config file's folder, the
    current folder and this tool's own folder (which makes './config/5wiu_test.pdbqt' find the shipped example)."""
    value = os.path.expanduser(os.path.expandvars(value))
    candidates = [value] if os.path.isabs(value) else [os.path.join(base, value)
                                                       for base in (config_dir, os.getcwd(), REPO_DIR)]
    for candidate in candidates:
        if os.path.isfile(candidate):
            path = os.path.abspath(candidate)
            if os.path.getsize(path) == 0:
                raise SetupError(f"The receptor file is empty: {path}")
            return path
    raise SetupError(f"Receptor file not found: {value}\n"
                     "  Give its path in the config file (relative to the config file, or absolute).")


def check_path_text(label, path):
    # The workflow builds shell command lines without quoting, so spaces would break them.
    if any(ch.isspace() for ch in path):
        raise SetupError(f"{label} contains a space, which the workflow cannot handle: {path}")


def prepare_run_dir(run_dir, config_lines, receptor, force):
    """Create the run folder as described in the module docstring."""
    if os.path.lexists(run_dir):
        if not force:
            raise SetupError(f"{run_dir} already exists. Choose another folder with --output, or add --force to "
                             "replace it (only a folder made by this tool is replaced).")
        if not os.path.isfile(os.path.join(run_dir, MARKER)):
            raise SetupError(f"{run_dir} was not made by this tool, so it will not be replaced. "
                             "Choose another folder with --output.")
        shutil.rmtree(run_dir)

    receptor_name = os.path.basename(receptor)
    shipped = {n for n in os.listdir(SHIPPED_CONFIG_DIR) if n not in IGNORED_CONFIG_FILES}
    clash = os.path.join(SHIPPED_CONFIG_DIR, receptor_name)
    if receptor_name in shipped and os.path.realpath(receptor) != os.path.realpath(clash):
        raise SetupError(f"The receptor file name '{receptor_name}' is also the name of a helper file of this tool. "
                         "Rename your receptor file.")

    os.makedirs(os.path.join(run_dir, "config"))
    for name in sorted(shipped):
        os.symlink(os.path.join(SHIPPED_CONFIG_DIR, name), os.path.join(run_dir, "config", name))
    if receptor_name not in shipped:
        os.symlink(receptor, os.path.join(run_dir, "config", receptor_name))
    os.symlink(SHIPPED_EXECUTABLES, os.path.join(run_dir, "executables"))

    with open(os.path.join(run_dir, "config.txt"), "w") as fh:
        for line in config_lines:
            if line.strip().startswith("receptor") and line.split("=", 1)[0].strip() == "receptor":
                line = f"receptor=./config/{receptor_name}"
            fh.write(line + "\n")
    with open(os.path.join(run_dir, MARKER), "w") as fh:
        fh.write("Made by vs_cli.py; safe to replace with --force.\n")


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog=RUN_COMMAND,
        description="Run the virtual screening workflow (docking and optional re-scoring of a molecule) in a new "
                    "run folder, using the settings in a config file.")
    parser.add_argument("config", nargs="?", default="config.txt",
                        help="config file with the docking settings (default: ./config.txt)")
    parser.add_argument("-o", "--output", default=DEFAULT_RUN_DIR,
                        help=f"run folder to create; results are written inside it (default: ./{DEFAULT_RUN_DIR})")
    parser.add_argument("--force", action="store_true",
                        help="replace the run folder if an earlier run of this tool made it")
    parser.add_argument("--check", action="store_true",
                        help="check the config file and the receptor, then stop without running anything")
    parser.add_argument("--example", action="store_true",
                        help="print an annotated example config file and exit "
                             f"(save it with: {RUN_COMMAND} --example > config.txt)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)

    if args.example:
        with open(SHIPPED_CONFIG) as fh:
            sys.stdout.write(fh.read())
        return 0

    try:
        config_path = os.path.abspath(args.config)
        config_lines, settings = read_config(config_path)
        problems = check_settings(settings)
        if problems:
            raise SetupError(f"Problems in {args.config}:\n" + "\n".join(f"  - {p}" for p in problems))
        receptor = resolve_receptor(settings["receptor"], os.path.dirname(config_path))
        run_dir = os.path.abspath(args.output)
        check_path_text("The run folder path", run_dir)
        check_path_text("The receptor path", receptor)
        for note in program_notes(settings["program_choice"]):
            print(f"  [!]  {note}")

        print(f"Config:    {config_path}")
        print(f"Program:   {settings['program_choice']}")
        print(f"Receptor:  {receptor}")
        print(f"Run folder: {run_dir}")
        if args.check:
            print("Checks passed; nothing was run (--check).")
            return 0
        prepare_run_dir(run_dir, config_lines, receptor, args.force)
    except SetupError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print("\nRunning the workflow...\n")
    result = subprocess.run([sys.executable, WORKFLOW_SCRIPT], cwd=run_dir, check=False)

    print()
    results_csv = os.path.join(run_dir, "docking_output.csv")
    if result.returncode != 0:
        print(f"The workflow stopped with an error (exit code {result.returncode}). "
              f"Files so far are in {run_dir}.", file=sys.stderr)
        return result.returncode
    print(f"Finished. Results are in {run_dir}:")
    print(f"  docking_output.csv   {'(written)' if os.path.isfile(results_csv) else '(not written)'}")
    print("  outputs/             docked poses")
    print("  ligands/             prepared 3D ligands")
    return 0


if __name__ == "__main__":
    sys.exit(main())
