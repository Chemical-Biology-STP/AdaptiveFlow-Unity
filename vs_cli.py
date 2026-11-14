#!/usr/bin/env python3
"""Virtual screening: dock a library of molecules into a protein with a Vina-family program and rank them.

    vs                    the interactive assistant: asks for the protein, the binding site, the molecules and
                          the settings, then submits the job to the cluster (or runs it)
    vs run.yaml           the same steps with all the answers in a run file (the assistant writes one for every run)
    vs run.yaml --check   check a run file and show what would be done, without running anything
    vs config.txt         classic mode: the original VF Unity workflow from a config file (many programs)
    vs --example          print an annotated example run file
    vs status <folder>    staged screens (AdaptiveFlow): where the run is;  vs results <folder>: the best molecules;
    vs advance <folder>   the next step of a staged run (the pipeline calls it itself; add --retry after a failure)
    vs redock <folder>    dock the best molecules of a finished staged run again and keep their poses

Start it with `ddsuite vs` (Drug Discovery Suite) or `pixi run python vs_cli.py` (standalone).
"""

import argparse
import os
import sys
import warnings

# Meeko imports ProDy, whose source has invalid escape sequences; Python warns about them whenever it has to compile
# the files (on a shared install nobody can write the compiled copies). The warnings mean nothing to users.
warnings.filterwarnings("ignore")

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_DIR)

RUN_COMMAND = os.environ.get("DDSUITE_RUN_COMMAND") or "pixi run python vs_cli.py"
RUN_FILE_EXTENSIONS = (".yaml", ".yml")
STAGED_COMMANDS = ("status", "advance", "results", "redock")      # follow and continue a staged screen (vs_staged.py)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog=RUN_COMMAND,
        description="Dock a library of molecules into a protein with a Vina-family program and rank them by score. "
                    f"Run it with no arguments for the interactive assistant, or give it a run file: "
                    f"`{RUN_COMMAND} run.yaml`.")
    parser.add_argument("file", nargs="?",
                        help="a run file (.yaml) holding all the answers, or a classic config file (config.txt). "
                             f"Staged screens also have the commands: {', '.join(STAGED_COMMANDS)} <run folder>")
    parser.add_argument("--check", action="store_true",
                        help="check the file and show what would be done, without running anything")
    parser.add_argument("--example", action="store_true", help="print an annotated example run file and exit")
    parser.add_argument("--example-config", action="store_true",
                        help="print an annotated example classic config file and exit")
    parser.add_argument("-o", "--output", default=None,
                        help="classic mode only: run folder to create (default: ./vs_run)")
    parser.add_argument("--force", action="store_true",
                        help="classic mode only: replace the run folder if an earlier run of this tool made it")
    return parser.parse_args(argv)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in STAGED_COMMANDS:
        import vs_staged

        return vs_staged.main(argv)
    args = parse_args(argv)

    if args.example:
        import vs_runfile

        sys.stdout.write(vs_runfile.EXAMPLE)
        return 0
    if args.example_config:
        import vs_classic

        return vs_classic.main(["--example"])

    is_run_file = bool(args.file) and args.file.lower().endswith(RUN_FILE_EXTENSIONS)
    if args.file and not is_run_file:
        if args.output is None:
            args.output = "vs_run"
        import vs_classic

        return vs_classic.main([args.file, "-o", args.output] + (["--force"] if args.force else [])
                               + (["--check"] if args.check else []))
    if args.output is not None or args.force:
        print("--output and --force belong to classic mode (a config.txt file). The assistant and run files keep "
              "everything in a run folder that you choose.", file=sys.stderr)
        return 2
    if args.check and not args.file:
        print("--check needs a run file.", file=sys.stderr)
        return 2

    import vs_wizard

    vs_wizard.check_environment()
    if args.file:
        vs_wizard.run_from_file(args.file, args.check)
        return 0
    if not sys.stdin.isatty():
        print(f"The assistant asks questions, so it needs a terminal. In a script, give it a run file instead: "
              f"`{RUN_COMMAND} run.yaml`.", file=sys.stderr)
        return 2
    vs_wizard.run_interactive()
    return 0


if __name__ == "__main__":
    sys.exit(main())
