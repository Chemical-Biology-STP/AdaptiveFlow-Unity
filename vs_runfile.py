"""The run file: a YAML description of one virtual screening run, used as `vs run.yaml` instead of the interactive
assistant.

This module only reads, checks and writes the file. It knows nothing about structures or ligands; the checks that
need the data (which molecule is the reference ligand, are there flagged ligands, ...) are made by vs_wizard.py, which
stops with an explanation when the file does not answer a question the assistant would have asked.

Adapted from runfile.py of the anchored_docking_workflow repository (MIT License, see NOTICE.md).
"""
import difflib
import os
import re

import yaml

import vs_dock
import vs_staged

STRUCTURE_EXTENSIONS = (".pdb", ".ent", ".cif", ".mmcif")
LIGAND_EXTENSIONS = (".sdf", ".smi", ".smiles", ".txt", ".csv")
RUN_FILE_EXTENSIONS = (".yaml", ".yml")

TOP_KEYS = ("structure", "template_molecule", "chains", "keep_molecules", "receptor", "reference_ligand",
            "box_center", "ligands", "output", "existing_output", "protonation", "ph", "flagged_ligands",
            "screen", "docking", "staged", "run")
SCREEN_MODES = ("full", "staged")      # full: dock every molecule; staged: AFLP + AFVS (vs_staged.py)
STAGED_KEYS = ("strategy", "representatives", "primary_size", "tranche_properties", "ml_filter", "ml_cutoff",
               "max_stereoisomers", "max_tautomers")
TRANCHE_ONLY_KEYS = ("representatives", "primary_size", "tranche_properties", "ml_filter", "ml_cutoff")
STAGED_FORBIDDEN_DOCKING = ("num_modes", "rescore", "ionisation")
MAX_CPUS_FULL = 256
MAX_CPUS_STAGED = 3000                  # the cluster's limit for one user; the assistant checks the real figure
DOCKING_KEYS = ("program", "exhaustiveness", "num_modes", "seed", "box_size", "rescore", "ionisation", "ligand_ph")
RUN_KEYS = ("mode", "job_name", "time", "cpus", "partition", "watch")


class RunFileError(Exception):
    """The run file cannot be used. The message says what is wrong and what to do about it."""


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


class _Problems:
    def __init__(self):
        self.lines = []

    def add(self, key, message):
        self.lines.append(f"{key}: {message}")


def _unknown(keys, allowed, where, problems):
    for key in keys:
        if key not in allowed:
            close = difflib.get_close_matches(str(key), allowed, n=1)
            hint = f" Did you mean '{close[0]}'?" if close else ""
            problems.add(where + str(key), f"unknown key.{hint}")


def _text(value, key, problems):
    if isinstance(value, str) and value.strip():
        return value.strip()
    problems.add(key, "must be text")
    return None


def _path(value, key, base_dir, problems, extensions=(), must_exist=True):
    text = _text(value, key, problems)
    if text is None:
        return None
    path = os.path.abspath(os.path.join(base_dir, os.path.expanduser(os.path.expandvars(text))))
    if extensions and not path.lower().endswith(extensions):
        problems.add(key, f"expected a {' or '.join(extensions)} file, got {os.path.basename(path)}")
    elif must_exist and not os.path.isfile(path):
        problems.add(key, f"file not found: {path}")
    elif must_exist and os.path.getsize(path) == 0:
        problems.add(key, f"the file is empty: {path}")
    return path


def _choice(value, key, choices, problems):
    if isinstance(value, str) and value.lower() in choices:
        return value.lower()
    problems.add(key, f"must be one of: {', '.join(choices)}")
    return None


def _number(value, key, problems, cast=float, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or (cast is int and value != int(value)):
        problems.add(key, "must be a number" if cast is float else "must be a whole number")
        return None
    value = cast(value)
    if minimum is not None and value < minimum:
        problems.add(key, f"must be at least {minimum}")
        return None
    if maximum is not None and value > maximum:
        problems.add(key, f"must be at most {maximum}")
        return None
    return value


def _flag(value, key, problems):
    if isinstance(value, bool):
        return value
    problems.add(key, "must be true or false")
    return None


def _names(value, key, problems):
    """A list of names given as a YAML list or as one text ("A B" or "A, B")."""
    if isinstance(value, str):
        value = value.replace(",", " ").split()
    if isinstance(value, list) and all(isinstance(v, (str, int)) and not isinstance(v, bool) for v in value):
        return [str(v).strip() for v in value if str(v).strip()]
    problems.add(key, "must be a list of names, for example [A, B]")
    return None


def _section(data, name, allowed, problems):
    section = data.get(name)
    if section is None:
        return {}
    if not isinstance(section, dict):
        problems.add(name, "must be a group of settings (key: value lines indented below it)")
        return {}
    _unknown(section, allowed, f"{name}.", problems)
    return section


def load(path):
    """Read a run file and check everything that can be checked without looking at the molecules.

    Returns a dict of the settings given (paths absolute; keys that are not in the file are missing from it).
    Raises RunFileError listing every problem found, not only the first.
    """
    path = os.path.abspath(path)
    base = os.path.dirname(path)
    try:
        with open(path) as fh:
            data = yaml.safe_load(fh)
    except OSError as exc:
        raise RunFileError(f"Cannot read {path}: {exc.strerror or exc}")
    except yaml.YAMLError as exc:
        raise RunFileError(f"{path} is not valid YAML:\n{exc}")
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise RunFileError(f"{path} must contain 'key: value' lines.")

    problems = _Problems()
    _unknown(data, TOP_KEYS, "", problems)
    cfg = {"file": path, "base_dir": base}

    # Where the protein comes from, and where the search box goes
    has_structure = data.get("structure") is not None
    has_receptor = data.get("receptor") is not None
    has_ref, has_centre = data.get("reference_ligand") is not None, data.get("box_center") is not None
    if has_structure and has_receptor:
        problems.add("structure", "give either 'structure' (a complex to clean up) or 'receptor' (a protein file that "
                                  "is ready to use), not both")
    elif not has_structure and not has_receptor:
        problems.add("structure", "say what to dock into: either 'structure' (a PDB or mmCIF file of a protein-ligand "
                                  "complex, cleaned up for you) or 'receptor' (a protein PDB file)")
    elif has_structure:
        cfg["route"] = "structure"
        cfg["structure"] = _path(data["structure"], "structure", base, problems, STRUCTURE_EXTENSIONS)
        if data.get("template_molecule") is not None:
            cfg["template_molecule"] = _text(data["template_molecule"], "template_molecule", problems)
        if data.get("chains") is not None:
            cfg["chains"] = _names(data["chains"], "chains", problems)
        if data.get("keep_molecules") is not None:
            cfg["keep_molecules"] = _names(data["keep_molecules"], "keep_molecules", problems)
        if has_ref:
            problems.add("reference_ligand", "only applies with 'receptor'. With 'structure' the box is placed on "
                                             "'template_molecule'")
    else:
        cfg["route"] = "files"
        cfg["receptor"] = _path(data["receptor"], "receptor", base, problems, (".pdb",))
        if has_ref and has_centre:
            problems.add("box_center", "give either 'reference_ligand' or 'box_center', not both")
        elif not has_ref and not has_centre:
            problems.add("reference_ligand", "missing. The search box needs a place: give 'reference_ligand' (an SDF "
                                             "file of a ligand in the binding site, the box is centred on it) or "
                                             "'box_center' ([x, y, z] in Angstrom)")
        if has_ref:
            cfg["reference_ligand"] = _path(data["reference_ligand"], "reference_ligand", base, problems, (".sdf",))
        for key in ("template_molecule", "chains", "keep_molecules"):
            if data.get(key) is not None:
                problems.add(key, "only applies together with 'structure' (cleaning up a complex)")
    if has_centre:
        centre = data["box_center"]
        if (isinstance(centre, list) and len(centre) == 3
                and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in centre)):
            cfg["box_center"] = [float(v) for v in centre]
        else:
            problems.add("box_center", "must be three numbers, for example [12.5, -3.0, 40.2]")

    if data.get("ligands") is None:
        problems.add("ligands", "missing. Give the file with the molecules to dock (SDF, or SMILES)")
    else:
        cfg["ligands"] = _path(data["ligands"], "ligands", base, problems, LIGAND_EXTENSIONS)
    if data.get("output") is None:
        problems.add("output", "missing. Give the folder the results are written to")
    else:
        cfg["output"] = _path(data["output"], "output", base, problems, must_exist=False)

    if data.get("existing_output") is not None:
        cfg["existing_output"] = _choice(data["existing_output"], "existing_output", ("resume", "overwrite"), problems)
    if data.get("protonation") is not None:
        cfg["protonation"] = _choice(data["protonation"], "protonation", ("pdb2pqr", "adfr", "none"), problems)
    if data.get("ph") is not None:
        cfg["ph"] = _number(data["ph"], "ph", problems, float, 0.0, 14.0)
        if cfg.get("protonation") not in (None, "pdb2pqr"):
            problems.add("ph", "only applies with 'protonation: pdb2pqr'")
    if data.get("flagged_ligands") is not None:
        cfg["flagged_ligands"] = _choice(data["flagged_ligands"], "flagged_ligands", ("exclude", "include"), problems)

    cfg["screen"] = "full"
    if data.get("screen") is not None:
        cfg["screen"] = _choice(data["screen"], "screen", SCREEN_MODES, problems) or "full"
    docking = _section(data, "docking", DOCKING_KEYS, problems)
    cfg["docking"] = _read_docking(docking, problems)
    cfg["run"] = _read_run(_section(data, "run", RUN_KEYS, problems), problems)
    cfg["staged"] = _read_staged(_section(data, "staged", STAGED_KEYS, problems), problems)
    _check_screen_mode(cfg, data, docking, problems)

    if problems.lines:
        raise RunFileError(f"Problems in {path}:\n" + "\n".join(f"  - {line}" for line in problems.lines))
    return cfg


def _read_docking(section, problems):
    out = {}
    if "program" in section:
        out["program"] = _choice(section["program"], "docking.program", tuple(p.lower() for p in vs_dock.PROGRAMS),
                                 problems)
        if out["program"]:      # the file may say 'AutodockVina_1.2' in any case; use the real name
            out["program"] = next(p for p in vs_dock.PROGRAMS if p.lower() == out["program"])
    if "exhaustiveness" in section:
        out["exhaustiveness"] = _number(section["exhaustiveness"], "docking.exhaustiveness", problems, int, 1, 128)
    if "num_modes" in section:
        out["num_modes"] = _number(section["num_modes"], "docking.num_modes", problems, int, 1, 20)
    if "seed" in section:
        out["seed"] = _number(section["seed"], "docking.seed", problems, int, 0)
    if "box_size" in section:
        out["box_size"] = _number(section["box_size"], "docking.box_size", problems, float, 1.0, 126.0)
    if section.get("ionisation") is not None:
        out["ionisation"] = _choice(section["ionisation"], "docking.ionisation", vs_dock.IONISATION_MODES, problems)
    if section.get("ligand_ph") is not None:
        out["ligand_ph"] = _number(section["ligand_ph"], "docking.ligand_ph", problems, float, 0.0, 14.0)
        if out.get("ionisation") == "keep":
            problems.add("docking.ligand_ph", "only applies with 'ionisation: ph'")
    if section.get("rescore") is not None:
        value = section["rescore"]
        out["rescore"] = None if value is False or (isinstance(value, str) and value.lower() == "none") else \
            _choice(value, "docking.rescore", ("nnscore2", "none"), problems)
    return out


def _read_staged(section, problems):
    out = {}
    for key in ("max_stereoisomers", "max_tautomers"):
        if key in section:
            out[key] = _number(section[key], f"staged.{key}", problems, int, 1, 20)
    if section.get("strategy") is not None:
        out["strategy"] = _choice(section["strategy"], "staged.strategy", vs_staged.STRATEGIES, problems)
    if out.get("strategy", "all") != "tranche":
        for key in TRANCHE_ONLY_KEYS:
            if key in section:
                problems.add(f"staged.{key}", "only applies with 'strategy: tranche'")
        return out
    if "representatives" in section:
        out["representatives"] = _number(section["representatives"], "staged.representatives", problems, int, 1,
                                         vs_staged.MAX_REPRESENTATIVES)
    if "primary_size" in section:
        out["primary_size"] = _number(section["primary_size"], "staged.primary_size", problems, int, 1)
    if section.get("ml_filter") is not None:
        out["ml_filter"] = _flag(section["ml_filter"], "staged.ml_filter", problems)
    if section.get("ml_cutoff") is not None:
        out["ml_cutoff"] = _number(section["ml_cutoff"], "staged.ml_cutoff", problems, float, 0.05, 0.95)
        if not out.get("ml_filter"):
            problems.add("staged.ml_cutoff", "only applies with 'ml_filter: true'")
    if section.get("tranche_properties") is not None:
        names = _names(section["tranche_properties"], "staged.tranche_properties", problems)
        if names is not None:
            unknown = [n for n in names if n not in vs_staged.TRANCHE_PROPERTY_NAMES]
            if not names or unknown or len(set(names)) != len(names):
                problems.add("staged.tranche_properties",
                             "must be a list of different names from: " + ", ".join(vs_staged.TRANCHE_PROPERTY_NAMES)
                             + (f" (not understood: {', '.join(unknown)})" if unknown else ""))
            else:
                out["tranche_properties"] = names
    return out


def _check_screen_mode(cfg, data, docking, problems):
    """Settings that only make sense in one of the two modes."""
    run = cfg["run"]
    if cfg["screen"] == "staged":
        for key in STAGED_FORBIDDEN_DOCKING:
            if key in docking:
                problems.add(f"docking.{key}", "does not apply to the staged screen (it keeps scores, not poses, and "
                                               "prepares the molecules itself)")
        if run.get("mode") == "run":
            problems.add("run.mode", "the staged screen runs on the cluster: use 'submit' or 'save'")
        for key in ("time", "watch"):
            if key in run:
                problems.add(f"run.{key}", "does not apply to the staged screen (the time limit of every job is set "
                                           "from the size of the library; follow the run with `status`)")
    else:
        if data.get("staged") is not None:
            problems.add("staged", "only applies with 'screen: staged'")
        if (run.get("cpus") or 0) > MAX_CPUS_FULL:
            problems.add("run.cpus", f"must be at most {MAX_CPUS_FULL} (the staged screen allows more)")


def _read_run(section, problems):
    out = {}
    if section.get("mode") is None:
        problems.add("run.mode", "missing. Say what to do with the job: 'submit' (send it to the cluster queue), "
                                 "'run' (run it now in this terminal) or 'save' (only write the job script)")
    else:
        out["mode"] = _choice(section["mode"], "run.mode", ("submit", "run", "save"), problems)
    if section.get("job_name") is not None:
        name = _text(section["job_name"], "run.job_name", problems)
        if name and not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            problems.add("run.job_name", "use only letters, digits, '_', '.' and '-'")
        elif name:
            out["job_name"] = name
    if section.get("time") is not None:
        time_limit = section["time"]
        if not isinstance(time_limit, str):
            # YAML reads an unquoted 2:00:00 as the number 7200, which SLURM would take as 7200 minutes
            problems.add("run.time", f"put the time in quotes, for example \"2:00:00\" (without quotes YAML turns it "
                                     f"into the number {time_limit})")
        elif re.fullmatch(r"(\d+-)?\d+(:\d{2}){0,2}", time_limit.strip()):
            out["time"] = time_limit.strip()
        else:
            problems.add("run.time", "use hours:minutes:seconds in quotes, for example \"2:00:00\" "
                                     "(or days-hours:minutes:seconds)")
    if section.get("cpus") is not None:
        out["cpus"] = _number(section["cpus"], "run.cpus", problems, int, 1, MAX_CPUS_STAGED)
    if section.get("partition") is not None:
        out["partition"] = _text(section["partition"], "run.partition", problems)
    if "watch" in section:
        out["watch"] = _flag(section["watch"], "run.watch", problems)
    return out


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def _scalar(value):
    # Always quote a time such as 2:00:00: YAML reads some of those as numbers, and this keeps the file safe to edit.
    if isinstance(value, str) and re.fullmatch(r"(\d+-)?\d+(:\d{2}){1,2}", value):
        return f'"{value}"'
    text = yaml.safe_dump(value, default_flow_style=True, width=10 ** 6).strip()
    return text[:-4].rstrip() if text.endswith("\n...") else text


def _line(key, value, comment="", indent=0):
    text = f"{' ' * indent}{key}: {_scalar(value)}"
    return f"{text.ljust(44)} # {comment}" if comment else text


def render(values, run_command="vs"):
    """Write the answers of an assistant run as a run file. `values` has the keys of the file (see load()), with the
    sections 'docking' and 'run' as dicts. Keys with the value None are left out."""
    lines = [f"# Virtual screening run file, written by the assistant. Run it again with:  {run_command} <this file>",
             "# Edit the values to change the run; paths without a leading / are relative to this file.", ""]
    inputs = [("structure", "complex to clean up (PDB or mmCIF)"),
              ("template_molecule", "the ligand the box is centred on: residue code, chain and number"),
              ("chains", "protein chains kept"),
              ("keep_molecules", "ions or cofactors kept in the receptor ([] for none)"),
              ("receptor", "protein PDB file"),
              ("reference_ligand", "ligand in the binding site (SDF); the box is centred on it"),
              ("box_center", "box centre x, y, z in Angstrom")]
    for key, comment in inputs:
        if values.get(key) is not None:
            lines.append(_line(key, values[key], comment))
    if values.get("screen") and values["screen"] != "full":
        lines.append(_line("screen", values["screen"], "full: dock every molecule | staged: AdaptiveFlow (AFLP + AFVS)"))
    lines.append(_line("ligands", values["ligands"], "molecules to dock"))
    lines.append(_line("output", values["output"], "run folder: inputs, job script, log and results/"))
    for key, comment in (("existing_output", "output folder is not empty: resume | overwrite"),
                         ("protonation", "receptor had no hydrogens: pdb2pqr | adfr | none"),
                         ("ph", "pH for pdb2pqr"),
                         ("flagged_ligands", "ligands that were flagged: exclude | include")):
        if values.get(key) is not None:
            lines.append(_line(key, values[key], comment))
    for name, comments in (("docking", {
            "program": " | ".join(vs_dock.PROGRAMS), "exhaustiveness": "search effort per ligand",
            "num_modes": "poses kept per ligand", "seed": "random seed", "box_size": "box edge in Angstrom",
            "rescore": "nnscore2 | none", "ionisation": "ph | keep", "ligand_ph": "pH for the molecules"}),
            ("staged", {"strategy": "all: dock every form | tranche: dock the best tranches only",
                        "representatives": "forms docked per tranche in the prescreen (1 to 10)",
                        "primary_size": "molecules docked in the primary screen",
                        "tranche_properties": "properties that define the tranches",
                        "ml_filter": "true: a neural network drops forms of the selected tranches before docking",
                        "ml_cutoff": "forms below this predicted probability are not docked (0.05 to 0.95)",
                        "max_stereoisomers": "stereoisomers prepared per molecule (1 = as drawn)",
                        "max_tautomers": "tautomers prepared per molecule (1 = as drawn)"}),
            ("run", {"mode": "submit | run | save", "job_name": "", "time": "time limit, hours:minutes:seconds",
                     "cpus": "ligands docked at the same time", "partition": "", "watch": "wait for the job"})):
        section = {k: v for k, v in (values.get(name) or {}).items() if v is not None}
        if section:
            lines += ["", f"{name}:"]
            lines += [_line(k, v, comments.get(k, ""), 2) for k, v in section.items()]
    return "\n".join(lines) + "\n"


EXAMPLE = """\
# Virtual screening run file. Start it with:  vs run.yaml      (check it first with:  vs run.yaml --check)
# Paths without a leading / are relative to this file. Everything the assistant would ask is either set here or has
# the default noted below. Where the data forces a decision that is missing, the run stops and says what to add.

# --- 1. The protein: EITHER a complex to clean up ... ---------------------------------------------------------
structure: 4djp.pdb              # PDB or mmCIF file of a protein-ligand complex
template_molecule: B7Z A401      # the ligand the search box is centred on: residue code, chain and number (see the
                                 # error message for the candidates if you do not know them)
chains: [A]                      # protein chains to keep (needed when the file has more than one)
keep_molecules: []               # ions or cofactors near the ligand to keep, e.g. [ZN A601]; [] keeps none

# --- ... OR a protein file that is ready to use, plus a place for the box -------------------------------------
# receptor: protein.pdb          # protein PDB file, without the ligand
# reference_ligand: ligand.sdf   # a ligand in the binding site (SDF); the box is centred on it
# box_center: [12.5, -3.0, 40.2] # ... or give the centre yourself (Angstrom), instead of reference_ligand

# --- 2. What to dock and where the results go -----------------------------------------------------------------
ligands: library.smi             # SDF file, or SMILES file (one "SMILES name" per line)
output: results/run1             # results folder
# existing_output: resume        # needed only if the folder is not empty: resume (skip finished ligands) | overwrite

# --- 3. Decisions the assistant asks about, needed only when they arise ---------------------------------------
# protonation: pdb2pqr           # needed if the receptor has no hydrogens: pdb2pqr (predicts charges at a pH) | adfr (quick guess) | none
# ph: 7.4                        # pH for pdb2pqr (default 7.4)
# flagged_ligands: exclude       # needed if some ligands are flagged: exclude | include

# --- 4. Docking settings (all optional; these are the assistant's recommendations) ----------------------------
docking:
  program: qvina                 # qvina | qvina-w | vina | AutodockVina_1.2 | PSOVina | smina
  # exhaustiveness: 16           # search effort per ligand (higher is slower and more thorough)
  # num_modes: 9                 # poses kept per ligand
  # box_size: 20                 # box edge in Angstrom (default: the ligand's size + 10, at least 20)
  # seed: 1                      # random seed
  # rescore: nnscore2            # re-score the poses with NNScore2 (default: none)
  # ionisation: ph               # ph: set each molecule's charge state for a pH (default) | keep: use the charges in the file
  # ligand_ph: 7.4               # the pH, with 'ionisation: ph' 

# --- Staged screening with AdaptiveFlow (for libraries too big to dock whole; scores only, no poses) --------------
# screen: staged                 # default: full (dock every molecule, keep the poses)
# staged:
#   strategy: all                # all: dock every form (default) | tranche: prescreen the tranches, dock only the best
#   representatives: 3           # with 'tranche': forms docked per tranche in the prescreen (1 to 10)
#   primary_size: 100000         # with 'tranche': molecules docked in the primary screen (default: 20% of the library)
#   tranche_properties: [mw, logp, hba, hbd]   # with 'tranche': the properties that define the tranches (of 18, see README)
#   ml_filter: false             # with 'tranche': train a neural network on the prescreen and dock only the forms of the
#                                # selected tranches that it predicts to be good (primary_size is then the pool it looks at)
#   ml_cutoff: 0.5               # with ml_filter: forms below this predicted probability are dropped
#   max_stereoisomers: 1         # stereoisomers prepared per molecule; every form is docked, so cost grows with it
#   max_tautomers: 1             # tautomers prepared per molecule
# With 'screen: staged', run.mode must be submit or save, run.cpus may go up to the cluster limit (3,000), and
# docking.num_modes, docking.rescore, docking.ionisation, run.time and run.watch are not used.

# --- 5. How to run it -----------------------------------------------------------------------------------------
run:
  mode: submit                   # submit (cluster queue) | run (here) | save (only write the script)
  # job_name: vs_library         # default: from the ligand file name
  # cpus: 32                     # ligands docked at the same time (default: up to 32, never more than ligands)
  # time: "2:00:00"              # default: estimated from the number of ligands
  # partition: ncpu
  # watch: false                 # true: wait for the job to finish and show the results
"""
