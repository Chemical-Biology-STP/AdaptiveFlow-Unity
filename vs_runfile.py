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

STRUCTURE_EXTENSIONS = (".pdb", ".ent", ".cif", ".mmcif")
LIGAND_EXTENSIONS = (".sdf", ".smi", ".smiles", ".txt", ".csv")
RUN_FILE_EXTENSIONS = (".yaml", ".yml")

TOP_KEYS = ("structure", "template_molecule", "chains", "keep_molecules", "receptor", "reference_ligand",
            "box_center", "ligands", "output", "existing_output", "protonation", "ph", "flagged_ligands",
            "docking", "run")
DOCKING_KEYS = ("program", "exhaustiveness", "num_modes", "seed", "box_size", "rescore")
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

    cfg["docking"] = _read_docking(_section(data, "docking", DOCKING_KEYS, problems), problems)
    cfg["run"] = _read_run(_section(data, "run", RUN_KEYS, problems), problems)

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
    if section.get("rescore") is not None:
        value = section["rescore"]
        out["rescore"] = None if value is False or (isinstance(value, str) and value.lower() == "none") else \
            _choice(value, "docking.rescore", ("nnscore2", "none"), problems)
    return out


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
        out["cpus"] = _number(section["cpus"], "run.cpus", problems, int, 1, 256)
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
            "rescore": "nnscore2 | none"}),
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

# --- 5. How to run it -----------------------------------------------------------------------------------------
run:
  mode: submit                   # submit (cluster queue) | run (here) | save (only write the script)
  # job_name: vs_library         # default: from the ligand file name
  # cpus: 32                     # ligands docked at the same time (default: up to 32, never more than ligands)
  # time: "2:00:00"              # default: estimated from the number of ligands
  # partition: ncpu
  # watch: false                 # true: wait for the job to finish and show the results
"""
