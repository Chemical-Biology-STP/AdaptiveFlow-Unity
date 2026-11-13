#!/usr/bin/env python3
"""The docking engine behind the assistant: prepare a receptor and a library of ligands, dock every ligand with one
of the Vina-family programs, optionally re-score the poses with NNScore2, and write the results.

Nothing in here asks the user anything. The assistant (vs_wizard.py), a run file and the SLURM job script all end up
calling `python vs_dock.py <arguments>` (see build_parser), so a run behaves the same whichever way it was started.

What it does, in order:

  1. Receptor: ADFR's prepare_receptor makes receptor.pdbqt from the PDB file (done once).
  2. Ligands: each molecule of the input file is cleaned (largest fragment, neutral if possible), given 3D
     coordinates by RDKit and written as PDBQT by Meeko. Unspecified stereocentres and tautomers are NOT enumerated:
     what is in the file is what is docked.
  3. Docking: every ligand is docked by the chosen program with one CPU; several ligands run at the same time, up to
     the number of CPUs given with --cpus.
  4. Re-scoring (optional): NNScore2 scores each pose.
  5. Output: results.csv (one row per ligand), top_poses.sdf and all_poses.sdf (complete molecules with every
     hydrogen), and one folder per ligand with the raw program output.

A run can be stopped and started again with --resume: ligands that already have a result are skipped.
"""

import argparse
import concurrent.futures
import csv
import hashlib
import json
import multiprocessing
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
import warnings

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
EXECUTABLES = os.path.join(REPO_DIR, "executables")
NNSCORE2 = os.path.join(REPO_DIR, "config", "NNScore2.py")
# NNScore2 needs AutoDock Vina 1.1.2 to compute some of its terms.
NNSCORE2_VINA = os.path.join(EXECUTABLES, "AutodockVina_1.1.2")
DEFAULT_ADFR_PATH = os.path.join(REPO_DIR, ".pixi", "envs", "adfr")

# Programs that were run end to end on a Meeko-prepared ligand and an ADFR-prepared receptor.
#   seconds_per_ligand: CPU seconds per molecule at exhaustiveness 8 (used only to suggest a time limit). QuickVina:
#   38, half of the 75 s measured at exhaustiveness 16 on 99,982 random Enamine screening molecules (5th to 95th
#   percentile 26 to 142 s; the first 2,000 of them took 66.5 s on average). The earlier 20 s came from the smaller NCI
#   test molecules. The other programs were timed only on the NCI molecules (QuickVina-W 25, Vina 20, Vina 1.2 21,
#   PSOVina 3, smina 31 s) and are scaled by the same 1.9, so they are estimates for drug-like libraries, not
#   measurements.
#   unsupported: elements that the program refuses to read in a PDBQT file
PROGRAMS = {
    "qvina": {"title": "QuickVina 2.1", "seconds_per_ligand": 38, "unsupported": ("B", "Si", "Se"),
              "text": "fast Vina variant; a good default for screening"},
    "qvina-w": {"title": "QuickVina-W 1.1", "seconds_per_ligand": 48, "unsupported": ("B", "Si", "Se"),
                "text": "QuickVina for large boxes (blind docking)"},
    "vina": {"title": "AutoDock Vina 1.1.2", "seconds_per_ligand": 38, "unsupported": ("B", "Si", "Se"),
             "text": "the original Vina"},
    "AutodockVina_1.2": {"title": "AutoDock Vina 1.2.3", "seconds_per_ligand": 40, "unsupported": ("B", "Se"),
                         "text": "current Vina; also accepts silicon"},
    "PSOVina": {"title": "PSOVina 2.0", "seconds_per_ligand": 6, "unsupported": ("B", "Si", "Se"),
                "text": "fastest, but less accurate and not reproducible (it ignores the random seed)"},
    "smina": {"title": "smina (Oct 2019)", "seconds_per_ligand": 59, "unsupported": ("B", "Si", "Se"),
              "text": "Vina fork with extra scoring options"},
}
# Elements that Meeko can give an AutoDock atom type (tested element by element). A molecule with any other element
# cannot be prepared, so it is flagged before the job starts instead of failing with "atom has None type".
TYPED_ELEMENTS = frozenset("H B C N O F Mg Si P S Cl Ca Mn Fe Zn Br I".split())
DEFAULT_PROGRAM = "qvina"
DEFAULT_PH = 7.4
IONISATION_MODES = ("ph", "keep")     # ph: assign protonation states at a pH; keep: use the charges in the file
# 16, not Vina's usual 8: in a redocking test on 12 crystal complexes QuickVina found the crystal pose as its top
# pose for 7 of 12 at exhaustiveness 8 and 9 of 12 at 16 or 32 (the two it missed were both recovered at 16).
DEFAULT_EXHAUSTIVENESS = 16
DEFAULT_NUM_MODES = 9
DEFAULT_SEED = 1
DEFAULT_BOX = 20.0
EMBED_TIMEOUT_SECONDS = 30          # per attempt; a hopeless molecule otherwise takes about a minute to give up
DOCK_TIMEOUT_SECONDS = 3600         # one ligand taking longer than this is reported as failed
NNSCORE_TIMEOUT_SECONDS = 600
SIGNATURE_FILE = "run_settings.json"
SUMMARY_FILE = "results.csv"
RESULT_COLUMNS = ["index", "name", "status", "best_score", "n_poses", "pkd_best", "note", "smiles"]


class LigandError(Exception):
    """One ligand cannot be prepared or docked; the message says why in plain words."""


# --------------------------------------------------------------------------
# Receptor
# --------------------------------------------------------------------------


def prepare_receptor(workdir, receptor_pdb, adfr_path=DEFAULT_ADFR_PATH, add_hydrogens=False):
    """Make receptor.pdbqt in workdir with ADFR's prepare_receptor. Returns its path.

    add_hydrogens: let ADFR add hydrogens if the file has none (a quick rule-based guess).
    """
    exe = os.path.join(adfr_path, "bin", "prepare_receptor")
    if not os.access(exe, os.X_OK):
        raise RuntimeError(f"ADFRsuite's prepare_receptor was not found at {exe}.")
    os.makedirs(workdir, exist_ok=True)
    shutil.copyfile(receptor_pdb, os.path.join(workdir, "receptor.pdb"))
    command = [exe, "-r", "receptor.pdb", "-o", "receptor.pdbqt"]
    if add_hydrogens:
        command += ["-A", "checkhydrogens"]      # only adds hydrogens if the file has none
    result = subprocess.run(command, cwd=workdir, capture_output=True, text=True, check=False)
    pdbqt = os.path.join(workdir, "receptor.pdbqt")
    if result.returncode != 0 or not os.path.isfile(pdbqt) or os.path.getsize(pdbqt) == 0:
        tail = (result.stderr or result.stdout).strip().splitlines()[-3:]
        raise RuntimeError("ADFR could not prepare the receptor" + (": " + " | ".join(tail) if tail else "."))
    return pdbqt


# --------------------------------------------------------------------------
# Ligands
# --------------------------------------------------------------------------


def _skeleton(mol):
    """The molecule with all charges removed and no hydrogens: what must be unchanged by ionisation."""
    from rdkit import Chem

    copy = Chem.RWMol(Chem.RemoveHs(mol))
    for atom in copy.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(False)
    try:
        Chem.SanitizeMol(copy)
    except Exception:
        pass
    return Chem.MolToSmiles(copy)


# Groups Open Babel's pH model misses, added where the state is unambiguous. Each is (SMARTS of the acidic oxygen or
# nitrogen, the pKa used, the reason). They are applied only when pH is at least two units above the pKa.
EXTRA_ACID_RULES = (
    ("[SX4](=O)(=O)[OX2H1:1]", 0.0, "sulfonic acid, pKa about -2 to 1 (Open Babel misses it when the substituent is a "
                                    "carbon with no other heavy neighbours, e.g. methanesulfonic acid)"),
    ("[CX3](=O)[NX3H1:1][SX4](=O)(=O)", 4.5, "acylsulfonamide, pKa about 4 to 5"),
)


def _apply_extra_acids(mol, ph):
    """Deprotonate the acid groups of EXTRA_ACID_RULES that Open Babel left alone. Returns (molecule, changed)."""
    from rdkit import Chem

    work, changed = Chem.RWMol(mol), False
    for smarts, pka, _ in EXTRA_ACID_RULES:
        if ph < pka + 2.0:
            continue
        pattern = Chem.MolFromSmarts(smarts)
        for match in work.GetSubstructMatches(pattern):
            atom = work.GetAtomWithIdx(match[[a.GetAtomMapNum() for a in pattern.GetAtoms()].index(1)])
            if atom.GetFormalCharge() == 0 and atom.GetTotalNumHs() >= 1:
                atom.SetFormalCharge(-1)
                atom.SetNumExplicitHs(max(0, atom.GetTotalNumHs() - 1))
                atom.SetNoImplicit(True)
                changed = True
    if changed:
        try:
            Chem.SanitizeMol(work)
        except Exception:
            return mol, False
    return work.GetMol(), changed


def ionise(mol, ph=DEFAULT_PH):
    """Protonation state of `mol` at `ph`, using Open Babel's pH model (a list of about 30 pKa-tagged rules for
    carboxylic, sulfonic and phosphoric acids, aliphatic amines, amidines and guanidines, tetrazoles and so on).

    Returns (molecule, changed). Falls back to the input if Open Babel fails or if the result is not the same molecule
    apart from protons, so ionisation can never alter the structure of a ligand.

    Tested on 49 molecules with unambiguous states at pH 7.4: 45 correct without the extra rules, 48 with them. The
    one known miss is 4-aminopyridine-type bases (no safe general rule). Groups with a pKa close to the pH (imidazole, morpholine) are decided by a rule
    that cannot know the environment.
    """
    from openbabel import openbabel as ob
    from rdkit import Chem

    try:
        neutral = Chem.MolToSmiles(mol)
        conv = ob.OBConversion()
        conv.SetInAndOutFormats("smi", "smi")
        obmol = ob.OBMol()
        if not conv.ReadString(obmol, neutral):
            return mol, False
        obmol.AddHydrogens(False, True, float(ph))      # polaronly=False, correctForPH=True
        result = Chem.MolFromSmiles(conv.WriteString(obmol).split()[0])
    except Exception:
        return mol, False
    if result is None or _skeleton(result) != _skeleton(mol):
        return mol, False
    # stereochemistry must survive: compare the stereo of the skeletons
    if Chem.MolToSmiles(Chem.RemoveHs(result)).count("@") != Chem.MolToSmiles(Chem.RemoveHs(mol)).count("@"):
        return mol, False
    result, extra = _apply_extra_acids(result, ph)
    changed = extra or Chem.MolToSmiles(result) != neutral
    return result, changed


def clean_molecule(mol, ionisation="ph", ph=DEFAULT_PH):
    """Prepare a molecule's chemistry for docking: keep the largest fragment (drops salts and counter-ions), then set
    its protonation state.

    ionisation="ph":   assign the states at `ph` (carboxylic acids become anions, aliphatic amines cations, and so
                       on), whatever charge state the file shows. This is what a docking run at physiological pH
                       should use for a library of ordinary structures.
    ionisation="keep": use the charges exactly as written in the file. For libraries that were already prepared at a
                       given pH, where ionising again would undo that work.

    Returns (molecule, note) where note is "" or a sentence describing what was changed.
    """
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize

    notes = []
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if len(fragments) > 1:
        mol = max(fragments, key=lambda m: m.GetNumHeavyAtoms())
        notes.append(f"kept the largest of {len(fragments)} fragments")
    if ionisation == "keep":
        return mol, "; ".join(notes)
    # Open Babel is given the molecule as it is written. It moves protons to match the pH, so a group that is already
    # in the right state stays, and a wrongly drawn one is corrected. Neutralising first would turn, for example, a
    # sulfonate into a sulfonic acid whose rule Open Babel then misses.
    ionised, changed = ionise(mol, ph)
    if changed:
        notes.append(f"ionised for pH {ph:g}")
    return ionised, "; ".join(notes)


def ligand_name(mol, index):
    if mol is not None and mol.HasProp("_Name") and mol.GetProp("_Name").strip():
        return mol.GetProp("_Name").strip()
    return f"molecule_{index + 1}"


def iter_ligands(path):
    """Read a ligand file one molecule at a time (.sdf, .smi, .smiles, .txt, .csv with SMILES).

    Yields (index, name, mol_or_None, error). The index counts molecules in the file (a SMILES header line is not
    one), so the wizard and the job always agree on which molecule is which. Nothing is kept in memory.

    SMILES files: one molecule per line, "SMILES name"; a header line is skipped; space, "," ";" and tab separate.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    if path.lower().endswith(".sdf"):
        # The file is split at "$$$$" here and each record parsed on its own. RDKit's SDMolSupplier loses its place
        # after a malformed record and also discards the good molecule that follows it, which would shift every later
        # index (and make the count differ from count_ligands).
        record = []
        index = 0
        with open(path, errors="replace") as fh:
            for line in fh:
                if line.startswith("$$$$"):
                    mol = Chem.MolFromMolBlock("".join(record), removeHs=False) if "".join(record).strip() else None
                    if mol is None:
                        yield index, f"molecule_{index + 1}", None, "RDKit could not read this molecule"
                    else:
                        yield index, ligand_name(mol, index), mol, ""
                    index += 1
                    record = []
                else:
                    record.append(line)
        return
    index = 0
    with open(path, errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p for p in re.split(r"[\s,;]+", line) if p]
            smiles = parts[0]
            name = parts[1] if len(parts) > 1 else f"molecule_{index + 1}"
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                if index == 0:             # a header such as "smiles,name"
                    continue
                yield index, name, None, f"'{smiles[:40]}' is not a valid SMILES"
            else:
                mol.SetProp("_Name", name)
                yield index, name, mol, ""
            index += 1


def read_ligands(path):
    """All molecules of a file as a list of (name, mol_or_None, error). Fine for short files; use iter_ligands for
    libraries."""
    return [(name, mol, error) for _, name, mol, error in iter_ligands(path)]


def count_ligands(path):
    """How many molecules iter_ligands will yield, without parsing them (SMILES: only the first line is parsed)."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    if path.lower().endswith(".sdf"):
        with open(path, errors="replace") as fh:
            return sum(1 for line in fh if line.startswith("$$$$"))
    count, first = 0, True
    with open(path, errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if first:
                first = False
                parts = [p for p in re.split(r"[\s,;]+", line) if p]
                if Chem.MolFromSmiles(parts[0]) is None:
                    continue               # header
            count += 1
    return count


def blocked_elements(mol, program):
    """Elements in the molecule that the chosen program cannot read."""
    bad = PROGRAMS[program]["unsupported"]
    return sorted({a.GetSymbol() for a in mol.GetAtoms() if a.GetSymbol() in bad})


def untyped_elements(mol):
    """Elements in the molecule that have no AutoDock atom type (metals and metalloids, mostly)."""
    return sorted({a.GetSymbol() for a in mol.GetAtoms() if a.GetSymbol() not in TYPED_ELEMENTS})


def embed_ligand(mol, timeout=EMBED_TIMEOUT_SECONDS, seed=DEFAULT_SEED):
    """Add hydrogens and generate 3D coordinates. Raises LigandError if RDKit cannot. RDKit reports most failures only
    through its return value (and leaves a molecule without coordinates, which would be docked as a flat structure),
    but some molecules, for example metal complexes with unusual bonding, make it raise an exception instead."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    reason = ("RDKit could not generate 3D coordinates (strained or impossible geometry or stereochemistry, "
              "or an unusual structure such as a metal complex)")
    try:
        mol_h = Chem.AddHs(mol, addCoords=True)
        params = AllChem.ETKDGv3()
        params.timeout = timeout
        params.randomSeed = seed
        if AllChem.EmbedMolecule(mol_h, params) != 0:
            retry = AllChem.ETKDGv3()
            retry.timeout = timeout
            retry.randomSeed = seed
            retry.useRandomCoords = True            # RDKit's usual fallback for hard-to-embed molecules
            if AllChem.EmbedMolecule(mol_h, retry) != 0:
                raise LigandError(reason)
    except (RuntimeError, ValueError):
        raise LigandError(reason)
    return mol_h


def ligand_to_pdbqt(mol, seed=DEFAULT_SEED, ionisation="ph", ph=DEFAULT_PH):
    """Clean, embed and convert one molecule. Returns (pdbqt_text, note, cleaned_smiles)."""
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")        # importing Meeko turns RDKit's log back on; its messages are harmless
    mol, note = clean_molecule(mol, ionisation, ph)
    cleaned_smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
    mol_h = embed_ligand(mol, seed=seed)
    try:
        # rigid_macrocycles: by default Meeko writes special ring-closure atom types (CG, G0) that Vina-family
        # programs cannot read; 23 of 4,961 molecules of the NCI test set failed that way. Rings of 7 or more atoms
        # are then rigid instead of flexible.
        setups = MoleculePreparation(rigid_macrocycles=True).prepare(mol_h)
        text, success, error = PDBQTWriterLegacy.write_string(setups[0])
    except Exception as exc:
        raise LigandError(f"Meeko could not prepare this molecule ({type(exc).__name__}: {str(exc)[:100]})")
    if not success:
        reason = (error or "").strip().splitlines()
        raise LigandError("Meeko could not assign AutoDock atom types"
                          + (f" ({reason[0][:80]})" if reason else ""))
    return text, note, cleaned_smiles


def check_ligand(name, mol, program):
    """Quick checks done before the job starts. Returns (usable, problems) where problems are plain-language
    sentences. A ligand with problems is still reported by the job, but will fail there.

    The checks look at the largest fragment, because that is the part the job docks: the sodium of a sodium salt is
    removed, so it must not make the molecule unusable.
    """
    from rdkit import Chem

    problems = []
    if mol is None:
        return False, ["could not be read"]
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    docked = max(fragments, key=lambda m: m.GetNumHeavyAtoms()) if len(fragments) > 1 else mol
    elements = blocked_elements(docked, program)
    if elements:
        problems.append(f"contains {', '.join(elements)}, which {PROGRAMS[program]['title']} cannot read")
    untyped = [e for e in untyped_elements(docked) if e not in elements]
    if untyped:
        problems.append(f"contains {', '.join(untyped)}, which has no AutoDock atom type (metal complexes and "
                        "organometallics cannot be docked)")
    heavy = docked.GetNumHeavyAtoms()
    if heavy < 3:
        problems.append(f"has only {heavy} heavy atom(s)")
    elif heavy > 80:
        problems.append(f"has {heavy} heavy atoms; docking such a large molecule is unreliable")
    if len(fragments) > 1:
        problems.append("has several fragments (salt?); only the largest is docked")
    return not any(p.startswith(("contains", "could not", "has only")) for p in problems), problems


# --------------------------------------------------------------------------
# Running a program
# --------------------------------------------------------------------------


def program_command(program, receptor, ligand, out, box, exhaustiveness, num_modes, seed):
    """The command line for one docking run (one CPU)."""
    center, size = box["center"], box["size"]
    if program == "smina":
        command = [os.path.join(EXECUTABLES, program), "-r", receptor, "-l", ligand, "-o", out]
    else:
        command = [os.path.join(EXECUTABLES, program), "--receptor", receptor, "--ligand", ligand, "--out", out]
    command += ["--center_x", f"{center[0]:.3f}", "--center_y", f"{center[1]:.3f}", "--center_z", f"{center[2]:.3f}",
                "--size_x", f"{size[0]:.3f}", "--size_y", f"{size[1]:.3f}", "--size_z", f"{size[2]:.3f}",
                "--exhaustiveness", str(exhaustiveness), "--num_modes", str(num_modes), "--cpu", "1",
                "--seed", str(seed)]
    return command


def parse_scores(pdbqt_text):
    """Scores (kcal/mol) of the poses in a docked PDBQT file, in file order.

    Vina-family programs write 'REMARK VINA RESULT:' and smina writes 'REMARK minimizedAffinity'.
    """
    scores = []
    for line in pdbqt_text.splitlines():
        if line.startswith("REMARK VINA RESULT:"):
            scores.append(float(line.split()[3]))
        elif line.startswith("REMARK minimizedAffinity"):
            scores.append(float(line.split()[2]))
    return scores


def poses_to_mols(pdbqt_text, name):
    """Complete molecules (every hydrogen) for the poses in a docked PDBQT file, as one RDKit molecule with one
    conformer per pose. Returns (mol, scores)."""
    from meeko import PDBQTMolecule, RDKitMolCreate

    scores = parse_scores(pdbqt_text)
    pdbqt_mol = PDBQTMolecule(pdbqt_text, name=name, skip_typing=True)
    mols = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)
    if not mols or mols[0] is None:
        raise LigandError("the poses could not be converted back to a molecule")
    return mols[0], scores


NNSCORE_PATTERN = re.compile(r"Best Score:[^,]*,\s*(-?[\d.]+)\s*,")


def nnscore2_pkd(receptor, poses_pdbqt):
    """pKd of every pose according to NNScore2 (higher is stronger binding), in file order."""
    result = subprocess.run([sys.executable, "-W", "ignore", NNSCORE2, "-receptor", receptor, "-ligand", poses_pdbqt,
                             "-vina_executable", NNSCORE2_VINA],
                            capture_output=True, text=True, timeout=NNSCORE_TIMEOUT_SECONDS, check=False)
    values = [float(v) for v in NNSCORE_PATTERN.findall(result.stdout)]
    if result.returncode != 0 or not values:
        tail = (result.stderr or result.stdout).strip().splitlines()[-1:]
        raise LigandError("NNScore2 gave no score" + (f" ({tail[0][:80]})" if tail else ""))
    return values


def _worker_start():
    """Workers are forked from the engine, which turns SIGTERM into KeyboardInterrupt. A worker stopped that way
    prints a long traceback, so give workers the default behaviour: they just end."""
    signal.signal(signal.SIGTERM, signal.SIG_DFL)


def dock_one(job):
    """Prepare, dock and (optionally) re-score one ligand. Runs in a worker process; never raises.

    job: dict with index, name, mol (RDKit binary), folder, and the run settings.
    Returns a result row (dict with RESULT_COLUMNS). The poses are written to the ligand's own poses.sdf.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    warnings.filterwarnings("ignore")
    index, name, folder = job["index"], job["name"], job["folder"]
    row = {"index": index, "name": name, "status": "failed", "best_score": "", "n_poses": "", "pkd_best": "",
           "note": "", "smiles": ""}
    all_blocks = []
    try:
        os.makedirs(folder, exist_ok=True)
        mol = Chem.Mol(job["mol"])        # binary form: keeps stereochemistry that a molblock without coordinates loses
        row["smiles"] = Chem.MolToSmiles(Chem.RemoveHs(mol))
        cleaned = clean_molecule(mol, job["ionisation"], job["ph"])[0]
        blocked = blocked_elements(cleaned, job["program"])
        if blocked:
            raise LigandError(f"contains {', '.join(blocked)}, which {PROGRAMS[job['program']]['title']} cannot read")
        untyped = untyped_elements(cleaned)
        if untyped:
            raise LigandError(f"contains {', '.join(untyped)}, which has no AutoDock atom type (metal complexes and "
                              "organometallics cannot be docked)")
        pdbqt, clean_note, row["smiles"] = ligand_to_pdbqt(mol, job["seed"], job["ionisation"], job["ph"])   # the SMILES that was docked
        ligand_file = os.path.join(folder, "ligand.pdbqt")
        with open(ligand_file, "w") as fh:
            fh.write(pdbqt)
        out_file = os.path.join(folder, "poses.pdbqt")        # smina picks the format from the extension
        command = program_command(job["program"], job["receptor"], ligand_file, out_file, job["box"],
                                  job["exhaustiveness"], job["num_modes"], job["seed"])
        started = time.time()
        try:
            run = subprocess.run(command, capture_output=True, text=True, timeout=DOCK_TIMEOUT_SECONDS, check=False)
        except subprocess.TimeoutExpired:
            raise LigandError(f"docking took longer than {DOCK_TIMEOUT_SECONDS // 60} minutes")
        with open(os.path.join(folder, "program.log"), "w") as fh:
            fh.write(" ".join(command) + "\n\n" + run.stdout + run.stderr)
        if not os.path.isfile(out_file) or os.path.getsize(out_file) == 0:
            lines = [l for l in (run.stderr + run.stdout).replace("\r", "\n").splitlines() if l.strip()]
            raise LigandError(f"{job['program']} wrote no poses" + (f" ({lines[-1].strip()[:100]})" if lines else ""))
        with open(out_file) as fh:
            poses_text = fh.read()
        mol_poses, scores = poses_to_mols(poses_text, name)
        n_poses = mol_poses.GetNumConformers()
        if not scores or len(scores) != n_poses:
            raise LigandError(f"found {len(scores)} scores for {n_poses} poses in the program's output")
        order = sorted(range(n_poses), key=lambda i: scores[i])         # best (most negative) first
        pkd = []
        if job.get("rescore") == "nnscore2":
            try:
                pkd = nnscore2_pkd(job["receptor"], out_file)
            except (LigandError, subprocess.TimeoutExpired) as exc:
                row["note"] = str(exc) if isinstance(exc, LigandError) else "NNScore2 timed out"
        for rank, i in enumerate(order, 1):
            pose = Chem.Mol(mol_poses, confId=mol_poses.GetConformer(i).GetId())
            pose.SetProp("_Name", name)
            pose.SetProp("ligand", name)
            pose.SetProp("pose_rank", str(rank))
            pose.SetProp("score_kcal_mol", f"{scores[i]:.3f}")
            pose.SetProp("program", job["program"])
            if pkd and len(pkd) == n_poses:
                pose.SetProp("nnscore2_pkd", f"{pkd[i]:.3f}")
            block = Chem.MolToMolBlock(pose)
            props = [f">  <{k}>\n{pose.GetProp(k)}\n" for k in
                     ("ligand", "pose_rank", "score_kcal_mol", "program") + (("nnscore2_pkd",) if pose.HasProp("nnscore2_pkd") else ())]
            sd_block = block + "".join(p + "\n" for p in props) + "$$$$\n"
            all_blocks.append(sd_block)
        row.update(status="docked", best_score=f"{scores[order[0]]:.3f}", n_poses=n_poses,
                   pkd_best=f"{pkd[order[0]]:.3f}" if pkd and len(pkd) == n_poses else "",
                   note="; ".join(x for x in (clean_note, row["note"], f"{time.time() - started:.0f} s") if x))
        with open(os.path.join(folder, "poses.sdf"), "w") as fh:
            fh.write("".join(all_blocks))
        if job.get("clean"):
            for leftover in ("ligand.pdbqt", "poses.pdbqt", "program.log"):
                try:
                    os.remove(os.path.join(folder, leftover))
                except OSError:
                    pass
    except LigandError as exc:
        row["note"] = str(exc)
    except Exception as exc:                                       # a bug must not stop the other ligands
        row["note"] = f"unexpected error: {type(exc).__name__}: {exc}"
        with open(os.path.join(folder, "error.txt"), "w") as fh:
            fh.write(traceback.format_exc())
    if row["status"] == "failed":
        try:
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, "error.txt"), "a") as fh:
                fh.write(row["note"] + "\n")
        except OSError:
            pass
    return row


# --------------------------------------------------------------------------
# A run: settings, results on disk, resuming
# --------------------------------------------------------------------------


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def run_signature(args):
    """What must be identical for a run to be resumed: the inputs and everything that changes the scores."""
    return {"receptor": file_digest(args.receptor), "ligands": file_digest(args.ligands), "program": args.program,
            "center": [round(c, 3) for c in args.center], "size": [round(c, 3) for c in args.size],
            "exhaustiveness": args.exhaustiveness, "num_modes": args.num_modes, "seed": args.seed,
            "rescore": args.rescore or "", "add_hydrogens": bool(args.add_hydrogens),
            "ionisation": args.ionisation, "ph": args.ph if args.ionisation == "ph" else None}


def signature_differences(old, new):
    names = {"receptor": "the receptor", "ligands": "the ligand file", "program": "the docking program",
             "center": "the box centre", "size": "the box size", "exhaustiveness": "the exhaustiveness",
             "num_modes": "the number of poses", "seed": "the random seed", "rescore": "the re-scoring",
             "add_hydrogens": "the hydrogen setting", "ionisation": "the ionisation setting",
             "ph": "the ligand pH"}
    return [names[k] for k in names if old.get(k) != new.get(k)]


def read_results(outdir):
    """Results written so far, as {index: row}. The file is a journal while a run is going (one line appended per
    finished molecule, so a stopped job loses nothing); when the same molecule appears twice, the later line wins.
    Lines cut off by a killed job are ignored, and that molecule is simply docked again on resume."""
    path = os.path.join(outdir, SUMMARY_FILE)
    rows = {}
    if not os.path.isfile(path):
        return rows
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            index, status = row.get("index") or "", row.get("status") or ""
            if index.isdigit() and status and None not in row.values():
                rows[int(index)] = row
    return rows


class ResultsJournal:
    """Appends one result line per finished molecule: constant cost however long the run is (rewriting the whole
    table each time took 144 s of a 10,000-molecule run)."""

    def __init__(self, outdir, fresh):
        self.path = os.path.join(outdir, SUMMARY_FILE)
        if fresh or not os.path.isfile(self.path):
            with open(self.path, "w", newline="") as fh:
                csv.DictWriter(fh, fieldnames=RESULT_COLUMNS).writeheader()
        self.fh = open(self.path, "a", newline="")
        self.writer = csv.DictWriter(self.fh, fieldnames=RESULT_COLUMNS, extrasaction="ignore", restval="")

    def add(self, row):
        self.writer.writerow(row)
        self.fh.flush()

    def close(self):
        self.fh.close()


def write_results(outdir, rows):
    """The final table: one line per molecule, in input order (replaces the journal)."""
    path = os.path.join(outdir, SUMMARY_FILE)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_COLUMNS, extrasaction="ignore", restval="")
        writer.writeheader()
        for index in sorted(rows):
            writer.writerow(rows[index])
    os.replace(tmp, path)               # a job killed mid-write must not leave a half-written table


def ligand_folder(outdir, index):
    """The folder of one molecule. Folders are grouped by thousands (ligands/0000/000042): a single folder with
    100,000 subfolders is very slow on a shared file system."""
    return os.path.join(outdir, "ligands", f"{index // 1000:04d}", f"{index:06d}")


def write_sdf_files(outdir, rows_by_index):
    """top_poses.sdf and all_poses.sdf, rebuilt from the per-ligand folders (so a resumed run is complete)."""
    top_path, all_path = os.path.join(outdir, "top_poses.sdf"), os.path.join(outdir, "all_poses.sdf")
    with open(top_path, "w") as top, open(all_path, "w") as every:
        for index in sorted(rows_by_index):
            if rows_by_index[index].get("status") != "docked":
                continue
            sdf = os.path.join(ligand_folder(outdir, index), "poses.sdf")
            if not os.path.isfile(sdf):
                continue
            with open(sdf) as fh:
                text = fh.read()
            records = [r + "$$$$\n" for r in text.split("$$$$\n") if r.strip()]
            every.writelines(records)
            if records:
                top.write(records[0])


def sort_key(row):
    try:
        return (0, float(row.get("best_score")))
    except (TypeError, ValueError):
        return (1, 0.0)


def write_ranked_table(outdir, rows_by_index):
    """ranking.csv: docked ligands ordered from the best score, then the failed ones."""
    ordered = sorted(rows_by_index.values(), key=sort_key)
    with open(os.path.join(outdir, "ranking.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["rank"] + RESULT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for rank, row in enumerate(ordered, 1):
            writer.writerow(dict(row, rank=rank if row.get("status") == "docked" else ""))


def build_parser():
    parser = argparse.ArgumentParser(description="Dock a library of ligands with a Vina-family program.")
    parser.add_argument("--receptor", required=True, help="receptor PDB file (hydrogens added beforehand)")
    parser.add_argument("--ligands", required=True, help="ligand file: .sdf or SMILES (.smi/.txt/.csv)")
    parser.add_argument("--outdir", required=True, help="results folder")
    parser.add_argument("--center", required=True, nargs=3, type=float, metavar=("X", "Y", "Z"), help="box centre (A)")
    parser.add_argument("--size", nargs=3, type=float, metavar=("X", "Y", "Z"), default=[DEFAULT_BOX] * 3,
                        help="box size (A)")
    parser.add_argument("--program", choices=sorted(PROGRAMS), default=DEFAULT_PROGRAM)
    parser.add_argument("--exhaustiveness", type=int, default=DEFAULT_EXHAUSTIVENESS)
    parser.add_argument("--num_modes", type=int, default=DEFAULT_NUM_MODES, help="poses kept per ligand")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--rescore", choices=["nnscore2"], default=None, help="re-score the poses with NNScore2")
    parser.add_argument("--ionisation", choices=IONISATION_MODES, default="ph",
                        help="ph: assign protonation states at --ph (default); keep: use the charges in the file")
    parser.add_argument("--ph", type=float, default=DEFAULT_PH, help="pH for --ionisation ph (default 7.4)")
    parser.add_argument("--cpus", type=int, default=0, help="ligands docked at the same time (default: all CPUs the "
                                                            "job may use)")
    parser.add_argument("--adfr_path", default=DEFAULT_ADFR_PATH)
    parser.add_argument("--add_hydrogens", action="store_true",
                        help="let ADFR add hydrogens to the receptor if it has none")
    parser.add_argument("--clean", action="store_true",
                        help="delete each ligand's intermediate files (PDBQT, program log) once it is docked; "
                             "its poses.sdf is kept. Recommended for large libraries (fewer files)")
    parser.add_argument("--resume", action="store_true", help="continue a run in --outdir; finished ligands are skipped")
    parser.add_argument("--force", action="store_true", help="delete --outdir first if it exists")
    return parser


def available_cpus():
    """CPUs this process may use: SLURM's allocation if there is one, otherwise the machine's."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def validate_args(args):
    problems = []
    if args.exhaustiveness < 1:
        problems.append("--exhaustiveness must be at least 1")
    if args.num_modes < 1:
        problems.append("--num_modes must be at least 1")
    if any(s <= 0 for s in args.size):
        problems.append("--size values must be positive")
    if args.size and max(args.size) > 126:
        problems.append("--size above 126 A is not supported by the programs")
    for flag in ("receptor", "ligands"):
        path = getattr(args, flag)
        if not os.path.isfile(path):
            problems.append(f"--{flag}: file not found: {path}")
    if args.resume and args.force:
        problems.append("--resume and --force cannot be combined")
    if not 0.0 <= args.ph <= 14.0:
        problems.append("--ph must be between 0 and 14")
    return problems


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.receptor, args.ligands = os.path.abspath(args.receptor), os.path.abspath(args.ligands)
    args.outdir, args.adfr_path = os.path.abspath(args.outdir), os.path.abspath(args.adfr_path)
    problems = validate_args(args)
    if problems:
        print("Error:\n" + "\n".join(f"  - {p}" for p in problems), file=sys.stderr)
        return 2
    signature = run_signature(args)

    if os.path.isdir(args.outdir) and os.listdir(args.outdir):
        if args.resume:
            try:
                with open(os.path.join(args.outdir, SIGNATURE_FILE)) as fh:
                    old = json.load(fh)
            except (OSError, ValueError):
                print(f"Error: {args.outdir} holds no earlier run to resume.", file=sys.stderr)
                return 2
            differences = signature_differences(old, signature)
            if differences:
                print("Error: cannot resume, these differ from the earlier run: " + ", ".join(differences),
                      file=sys.stderr)
                return 2
        elif args.force:
            if not os.path.isfile(os.path.join(args.outdir, SIGNATURE_FILE)):
                print(f"Error: {args.outdir} was not made by this tool, so it will not be deleted. "
                      "Choose another folder or empty it yourself.", file=sys.stderr)
                return 2
            shutil.rmtree(args.outdir)
        else:
            print(f"Error: {args.outdir} is not empty. Use --resume or --force.", file=sys.stderr)
            return 2
    elif args.resume:
        print(f"Error: {args.outdir} holds no earlier run to resume.", file=sys.stderr)
        return 2
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, SIGNATURE_FILE), "w") as fh:
        json.dump(signature, fh, indent=2)

    print(f"Docking with {PROGRAMS[args.program]['title']}; box centre {args.center}, size {args.size}", flush=True)
    workdir = os.path.join(args.outdir, "receptor")
    receptor_pdbqt = os.path.join(workdir, "receptor.pdbqt")
    if not (args.resume and os.path.isfile(receptor_pdbqt)):
        try:
            prepare_receptor(workdir, args.receptor, args.adfr_path, args.add_hydrogens)
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
    print("Receptor prepared.", flush=True)

    total = count_ligands(args.ligands)
    if total == 0:
        print("Error: no molecules were found in the ligand file.", file=sys.stderr)
        return 2

    earlier = read_results(args.outdir) if args.resume else {}
    final = {i: r for i, r in earlier.items() if r.get("status") == "docked"}       # failed molecules are redone
    write_results(args.outdir, final)                  # a clean table of what is done; new lines are appended to it
    journal = ResultsJournal(args.outdir, fresh=False)
    workers = max(1, min(args.cpus or available_cpus(), total - len(final) or 1))
    print(f"{total} molecule(s): {len(final)} already done, {total - len(final)} to dock, {workers} at a time.",
          flush=True)

    def jobs():
        """Molecules still to do, read from the file one at a time (a library may hold millions)."""
        for index, name, mol, error in iter_ligands(args.ligands):
            if index in final:
                continue
            if mol is None:
                yield {"unreadable": True, "index": index, "name": name, "note": error}
                continue
            yield {"index": index, "name": name, "mol": mol.ToBinary(), "folder": ligand_folder(args.outdir, index),
                   "program": args.program, "receptor": receptor_pdbqt,
                   "box": {"center": args.center, "size": args.size}, "exhaustiveness": args.exhaustiveness,
                   "num_modes": args.num_modes, "seed": args.seed, "rescore": args.rescore, "clean": args.clean,
                   "ionisation": args.ionisation, "ph": args.ph}

    def record(row):
        final[row["index"]] = {k: row.get(k, "") for k in RESULT_COLUMNS}
        journal.add(final[row["index"]])

    def terminate(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)       # SLURM sends SIGTERM when the time limit is reached
    todo = total - len(final)
    done, started, last_report = 0, time.time(), time.time()
    broken = False
    pool = concurrent.futures.ProcessPoolExecutor(max_workers=workers, initializer=_worker_start)
    pending = {}                                    # future -> (index, name); never more than a few per worker
    stream = jobs()
    window = workers * 4
    try:
        while True:
            while len(pending) < window and not broken:
                job = next(stream, None)
                if job is None:
                    break
                if job.get("unreadable"):
                    record({"index": job["index"], "name": job["name"], "status": "failed", "note": job["note"]})
                    done += 1
                    print(f"[{done}/{todo}] {job['name']}: FAILED: {job['note']}", flush=True)
                    continue
                try:
                    pending[pool.submit(dock_one, job)] = (job["index"], job["name"])
                except concurrent.futures.BrokenExecutor:
                    broken = True
            if not pending:
                break
            finished, _ = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in finished:
                index, name = pending.pop(future)
                try:
                    row = future.result()
                except concurrent.futures.BrokenExecutor:
                    # A worker was killed (for example by the out-of-memory killer). Python then fails every queued
                    # molecule, including ones that did nothing wrong, so these are NOT recorded as failures: they
                    # are simply not done yet, and a resume docks them.
                    broken = True
                    continue
                except Exception as exc:                  # an error in this molecule's own worker
                    row = {"index": index, "name": name, "status": "failed",
                           "note": f"worker stopped: {type(exc).__name__}: {exc}"}
                record(row)
                done += 1
                failed_now = row["status"] != "docked"
                now = time.time()
                if failed_now or todo <= 200 or now - last_report >= 60:
                    shown = f"FAILED: {row['note']}" if failed_now else row["best_score"]
                    rate = done / max(1e-9, now - started)
                    eta = f", about {(todo - done) / rate / 60:.0f} min left" if done >= workers and todo > 200 else ""
                    print(f"[{done}/{todo}] {row['name']}: {shown}{eta}", flush=True)
                    last_report = now
            if broken:
                pending.clear()                           # the rest cannot complete: stop at once
                break
    except KeyboardInterrupt:
        pool.shutdown(wait=False, cancel_futures=True)
        journal.close()
        write_results(args.outdir, final)
        print(f"Stopped after {done} of {todo} molecule(s). Finished molecules are kept; run again with --resume.",
              file=sys.stderr, flush=True)
        return 130
    if broken:
        # Do not wait for the surviving workers: each may be minutes into docking a molecule whose result nobody
        # will collect. Stop them, keep everything that finished, and say how to carry on.
        pool.shutdown(wait=False, cancel_futures=True)
        for process in multiprocessing.active_children():     # the pool forgets its workers once it is broken
            try:
                process.kill()
            except Exception:
                pass
        journal.close()
        write_results(args.outdir, final)
        print(f"A worker process died (most likely killed for using too much memory), after {done} of {todo} "
              "molecule(s). Finished molecules are kept; run again with --resume, with fewer CPUs or more memory.",
              file=sys.stderr, flush=True)
        return 1
    pool.shutdown(wait=True, cancel_futures=True)
    journal.close()

    write_results(args.outdir, final)                  # sorted by input order, replacing the journal
    write_sdf_files(args.outdir, final)
    write_ranked_table(args.outdir, final)
    failed = [r for r in final.values() if r.get("status") != "docked"]
    print(f"\nFinished: {len(final) - len(failed)} docked, {len(failed)} failed. Results are in {args.outdir}",
          flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
