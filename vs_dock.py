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
import os
import re
import shutil
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
#   seconds_per_ligand: measured on one CPU at exhaustiveness 8 (used only to suggest a time limit)
#   unsupported: elements that the program refuses to read in a PDBQT file
PROGRAMS = {
    "qvina": {"title": "QuickVina 2.1", "seconds_per_ligand": 20, "unsupported": ("B", "Si", "Se"),
              "text": "fast Vina variant; a good default for screening"},
    "qvina-w": {"title": "QuickVina-W 1.1", "seconds_per_ligand": 25, "unsupported": ("B", "Si", "Se"),
                "text": "QuickVina for large boxes (blind docking)"},
    "vina": {"title": "AutoDock Vina 1.1.2", "seconds_per_ligand": 20, "unsupported": ("B", "Si", "Se"),
             "text": "the original Vina"},
    "AutodockVina_1.2": {"title": "AutoDock Vina 1.2.3", "seconds_per_ligand": 21, "unsupported": ("B", "Se"),
                         "text": "current Vina; also accepts silicon"},
    "PSOVina": {"title": "PSOVina 2.0", "seconds_per_ligand": 3, "unsupported": ("B", "Si", "Se"),
                "text": "particle-swarm search; very fast"},
    "smina": {"title": "smina (Oct 2019)", "seconds_per_ligand": 31, "unsupported": ("B", "Si", "Se"),
              "text": "Vina fork with extra scoring options"},
}
DEFAULT_PROGRAM = "qvina"
DEFAULT_EXHAUSTIVENESS = 8
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


def clean_molecule(mol):
    """Keep the largest fragment (drops salts and counter-ions) and neutralise simple charges.

    Returns (molecule, note) where note is "" or a sentence describing what was changed.
    """
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize

    notes = []
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if len(fragments) > 1:
        mol = max(fragments, key=lambda m: m.GetNumHeavyAtoms())
        notes.append(f"kept the largest of {len(fragments)} fragments")
    try:
        neutral = rdMolStandardize.Uncharger().uncharge(mol)
        if Chem.MolToSmiles(neutral) != Chem.MolToSmiles(mol):
            notes.append("neutralised charges")
        mol = neutral
    except Exception:
        pass
    return mol, "; ".join(notes)


def ligand_name(mol, index):
    if mol is not None and mol.HasProp("_Name") and mol.GetProp("_Name").strip():
        return mol.GetProp("_Name").strip()
    return f"molecule_{index + 1}"


def read_ligands(path):
    """Read a ligand file (.sdf, .smi, .smiles, .txt, .csv with SMILES). Returns [(name, mol_or_None, error)].

    SMILES files: one molecule per line, "SMILES name"; a header line is skipped; "," and tab also separate.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    rows = []
    if path.lower().endswith(".sdf"):
        for i, mol in enumerate(Chem.SDMolSupplier(path)):
            if mol is None:
                rows.append((f"molecule_{i + 1}", None, "RDKit could not read this molecule"))
            else:
                rows.append((ligand_name(mol, i), mol, ""))
        return rows
    with open(path, errors="replace") as fh:
        lines = [l.strip() for l in fh if l.strip() and not l.lstrip().startswith("#")]
    for i, line in enumerate(lines):
        parts = [p for p in re.split(r"[\s,;]+", line) if p]
        smiles = parts[0]
        name = parts[1] if len(parts) > 1 else f"molecule_{len(rows) + 1}"
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            if i == 0 and not rows:            # a header such as "smiles,name"
                continue
            rows.append((name, None, f"'{smiles[:40]}' is not a valid SMILES"))
            continue
        mol.SetProp("_Name", name)
        rows.append((name, mol, ""))
    return rows


def blocked_elements(mol, program):
    """Elements in the molecule that the chosen program cannot read."""
    bad = PROGRAMS[program]["unsupported"]
    return sorted({a.GetSymbol() for a in mol.GetAtoms() if a.GetSymbol() in bad})


def embed_ligand(mol, timeout=EMBED_TIMEOUT_SECONDS, seed=DEFAULT_SEED):
    """Add hydrogens and generate 3D coordinates. Raises LigandError if RDKit cannot (it only signals that through
    its return value and leaves a molecule without coordinates, which would be docked as a flat structure)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

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
            raise LigandError("RDKit could not generate 3D coordinates (strained or impossible geometry or "
                              "stereochemistry)")
    return mol_h


def ligand_to_pdbqt(mol, seed=DEFAULT_SEED):
    """Clean, embed and convert one molecule. Returns (pdbqt_text, note, cleaned_smiles)."""
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    from rdkit import Chem

    mol, note = clean_molecule(mol)
    cleaned_smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
    mol_h = embed_ligand(mol, seed=seed)
    try:
        setups = MoleculePreparation().prepare(mol_h)
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
    sentences. A ligand with problems is still reported by the job, but will fail there."""
    problems = []
    if mol is None:
        return False, ["could not be read"]
    elements = blocked_elements(mol, program)
    if elements:
        problems.append(f"contains {', '.join(elements)}, which {PROGRAMS[program]['title']} cannot read")
    heavy = mol.GetNumHeavyAtoms()
    if heavy < 3:
        problems.append(f"has only {heavy} heavy atom(s)")
    elif heavy > 80:
        problems.append(f"has {heavy} heavy atoms; docking such a large molecule is unreliable")
    from rdkit import Chem

    if len(Chem.GetMolFrags(mol)) > 1:
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


def dock_one(job):
    """Prepare, dock and (optionally) re-score one ligand. Runs in a worker process; never raises.

    job: dict with index, name, mol (RDKit binary), folder, and the run settings.
    Returns a result row (dict with RESULT_COLUMNS) plus 'mol_block_top' and 'mol_blocks_all' for the SDF files.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    warnings.filterwarnings("ignore")
    index, name, folder = job["index"], job["name"], job["folder"]
    row = {"index": index, "name": name, "status": "failed", "best_score": "", "n_poses": "", "pkd_best": "",
           "note": "", "smiles": ""}
    top_block, all_blocks = "", []
    try:
        os.makedirs(folder, exist_ok=True)
        mol = Chem.Mol(job["mol"])        # binary form: keeps stereochemistry that a molblock without coordinates loses
        row["smiles"] = Chem.MolToSmiles(Chem.RemoveHs(mol))
        blocked = blocked_elements(clean_molecule(mol)[0], job["program"])
        if blocked:
            raise LigandError(f"contains {', '.join(blocked)}, which {PROGRAMS[job['program']]['title']} cannot read")
        pdbqt, clean_note, row["smiles"] = ligand_to_pdbqt(mol, job["seed"])   # the SMILES that was docked
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
            if rank == 1:
                top_block = sd_block
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
    row["top_block"], row["all_blocks"] = top_block, all_blocks
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
            "rescore": args.rescore or "", "add_hydrogens": bool(args.add_hydrogens)}


def signature_differences(old, new):
    names = {"receptor": "the receptor", "ligands": "the ligand file", "program": "the docking program",
             "center": "the box centre", "size": "the box size", "exhaustiveness": "the exhaustiveness",
             "num_modes": "the number of poses", "seed": "the random seed", "rescore": "the re-scoring",
             "add_hydrogens": "the hydrogen setting"}
    return [names[k] for k in names if old.get(k) != new.get(k)]


def read_results(outdir):
    path = os.path.join(outdir, SUMMARY_FILE)
    if not os.path.isfile(path):
        return {}
    with open(path, newline="") as fh:
        return {int(r["index"]): r for r in csv.DictReader(fh) if r.get("index", "").isdigit()}


def write_results(outdir, rows):
    path = os.path.join(outdir, SUMMARY_FILE)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for index in sorted(rows):
            writer.writerow(rows[index])
    os.replace(tmp, path)               # a job killed mid-write must not leave a half-written table


def write_sdf_files(outdir, rows_by_index, folders):
    """top_poses.sdf and all_poses.sdf, rebuilt from the per-ligand folders (so a resumed run is complete)."""
    top_path, all_path = os.path.join(outdir, "top_poses.sdf"), os.path.join(outdir, "all_poses.sdf")
    with open(top_path, "w") as top, open(all_path, "w") as every:
        for index in sorted(rows_by_index):
            if rows_by_index[index].get("status") != "docked":
                continue
            sdf = os.path.join(folders[index], "poses.sdf")
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

    library = read_ligands(args.ligands)
    if not library:
        print("Error: no molecules were found in the ligand file.", file=sys.stderr)
        return 2
    # Folders are grouped by thousands (ligands/0000/000042): a single folder with 100,000 subfolders is very slow
    # on a shared file system.
    folders = {i: os.path.join(args.outdir, "ligands", f"{i // 1000:04d}", f"{i:06d}") for i in range(len(library))}
    rows = read_results(args.outdir) if args.resume else {}
    rows = {i: r for i, r in rows.items() if r.get("status") == "docked"}        # failed ligands are redone
    jobs, final = [], dict(rows)
    from rdkit import Chem

    for i, (name, mol, error) in enumerate(library):
        if i in rows:
            continue
        if mol is None:
            final[i] = {"index": i, "name": name, "status": "failed", "note": error}
            continue
        jobs.append({"index": i, "name": name, "mol": mol.ToBinary(), "folder": folders[i],
                     "program": args.program, "receptor": receptor_pdbqt,
                     "box": {"center": args.center, "size": args.size}, "exhaustiveness": args.exhaustiveness,
                     "num_modes": args.num_modes, "seed": args.seed, "rescore": args.rescore, "clean": args.clean})
    skipped = len(rows)
    workers = max(1, min(args.cpus or available_cpus(), len(jobs) or 1))
    print(f"{len(library)} ligand(s): {skipped} already done, {len(jobs)} to dock, {workers} at a time.", flush=True)

    done = 0
    write_results(args.outdir, final)
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(dock_one, job): job for job in jobs}
            for future in concurrent.futures.as_completed(futures):
                job = futures[future]
                try:
                    row = future.result()
                except Exception as exc:                          # the worker itself died (for example out of memory)
                    row = {"index": job["index"], "name": job["name"], "status": "failed",
                           "note": f"worker stopped: {type(exc).__name__}: {exc}"}
                final[row["index"]] = {k: row.get(k, "") for k in RESULT_COLUMNS}
                done += 1
                write_results(args.outdir, final)                  # after every ligand, so a killed job loses nothing
                shown = row["best_score"] if row["status"] == "docked" else f"FAILED: {row['note']}"
                print(f"[{done}/{len(jobs)}] {row['name']}: {shown}", flush=True)
    except KeyboardInterrupt:
        print("Interrupted. Finished ligands are kept; run again with --resume.", file=sys.stderr)
        return 130

    write_sdf_files(args.outdir, final, folders)
    write_ranked_table(args.outdir, final)
    failed = [r for r in final.values() if r.get("status") != "docked"]
    docked = len(final) - len(failed)
    print(f"\nFinished: {docked} docked, {len(failed)} failed. Results are in {args.outdir}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
