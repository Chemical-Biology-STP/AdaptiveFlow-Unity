#!/usr/bin/env python3
"""Staged screening with AdaptiveFlow (phase 1: your own library).

The molecules are prepared by AFLP (stereoisomers, tautomers, protonation, 3D), then docked by AFVS, both on SLURM, and
the best form of every molecule is ranked. The upstream code is in third_party/ and is not changed; this module writes
the configuration of each run into its run folder and drives the stages.

A run is a state machine kept in <run>/state.json. Each stage ends by submitting a small `advance` job that waits for
the stage's SLURM jobs (--dependency=afterany) and then checks their output and submits the next stage, so nothing has
to stay running. `advance` can also be started by hand to resume a run that stopped.

    vs_staged.py advance <run folder>     the next step (normally called by the pipeline itself)
    vs_staged.py status <run folder>
    vs_staged.py results <run folder> [--top N]
    vs_staged.py redock <run folder> [--top N]   dock the best molecules again with poses kept (full mode)

AFLP and AFVS: https://github.com/QuantumAI4Bio (GNU GPL v2). See third_party/VERSIONS.md.
"""

import argparse
import csv
import fcntl
import glob
import gzip
import hashlib
import io
import json
import math
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

import vs_dock  # noqa: E402
import vs_mlfilter  # noqa: E402
import vs_slurm  # noqa: E402

THIRD_PARTY = os.path.join(REPO_DIR, "third_party")
STATE_FILE = "state.json"
LOCK_FILE = "state.lock"

# AFVS names for the programs of this tool. The programs themselves are the ones in executables/ (linked per run).
PROGRAM_METHODS = {"qvina": "qvina02", "qvina-w": "qvina_w", "vina": "vina", "AutodockVina_1.2": "AutodockVina_1.2",
                   "smina": "smina", "PSOVina": "PSOVina"}
PROGRAM_BINARIES = {"qvina02": "qvina", "qvina_w": "qvina-w", "vina": "vina", "AutodockVina_1.2": "AutodockVina_1.2",
                    "smina": "smina", "PSOVina": "PSOVina"}

SUGGESTED_CPUS = 500            # suggested number of CPUs working at the same time
FALLBACK_MAX_CPUS = 3000        # the `ncpu` QOS limit per user, used when the cluster cannot be asked
CONFIRM_CPUS = 1000             # above this the assistant asks for confirmation
CONFIRM_CPU_HOURS = 100000
SUBJOB_CPUS = 32                # CPUs of one SLURM array task (a whole node holds several)
SUBJOB_MINUTES = 25             # a subjob should take about this long
MOLECULES_PER_COLLECTION = 500
AFLP_SECONDS_PER_MOLECULE = 10  # CPU seconds to prepare one molecule (measured: 6 to 10 on 200 Enamine molecules)
FORMS_MEASURED = 4.57           # forms per molecule with AFLP's defaults (5 stereoisomers, 5 tautomers): measured
MAX_ARRAY = 1000                # SLURM array tasks per workunit (the cluster allows 10,000)
TIME_SAFETY = 3.0
MIN_TIME_MINUTES = 60
MAX_ARRAY_SUBJOBS_PER_WORKUNIT = MAX_ARRAY
SCENARIO = "ds1"

# --- the tranche strategy (the ATG screen of AdaptiveFlow) -----------------------------------------------------
STRATEGIES = ("all", "tranche")        # all: dock every prepared form; tranche: prescreen tranches, dock the best
DEFAULT_REPRESENTATIVES = 3            # ligands per tranche docked in the prescreen (AdaptiveFlow allows 1 to 10)
MAX_REPRESENTATIVES = 10
TRANCHE_PROPERTY_NAMES = ("mw", "logp", "hba", "hbd", "tpsa", "atomcount", "bondcount", "mr", "ringcount",
                          "aromaticringcount", "formalcharge", "positivechargecount", "negativechargecount", "fsp3",
                          "chiralcentercount", "halogencount", "sulfurcount", "NOcount")      # 18, the order of the ID
DEFAULT_TRANCHE_PROPERTIES = ("mw", "logp", "hba", "hbd")
TRANCHE_COLLECTION_SIZE = 500          # forms per collection of a tranche
SPARSE_COLLECTION_SIZE = 400           # representatives per sparse collection
ESTIMATE_SAMPLE = 3000                 # molecules used to estimate the size of the prescreen
TRANCHE_LETTERS = "ABCDEF"
DEFAULT_PRIMARY_FRACTION = 0.2          # share of the library the primary screen docks when none is chosen
DEFAULT_ML_CUTOFF = 0.5               # forms with a lower predicted probability are not docked (AdaptiveFlow's default)
ML_MIN_TRAIN = 200                     # prescreen scores needed to train the neural-network filter
ML_MIN_AUC = 0.6                       # a model worse than this on forms it never saw is not used
ML_KEEP_ESTIMATE = 0.5                 # share of the selected forms the filter is expected to keep (measured: 0.49)
ML_MEMORY_GB = 6                       # the advance job trains and applies the filter
TRANCHE_MEMORY_GB_PER_MILLION = 3.0    # building the tranche library keeps the compressed forms in memory


class StagedError(Exception):
    """Something the user must fix; the message says what, in plain words."""


# --------------------------------------------------------------------------
# Sizes and cost
# --------------------------------------------------------------------------


def cluster_cpu_limit(partition=vs_slurm.DEFAULT_PARTITION):
    """CPUs one user may use at the same time in this partition's QOS (3,000 for ncpu), or the fallback."""
    try:
        qos = subprocess.run(["scontrol", "show", "partition", partition], capture_output=True, text=True,
                             timeout=20).stdout
        match = re.search(r"\bQoS=(\S+)", qos)
        name = match.group(1) if match and match.group(1) != "N/A" else partition
        out = subprocess.run(["sacctmgr", "-n", "-P", "show", "qos", name, "format=MaxTRESPU"], capture_output=True,
                             text=True, timeout=20).stdout
        found = re.search(r"cpu=(\d+)", out)
        if found:
            return int(found.group(1))
    except (OSError, subprocess.SubprocessError):
        pass
    return FALLBACK_MAX_CPUS


def seconds_per_form(program, exhaustiveness):
    return vs_dock.PROGRAMS[program]["seconds_per_ligand"] * exhaustiveness / vs_slurm.MEASURED_AT_EXHAUSTIVENESS


def forms_per_molecule(max_stereoisomers, max_tautomers):
    """Forms docked per molecule. 1 and 1 give exactly one; the measured 4.57 holds for 5 and 5 (AFLP's defaults);
    in between it is an interpolation, shown only as an estimate."""
    if max_stereoisomers <= 1 and max_tautomers <= 1:
        return 1.0
    return max(1.0, min(FORMS_MEASURED, 1 + (FORMS_MEASURED - 1) * ((max_stereoisomers * max_tautomers) - 1) / 24))


def plan_sizes(n_molecules, program, exhaustiveness, max_stereoisomers, max_tautomers, cpus, strategy="all",
               primary_size=None, prescreen_forms=None, ml_filter=False):
    """Numbers shown on the review page and used to configure the jobs.

    strategy "all" docks every form. "tranche" docks the prescreen (`prescreen_forms`, estimated from a sample) and the
    primary screen (`primary_size` molecules); n_forms stays the number of forms AFLP prepares. With `ml_filter` the
    primary screen docks about ML_KEEP_ESTIMATE of the selected forms (the real share is known when it starts)."""
    subjob_cpus = max(1, min(SUBJOB_CPUS, cpus))
    forms = forms_per_molecule(max_stereoisomers, max_tautomers)
    n_forms = int(math.ceil(n_molecules * forms))
    if strategy == "tranche":
        primary_forms = min(n_forms, int(math.ceil((primary_size or n_molecules) * forms)))
        if ml_filter:
            primary_forms = max(1, int(math.ceil(primary_forms * ML_KEEP_ESTIMATE)))
        dock_forms = min(n_forms, int(prescreen_forms or 0) + primary_forms)
    else:
        primary_forms, dock_forms = n_forms, n_forms
    sec_form = seconds_per_form(program, exhaustiveness)
    dock_per_subjob = max(1, int(subjob_cpus * SUBJOB_MINUTES * 60 / sec_form))
    prep_per_subjob = max(MOLECULES_PER_COLLECTION,
                          int(subjob_cpus * SUBJOB_MINUTES * 60 / AFLP_SECONDS_PER_MOLECULE))
    dock_subjobs = max(1, math.ceil(dock_forms / dock_per_subjob))
    prep_subjobs = max(1, math.ceil(n_molecules / prep_per_subjob))
    running = max(1, min(cpus // subjob_cpus or 1, dock_subjobs))
    cpu_hours = (dock_forms * sec_form + n_molecules * AFLP_SECONDS_PER_MOLECULE) / 3600
    return {
        "strategy": strategy, "ml_filter": bool(ml_filter), "subjob_cpus": subjob_cpus, "forms_per_molecule": forms, "n_forms": n_forms,
        "dock_forms": dock_forms, "primary_forms": primary_forms, "prescreen_forms": int(prescreen_forms or 0),
        "seconds_per_form": sec_form,
        "dockings_per_subjob": dock_per_subjob, "molecules_per_prep_subjob": prep_per_subjob,
        "dock_subjobs": dock_subjobs, "prep_subjobs": prep_subjobs, "subjobs_at_once": running,
        "dock_cpu_hours": dock_forms * sec_form / 3600, "prep_cpu_hours": n_molecules * AFLP_SECONDS_PER_MOLECULE / 3600,
        "cpu_hours": cpu_hours, "wall_hours": cpu_hours / max(1, running * subjob_cpus),
        "gb": dock_forms * 0.00006 + n_molecules * 0.00002 + (n_forms * 0.00004 if strategy == "tranche" else 0),
    }


def stage_sizes(n_forms, plan):
    """Sizes of one docking stage that has `n_forms` forms to dock (the counts are known when the stage starts)."""
    return plan_sizes(max(1, n_forms), plan["program"], plan["exhaustiveness"], 1, 1, plan["cpus"])


def minutes_for_subjob(work_cpu_seconds, subjob_cpus):
    minutes = work_cpu_seconds / subjob_cpus / 60 * TIME_SAFETY
    return int(min(vs_slurm.MAX_TIME_MINUTES, max(MIN_TIME_MINUTES, math.ceil(minutes))))



def tranche_dimensions():
    """The 18 properties of the tranche ID: {name: (function of an RDKit molecule, class cutoffs)}. A value up to the
    first cutoff is class A, up to the second B, and so on (at most six classes, A to F). The cutoffs are AdaptiveFlow's
    defaults."""
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

    def aromatic_rings(m):
        return sum(1 for ring in m.GetRingInfo().AtomRings() if all(m.GetAtomWithIdx(i).GetIsAromatic() for i in ring))

    return {
        "mw": (lambda m: Descriptors.MolWt(m), [200, 300, 400, 450]),
        "logp": (lambda m: Crippen.MolLogP(m), [0, 2, 3, 4, 5]),
        "hba": (lambda m: Lipinski.NumHAcceptors(m), [2, 4, 7, 10]),
        "hbd": (lambda m: Lipinski.NumHDonors(m), [0, 2, 4, 5]),
        "tpsa": (lambda m: rdMolDescriptors.CalcTPSA(m), [40, 80, 110, 140]),
        "atomcount": (lambda m: Chem.AddHs(m).GetNumAtoms(), [10, 20, 30, 40, 50]),
        "bondcount": (lambda m: Chem.AddHs(m).GetNumBonds(), [10, 20, 30, 40, 50]),
        "mr": (lambda m: Crippen.MolMR(m), [40, 80, 130]),
        "ringcount": (lambda m: rdMolDescriptors.CalcNumRings(m) - aromatic_rings(m), [0, 1, 2, 3, 4]),
        "aromaticringcount": (aromatic_rings, [0, 1, 2, 3, 4]),
        "formalcharge": (lambda m: Chem.GetFormalCharge(m), [-2, -1, 0, 1]),
        "positivechargecount": (lambda m: sum(1 for a in m.GetAtoms() if a.GetFormalCharge() > 0), [0, 1]),
        "negativechargecount": (lambda m: sum(1 for a in m.GetAtoms() if a.GetFormalCharge() < 0), [0, 1]),
        "fsp3": (lambda m: rdMolDescriptors.CalcFractionCSP3(m), [0.2, 0.4, 0.6, 0.8]),
        "chiralcentercount": (lambda m: len(Chem.FindMolChiralCenters(m, includeUnassigned=True)), [0, 1]),
        "halogencount": (lambda m: sum(a.GetSymbol() in ("F", "Cl", "Br", "I") for a in m.GetAtoms()), [0]),
        "sulfurcount": (lambda m: sum(a.GetSymbol() == "S" for a in m.GetAtoms()), [0]),
        "NOcount": (lambda m: sum(a.GetSymbol() in ("N", "O") for a in m.GetAtoms()), [2, 4, 6, 8, 10]),
    }


def tranche_letter(value, cutoffs):
    last = len(TRANCHE_LETTERS) - 1
    for index, cutoff in enumerate(cutoffs):
        if value <= cutoff:
            return TRANCHE_LETTERS[min(index, last)]
    return TRANCHE_LETTERS[min(len(cutoffs), last)]


def tranche_of(mol, properties=DEFAULT_TRANCHE_PROPERTIES, dimensions=None):
    """The 18-letter tranche ID of a molecule. Only the chosen properties are measured; the other positions are always
    A, so the library falls into few tranches that each hold many molecules (the real library has about 5,600 per
    tranche). AFVS and AdaptiveFlow's scripts expect 18 letters."""
    dimensions = dimensions or tranche_dimensions()
    letters = []
    for name in TRANCHE_PROPERTY_NAMES:
        if name in properties:
            function, cutoffs = dimensions[name]
            letters.append(tranche_letter(function(mol), cutoffs))
        else:
            letters.append("A")
    return "".join(letters)


def sample_molecules(source, total, wanted=ESTIMATE_SAMPLE):
    """About `wanted` RDKit molecules spread evenly over a library. `source` is a molecule file, or the names.tsv that
    a run writes (columns id, name, smiles); `total` is the number of molecules in it."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    stride = max(1, int(total) // wanted)
    molecules = []
    if source.lower().endswith(".tsv"):
        with open(source) as fh:
            for index, row in enumerate(csv.DictReader(fh, delimiter="\t")):
                if index % stride == 0:
                    mol = Chem.MolFromSmiles(row["smiles"])
                    if mol is not None:
                        molecules.append(mol)
    else:
        for index, _name, mol, _error in vs_dock.iter_ligands(source):
            if index % stride == 0 and mol is not None:
                molecules.append(mol)
    return molecules


def estimate_prescreen(source, n_molecules, forms_per_mol, properties, representatives, wanted=ESTIMATE_SAMPLE):
    """Estimate the number of tranches and of prescreen dockings from a sample of the library. The prescreen docks up to
    `representatives` forms of every tranche. The number of tranches of the whole library is estimated from the
    sample with the bias-corrected Chao1 estimator (observed tranches plus a term from those seen once and twice), so
    the estimate is the prescreen size for a library of this size and not just the sample's share scaled up. The result
    is capped at the number of forms. Exact when the sample is the whole library."""
    n_forms = int(math.ceil(n_molecules * forms_per_mol))
    molecules = sample_molecules(source, n_molecules, wanted)
    if not molecules:
        return {"sample": 0, "tranches_in_sample": 0, "tranches": 0, "prescreen_forms": n_forms, "exact": False}
    dimensions = tranche_dimensions()
    counts = {}
    for mol in molecules:
        tranche = tranche_of(mol, properties, dimensions)
        counts[tranche] = counts.get(tranche, 0) + 1
    exact = len(molecules) >= n_molecules
    seen = len(counts)
    once = sum(1 for c in counts.values() if c == 1)
    twice = sum(1 for c in counts.values() if c == 2)
    tranches = seen if exact else int(math.ceil(seen + once * once / (2.0 * (twice + 1))))
    if exact:
        prescreen = int(math.ceil(sum(min(c, representatives) for c in counts.values()) * forms_per_mol))
    else:
        prescreen = int(math.ceil(tranches * representatives))
    return {"sample": len(molecules), "tranches_in_sample": seen, "tranches": tranches,
            "prescreen_forms": min(n_forms, prescreen), "exact": exact}


def base26(number, width=1):
    """0 -> A, 25 -> Z, 26 -> BA ...; padded with A to `width` letters."""
    letters = ""
    while True:
        number, rest = divmod(number, 26)
        letters = chr(65 + rest) + letters
        if number == 0:
            break
    return letters.rjust(width, "A")


# --------------------------------------------------------------------------
# Library: SMILES collections for AFLP
# --------------------------------------------------------------------------


def molecule_id(index):
    return f"M{index:09d}"


def write_collections(library_path, collections_dir, names_path, per_collection=MOLECULES_PER_COLLECTION):
    """Cut the library into AFLP collections (<meta>/<tranche>/<collection>.txt.gz of 'SMILES<TAB>id') and write
    names.tsv (id, original name, input SMILES). Molecules that RDKit cannot read are skipped and listed in the
    returned `unreadable`. Returns (n_molecules, todo_lines, unreadable)."""
    from rdkit import Chem

    os.makedirs(collections_dir, exist_ok=True)
    todo, unreadable = [], []
    collection, in_collection, n = 0, [], 0

    def flush(rows, number):
        tranche = base26(number // 1000, 3)
        key = base26(number % 1000, 2)
        folder = os.path.join(collections_dir, "AA", tranche)
        os.makedirs(folder, exist_ok=True)
        with gzip.open(os.path.join(folder, f"{key}.txt.gz"), "wt") as fh:
            for smiles, ident in rows:
                fh.write(f"{smiles}\t{ident}\n")
        todo.append(f"AA_{tranche}_{key} {len(rows)}")

    with open(names_path, "w", newline="") as names:
        writer = csv.writer(names, delimiter="\t")
        writer.writerow(["id", "name", "smiles"])
        for index, name, mol, error in vs_dock.iter_ligands(library_path):
            if mol is None:
                unreadable.append((index, name, error))
                continue
            ident = molecule_id(n)
            smiles = Chem.MolToSmiles(mol)
            writer.writerow([ident, name, smiles])
            in_collection.append((smiles, ident))
            n += 1
            if len(in_collection) == per_collection:
                flush(in_collection, collection)
                collection, in_collection = collection + 1, []
        if in_collection:
            flush(in_collection, collection)
    return n, todo, unreadable


# --------------------------------------------------------------------------
# Configuration files of AFLP and AFVS
# --------------------------------------------------------------------------


def set_ctrl(text, key, value):
    """Set `key=value` in an all.ctrl text. Stops if the key is not there (an upstream change)."""
    pattern = re.compile(rf"^{re.escape(key)}=.*$", re.M)
    if not pattern.search(text):
        raise StagedError(f"The AdaptiveFlow template has no '{key}=' line. third_party/ was probably updated; "
                          "vs_staged.py must be adapted.")
    return pattern.sub(lambda _m: f"{key}={value}", text, count=1)


def patch_slurm_template(text, env_bin, minutes, mem_per_cpu=vs_slurm.MEM_PER_CPU, name="template"):
    """The upstream job template with the cluster-specific lines replaced: no account, no virtualenv, conda or modules,
    the tool's own environment first on the PATH, a time limit and a memory request."""
    def sub(pattern, repl, text, required=True):
        new, n = re.subn(pattern, repl, text, flags=re.M)
        if required and n == 0:
            raise StagedError(f"The AdaptiveFlow {name} has no line matching {pattern!r}. third_party/ was probably "
                              "updated; vs_staged.py must be adapted.")
        return new

    text = sub(r"^#SBATCH --account=.*\n", "", text)
    text = sub(r"^##SBATCH --time=.*$", f"#SBATCH --time={vs_slurm_time(minutes)}", text)
    text = sub(r"^##SBATCH --mem-per-cpu=.*$", f"#SBATCH --mem-per-cpu={mem_per_cpu}", text)
    text = sub(r"^##SBATCH --partition=", "#SBATCH --partition=", text, required=False)
    path_line = f"export PATH={env_bin}:$PATH"
    if re.search(r"^source \$HOME/\S*activate$", text, re.M):
        text = sub(r"^source \$HOME/\S*activate$", path_line, text)
    else:
        text = sub(r"^source \$HOME/\S*conda\.sh$", path_line, text)
    text = sub(r"^conda activate.*\n", "", text, required=False)
    text = sub(r"^module load.*\n", "", text, required=False)
    text = sub(r"^export AWS_\w+=\s*\n", "", text, required=False)
    for banned in ("--account", "conda activate", "module load"):
        if re.search(rf"^{re.escape(banned)}", text, re.M) or re.search(rf"^#SBATCH {re.escape(banned)}", text, re.M):
            raise StagedError(f"The patched {name} still contains '{banned}'.")
    return text


def vs_slurm_time(minutes):
    minutes = int(minutes)
    days, rest = divmod(minutes, 1440)
    hours, mins = divmod(rest, 60)
    return f"{days}-{hours:02d}:{mins:02d}:00" if days else f"{hours:02d}:{mins:02d}:00"


def link_programs(bin_dir, methods):
    """tools/bin of the run: the docking programs of this tool under the names AFVS expects."""
    os.makedirs(bin_dir, exist_ok=True)
    for method in methods:
        target = os.path.join(vs_dock.EXECUTABLES, PROGRAM_BINARIES[method])
        if not os.path.isfile(target):
            raise StagedError(f"The docking program {PROGRAM_BINARIES[method]} is missing from {vs_dock.EXECUTABLES}.")
        link = os.path.join(bin_dir, method)
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(target, link)


def copy_tools(tool, destination):
    """A working copy of the vendored tools/ folder of AFVS or AFLP inside the run folder."""
    source = os.path.join(THIRD_PARTY, tool, "tools")
    if not os.path.isdir(source):
        raise StagedError(f"third_party/{tool}/tools is missing.")
    shutil.copytree(source, destination, symlinks=True)


def configure_aflp(run_dir, plan, sizes, todo_lines):
    """AFLP working copy: run_dir/aflp/tools with all.ctrl, todo.all and the job template written for this run."""
    work = os.path.join(run_dir, "aflp")
    copy_tools("AFLP", os.path.join(work, "tools"))
    tpl = os.path.join(work, "tools", "templates")
    ctrl = open(os.path.join(tpl, "all.ctrl")).read()
    cpus = sizes["subjob_cpus"]
    values = {
        "job_name": "vs_prep", "threads_to_use": cpus, "batchsystem": "slurm",
        "slurm_partition": plan["partition"], "slurm_cpus": cpus,
        "slurm_array_job_size": min(sizes["prep_subjobs"], MAX_ARRAY),
        "slurm_array_job_throttle": min(sizes["subjobs_at_once"], MAX_ARRAY),
        "job_storage_mode": "sharedfs",
        "collection_folder": os.path.join(run_dir, "prep", "collections"),
        "ligands_todo_per_queue": sizes["molecules_per_prep_subjob"],
        "conformation_program_1": "rdkit", "conformation_program_2": "none",
        "rdkit_stereoisomer_max_count": plan["max_stereoisomers"], "rdkit_tautomer_max_count": plan["max_tautomers"],
        "protonation_pH_value": plan["ph"], "targetformats": "pdbqt",
    }
    for key, value in values.items():
        ctrl = set_ctrl(ctrl, key, value)
    open(os.path.join(tpl, "all.ctrl"), "w").write(ctrl)
    open(os.path.join(tpl, "todo.all"), "w").write("\n".join(todo_lines) + "\n")
    minutes = minutes_for_subjob(sizes["molecules_per_prep_subjob"] * AFLP_SECONDS_PER_MOLECULE, cpus)
    template = os.path.join(tpl, "template1.slurm.sh")
    patched = patch_slurm_template(open(template).read(), os.path.dirname(sys.executable), minutes,
                                   name="AFLP job template")
    open(template, "w").write(patched)
    return work


def configure_afvs(run_dir, plan, sizes, work_name="afvs", collections_dir=None, representatives=None):
    """AFVS working copy: run_dir/<work_name>/tools with all.ctrl and the docking scenario. todo.all is written later.

    representatives: set for the prescreen of the tranche strategy; AFVS then docks only that many ligands of each tranche
    that the sparse collections list."""
    work = os.path.join(run_dir, work_name)
    if os.path.isdir(work):
        shutil.rmtree(work)
    copy_tools("AFVS", os.path.join(work, "tools"))
    method = PROGRAM_METHODS[plan["program"]]
    link_programs(os.path.join(work, "tools", "bin"), [method])
    scenario = os.path.join(work, "input-files", SCENARIO)
    os.makedirs(scenario, exist_ok=True)
    shutil.copyfile(os.path.join(run_dir, "input", "receptor.pdbqt"), os.path.join(scenario, "receptor.pdbqt"))
    cx, cy, cz = plan["center"]
    size = plan["box_size"]
    with open(os.path.join(scenario, "config.txt"), "w") as fh:
        fh.write(f"receptor=../input-files/{SCENARIO}/receptor.pdbqt\n"
                 f"center_x={cx:.3f}\ncenter_y={cy:.3f}\ncenter_z={cz:.3f}\n"
                 f"size_x={size:.3f}\nsize_y={size:.3f}\nsize_z={size:.3f}\n"
                 f"exhaustiveness={plan['exhaustiveness']}\nseed={plan['seed']}\n")
    tpl = os.path.join(work, "tools", "templates")
    ctrl = open(os.path.join(tpl, "all.ctrl")).read()
    cpus = sizes["subjob_cpus"]
    timeout = int(max(900, 10 * sizes["seconds_per_form"]))
    values = {
        "job_name": f"vs_{work_name[5:] or 'dock'}"[:20], "threads_to_use": cpus, "program_timeout": timeout,
        "batchsystem": "slurm",
        "slurm_partition": plan["partition"], "slurm_cpus": cpus,
        "slurm_array_job_size": min(sizes["dock_subjobs"], MAX_ARRAY),
        "slurm_array_job_throttle": min(sizes["subjobs_at_once"], MAX_ARRAY),
        "data_storage_mode": "sharedfs", "job_storage_mode": "sharedfs",
        "data_collection_addressing_mode": "metatranche", "data_collection_identifier": "",
        "collection_folder": collections_dir or os.path.join(run_dir, "dock", "collections"),
        "dockings_per_subjob": sizes["dockings_per_subjob"], "dynamic_tranche_filtering": 0,
        "docking_scenario_names": SCENARIO, "docking_scenario_methods": method,
        "prescreen_mode": 1 if representatives else 0,
    }
    if representatives:
        values["prescreen_ligands_per_tranche"] = representatives
    for key, value in values.items():
        ctrl = set_ctrl(ctrl, key, value)
    open(os.path.join(tpl, "all.ctrl"), "w").write(ctrl)
    minutes = minutes_for_subjob(sizes["dockings_per_subjob"] * sizes["seconds_per_form"], cpus)
    template = os.path.join(tpl, "template1.slurm.sh")
    patched = patch_slurm_template(open(template).read(), os.path.dirname(sys.executable), minutes,
                                   name="AFVS job template")
    open(template, "w").write(patched)
    return work


# --------------------------------------------------------------------------
# State, locking, SLURM helpers
# --------------------------------------------------------------------------


def state_path(run_dir):
    return os.path.join(run_dir, STATE_FILE)


def load_state(run_dir):
    try:
        with open(state_path(run_dir)) as fh:
            return json.load(fh)
    except FileNotFoundError:
        raise StagedError(f"{run_dir} is not a staged run folder (no {STATE_FILE}).")


def save_state(run_dir, state):
    tmp = state_path(run_dir) + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(state, fh, indent=1)
    os.replace(tmp, state_path(run_dir))


def log_event(state, text):
    state.setdefault("events", []).append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "text": text})
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


class Lock:
    """Only one `advance` at a time per run (a manual one must not collide with the pipeline's own)."""

    def __init__(self, run_dir):
        self.path = os.path.join(run_dir, LOCK_FILE)
        self.fh = None

    def __enter__(self):
        self.fh = open(self.path, "w")
        try:
            fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.fh.close()
            raise StagedError("Another `advance` is working on this run right now. Try again in a minute.")
        return self

    def __exit__(self, *exc):
        fcntl.flock(self.fh, fcntl.LOCK_UN)
        self.fh.close()


def job_states(job_id):
    """States of all tasks of a job (an array has one per task), from squeue and sacct."""
    states = []
    try:
        out = subprocess.run(["squeue", "-h", "-j", str(job_id), "-o", "%T"], capture_output=True, text=True,
                             timeout=60).stdout
        states += [s for s in out.split() if s]
        out = subprocess.run(["sacct", "-j", str(job_id), "-X", "-n", "-P", "-o", "State"], capture_output=True,
                             text=True, timeout=60).stdout
        states += [re.split(r"[ +]", s)[0] for s in out.split("\n") if s.strip()]
    except (OSError, subprocess.SubprocessError):
        pass
    return states


ACTIVE = {"PENDING", "RUNNING", "REQUEUED", "RESIZING", "SUSPENDED", "COMPLETING", "CONFIGURING"}


def job_unfinished(job_id):
    return any(s in ACTIVE for s in job_states(job_id))


def job_failures(job_id):
    """Tasks of a finished job that did not complete, as {state: count}."""
    counts = {}
    for s in job_states(job_id):
        if s not in ACTIVE and s != "COMPLETED":
            counts[s] = counts.get(s, 0) + 1
    return counts


def submit_advance(run_dir, state, after_job, final=False):
    """Queue the next `advance` to start when `after_job` has ended (or at once if after_job is None)."""
    script = os.path.join(run_dir, "logs", "advance.sh")
    os.makedirs(os.path.dirname(script), exist_ok=True)
    plan = state["plan"]
    mail = vs_slurm.mail_address()
    big = state["counts"]["molecules"]
    mem_gb = big / 1e6 * 1.3                                  # the ranking holds about 1.1 GB per million molecules
    if plan.get("strategy") == "tranche":                     # building the tranche library holds the forms
        mem_gb += state["sizes"]["n_forms"] / 1e6 * TRANCHE_MEMORY_GB_PER_MILLION
        if plan.get("ml_filter"):
            mem_gb += ML_MEMORY_GB
    mem_gb = int(math.ceil(mem_gb)) + 4
    lines = ["#!/bin/bash", f"#SBATCH --job-name=vs_advance", f"#SBATCH --partition={plan['partition']}",
             "#SBATCH --cpus-per-task=4", f"#SBATCH --mem={mem_gb}G", "#SBATCH --time=1-00:00:00",
             f"#SBATCH --output={os.path.join(run_dir, 'logs', 'advance_%j.log')}"]
    if mail:
        lines += [f"#SBATCH --mail-type={'END,FAIL' if final else 'FAIL'}", f"#SBATCH --mail-user={mail}"]
    lines += [f"cd {run_dir}", f"{sys.executable} -W ignore {os.path.join(REPO_DIR, 'vs_staged.py')} advance {run_dir}"]
    with open(script, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    command = ["sbatch", "--parsable"]
    if after_job:
        command += [f"--dependency=afterany:{after_job}"]
    result = subprocess.run(command + [script], capture_output=True, text=True, timeout=120)
    if result.returncode != 0 or not result.stdout.strip():
        raise StagedError("sbatch did not accept the job: " + (result.stderr.strip() or result.stdout.strip()))
    job_id = result.stdout.strip().split(";")[0]
    state["advance_job"] = job_id
    return job_id


# --------------------------------------------------------------------------
# Running the upstream scripts
# --------------------------------------------------------------------------


def upstream(tool_dir, script, *args):
    """Run an upstream script from its tools/ folder with the tool's environment first on the PATH."""
    env = dict(os.environ)
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
    result = subprocess.run([sys.executable, "-W", "ignore", script, *args], cwd=os.path.join(tool_dir, "tools"),
                            env=env, capture_output=True, text=True, input="y\n")
    if result.returncode != 0:
        tail = "\n".join((result.stderr or result.stdout).strip().splitlines()[-8:])
        raise StagedError(f"{script} failed:\n{tail}")
    return result.stdout


def submitted_job_id(tool_dir, workunit):
    path = os.path.join(tool_dir, "workflow", "status.submission.json")
    with open(path) as fh:
        data = json.load(fh)
    try:
        return int(data["workunits"][str(workunit)]["status"]["job_id"])
    except (KeyError, TypeError, ValueError):
        raise StagedError(f"Could not find the job id of work unit {workunit} in {path}.")


def count_workunits(output):
    match = re.search(r"Generated (\d+) workunits", output)
    if not match:
        raise StagedError("The preparation of the work units printed no count:\n" + output[-300:])
    return int(match.group(1))


# --------------------------------------------------------------------------
# Start a run
# --------------------------------------------------------------------------

REQUIRED_PLAN = ("run_dir", "receptor", "center", "box_size", "ligands", "program", "exhaustiveness", "seed", "ph",
                 "max_stereoisomers", "max_tautomers", "cpus", "partition", "job_name")


def check_plan(plan):
    for key in REQUIRED_PLAN:
        if key not in plan:
            raise StagedError(f"The plan has no '{key}'.")
    if plan["program"] not in PROGRAM_METHODS:
        raise StagedError(f"The staged mode supports {', '.join(PROGRAM_METHODS)}, not {plan['program']}.")
    limit = cluster_cpu_limit(plan["partition"])
    if not 1 <= plan["cpus"] <= limit:
        raise StagedError(f"cpus must be between 1 and {limit} (the limit of one user in this partition).")
    for key in ("max_stereoisomers", "max_tautomers"):
        if not 1 <= int(plan[key]) <= 20:
            raise StagedError(f"{key} must be between 1 and 20.")
    strategy = plan.get("strategy", "all")
    if strategy not in STRATEGIES:
        raise StagedError(f"strategy must be one of {', '.join(STRATEGIES)}, not {strategy!r}.")
    if strategy == "tranche":
        try:
            reps = int(plan.get("representatives", DEFAULT_REPRESENTATIVES))
        except (TypeError, ValueError):
            reps = 0
        if not 1 <= reps <= MAX_REPRESENTATIVES:
            raise StagedError(f"representatives must be a whole number from 1 to {MAX_REPRESENTATIVES}.")
        if plan.get("primary_size") is not None:
            try:
                size = int(plan["primary_size"])
            except (TypeError, ValueError):
                size = 0
            if size < 1:
                raise StagedError("primary_size must be a whole number of molecules, at least 1.")
        if plan.get("ml_filter") is not None and not isinstance(plan["ml_filter"], bool):
            raise StagedError("ml_filter must be true or false.")
        cutoff = plan.get("ml_cutoff")
        if cutoff is not None and (isinstance(cutoff, bool) or not isinstance(cutoff, (int, float))
                                   or not 0.05 <= cutoff <= 0.95):
            raise StagedError("ml_cutoff must be a number from 0.05 to 0.95.")
        properties = plan.get("tranche_properties") or DEFAULT_TRANCHE_PROPERTIES
        if isinstance(properties, str) or len(set(properties)) != len(properties):
            raise StagedError("tranche_properties must be a list of different property names.")
        unknown = [p for p in properties if p not in TRANCHE_PROPERTY_NAMES]
        if unknown:
            raise StagedError(f"Unknown tranche properties: {', '.join(map(str, unknown))}. "
                              f"Choose from: {', '.join(TRANCHE_PROPERTY_NAMES)}.")


def start(plan, submit=True):
    """Stage 0: write the run folder (inputs, AFLP and AFVS working copies, configuration, state) and, with
    submit=True, queue the first stage. Returns the state."""
    check_plan(plan)
    run_dir = os.path.abspath(plan["run_dir"])
    for sub in ("input", "prep", "dock", "logs", "results"):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
    if os.path.exists(state_path(run_dir)):
        raise StagedError(f"{run_dir} already holds a staged run. Use a new folder, or `advance` to continue it.")

    # inputs: the receptor, prepared once; the protein file itself is kept for `redock`
    if os.path.isfile(plan["receptor"]):
        shutil.copyfile(plan["receptor"], os.path.join(run_dir, "input", "receptor.pdb"))
    vs_dock.prepare_receptor(os.path.join(run_dir, "input"), plan["receptor"],
                             add_hydrogens=bool(plan.get("add_hydrogens")))
    collections = os.path.join(run_dir, "prep", "collections")
    names_path = os.path.join(run_dir, "input", "names.tsv")
    n, todo, unreadable = write_collections(plan["ligands"], collections, names_path)
    if n == 0:
        raise StagedError("The library has no molecule that RDKit can read.")
    plan = {**plan, "run_dir": run_dir, "strategy": plan.get("strategy", "all")}
    estimate, prescreen_forms = None, None
    forms = forms_per_molecule(plan["max_stereoisomers"], plan["max_tautomers"])
    if plan["strategy"] == "tranche":
        plan["representatives"] = int(plan.get("representatives") or DEFAULT_REPRESENTATIVES)
        plan["tranche_properties"] = list(plan.get("tranche_properties") or DEFAULT_TRANCHE_PROPERTIES)
        plan["primary_size"] = min(n, int(plan.get("primary_size") or max(1, round(n * DEFAULT_PRIMARY_FRACTION))))
        plan["ml_filter"] = bool(plan.get("ml_filter"))
        if plan["ml_filter"]:
            plan["ml_cutoff"] = float(plan.get("ml_cutoff") or DEFAULT_ML_CUTOFF)
        estimate = estimate_prescreen(names_path, n, forms, plan["tranche_properties"], plan["representatives"])
        prescreen_forms = estimate["prescreen_forms"]
    sizes = plan_sizes(n, plan["program"], plan["exhaustiveness"], plan["max_stereoisomers"], plan["max_tautomers"],
                       plan["cpus"], plan["strategy"], plan.get("primary_size"), prescreen_forms,
                       plan.get("ml_filter", False))
    configure_aflp(run_dir, plan, sizes, todo)
    if plan["strategy"] == "all":
        configure_afvs(run_dir, plan, sizes)

    def part():
        return {"workunits": 0, "submitted": 0, "jobs": {}}

    state = {"version": 2, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "plan": plan, "sizes": sizes,
             "estimate": estimate, "stage": "prepare", "status": "running",
             "counts": {"molecules": n, "unreadable": len(unreadable), "collections": len(todo)},
             "prepare": part(), "prescreen": part(), "dock": part(),
             "unreadable": [list(u) for u in unreadable[:50]], "events": []}
    log_event(state, f"Run folder written: {n:,} molecules in {len(todo):,} collections.")
    if estimate:
        log_event(state, f"Tranche strategy: about {estimate['tranches']:,} tranches, a prescreen of about "
                         f"{estimate['prescreen_forms']:,} forms, then the best tranches up to {plan['primary_size']:,} "
                         "molecules.")
    save_state(run_dir, state)
    if submit:
        with Lock(run_dir):
            state = load_state(run_dir)
            begin_prepare(run_dir, state)
            save_state(run_dir, state)
    return state


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------


def begin_prepare(run_dir, state):
    tool = os.path.join(run_dir, "aflp")
    upstream(tool, "aflp_prepare_folders.py")
    state["prepare"]["workunits"] = count_workunits(upstream(tool, "aflp_prepare_workunits.py"))
    state["stage"] = "prepare"
    log_event(state, f"Library preparation (AFLP): {state['prepare']['workunits']} work unit(s).")
    submit_next_workunit(run_dir, state, "prepare")


def stage_dirs(stage):
    """(working copy folder, script prefix, label) of the stages that run an upstream tool."""
    return {"prepare": ("aflp", "aflp", "Preparation"), "prescreen": ("afvs_prescreen", "afvs", "Prescreen"),
            "dock": ("afvs", "afvs", "Docking")}[stage]


def submit_next_workunit(run_dir, state, stage):
    """Submit the next work unit of a stage, and the `advance` that follows it."""
    folder, prefix, label = stage_dirs(stage)
    tool = os.path.join(run_dir, folder)
    part = state[stage]
    k = part["submitted"] + 1
    upstream(tool, f"{prefix}_submit_jobs.py", str(k), str(k))
    job_id = submitted_job_id(tool, k)
    part["submitted"] = k
    part["jobs"][str(k)] = job_id
    final = k == part["workunits"] and stage == "dock"
    advance_id = submit_advance(run_dir, state, job_id, final=final)
    log_event(state, f"{label} work unit {k} of {part['workunits']} submitted (job {job_id}); next step is job {advance_id}.")


def check_workunit_job(state, stage):
    """Stop the run when the last submitted work unit's job did not complete."""
    part = state[stage]
    if not part["submitted"]:
        return
    job_id = part["jobs"][str(part["submitted"])]
    if job_unfinished(job_id):
        raise StagedError(f"Job {job_id} (work unit {part['submitted']}) is still running or queued. "
                          "Nothing to do yet.")
    failures = job_failures(job_id)
    if failures:
        what = ", ".join(f"{n} task(s) {s}" for s, n in failures.items())
        folder = stage_dirs(stage)[0]
        raise StagedError(f"Job {job_id} (work unit {part['submitted']}) did not complete: {what}. See the logs in "
                          f"{os.path.join(state['plan']['run_dir'], folder, 'workflow', 'workunits')}.")


def prepared_forms(run_dir, state):
    """After AFLP: check every collection came out, repack for AFVS, write AFVS's todo.all. Returns the forms count."""
    todo = open(os.path.join(run_dir, "aflp", "tools", "templates", "todo.all")).read().split("\n")
    jobs = []
    for line in todo:
        if not line.strip():
            continue
        key, _count = line.split()
        meta, tranche, coll = key.split("_")
        src = os.path.join(run_dir, "aflp", "workflow", "complete", "pdbqt", meta, tranche, f"{coll}.tar.gz")
        jobs.append((key, src, os.path.join(run_dir, "dock", "collections", meta + tranche, f"{coll}.tar.gz"),
                     f"{meta}{tranche}_{coll}", coll))
    missing = [j[0] for j in jobs if not os.path.isfile(j[1])]
    if missing:
        raise StagedError(f"{len(missing)} of {len(jobs)} collections have no prepared output (first: "
                          f"{', '.join(missing[:5])}). Logs: {os.path.join(run_dir, 'aflp', 'workflow', 'workunits')}")
    workers = max(1, min(4, os.cpu_count() or 1))
    with multiprocessing.Pool(workers) as pool:
        results = pool.map(_repack, jobs)
    todo_afvs = [f"{name} {count}" for name, count in results if count > 0]
    empty = [name for name, count in results if count == 0]
    if not todo_afvs:
        raise StagedError("AFLP prepared no molecule at all. See the logs in the aflp/workflow folder.")
    with open(os.path.join(run_dir, "afvs", "tools", "templates", "todo.all"), "w") as fh:
        fh.write("\n".join(todo_afvs) + "\n")
    forms = sum(int(line.split()[1]) for line in todo_afvs)
    if empty:
        log_event(state, f"{len(empty)} collection(s) came out of AFLP empty.")
    return forms


def _repack(job):
    """One collection: the AFLP tar with its folder renamed to the collection name (what AFVS expects), and the
    COMPND line that Open Babel 3 writes removed (QuickVina and Vina reject it). Returns (AFVS key, forms)."""
    key, src, dst, afvs_key, folder = job
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    count = 0
    with tarfile.open(src) as old, tarfile.open(dst, "w:gz") as new:
        for member in old.getmembers():
            if member.isdir():
                member.name = folder
                new.addfile(member)
            elif member.isfile():
                lines = old.extractfile(member).read().decode().splitlines(keepends=True)
                data = "".join(line for line in lines if not line.startswith("COMPND")).encode()
                member.size = len(data)
                member.name = f"{folder}/{os.path.basename(member.name)}"
                new.addfile(member, io.BytesIO(data))
                count += 1
    return afvs_key, count


def begin_dock(run_dir, state):
    forms = prepared_forms(run_dir, state)
    state["counts"]["forms"] = forms
    log_event(state, f"AFLP prepared {forms:,} forms for {state['counts']['molecules']:,} molecules "
                     f"({forms / state['counts']['molecules']:.2f} per molecule).")
    tool = os.path.join(run_dir, "afvs")
    upstream(tool, "afvs_prepare_folders.py")
    state["dock"]["workunits"] = count_workunits(upstream(tool, "afvs_prepare_workunits.py"))
    state["stage"] = "dock"
    log_event(state, f"Docking (AFVS): {state['dock']['workunits']} work unit(s).")
    submit_next_workunit(run_dir, state, "dock")


def aflp_sources(run_dir):
    """Every collection AFLP was asked to prepare, as (AFLP key, its output tar). Stops if one is missing."""
    todo = open(os.path.join(run_dir, "aflp", "tools", "templates", "todo.all")).read().split("\n")
    sources = []
    for line in todo:
        if not line.strip():
            continue
        key, _count = line.split()
        meta, tranche, coll = key.split("_")
        sources.append((key, os.path.join(run_dir, "aflp", "workflow", "complete", "pdbqt", meta, tranche,
                                          f"{coll}.tar.gz")))
    missing = [key for key, src in sources if not os.path.isfile(src)]
    if missing:
        raise StagedError(f"{len(missing)} of {len(sources)} collections have no prepared output (first: "
                          f"{', '.join(missing[:5])}). Logs: {os.path.join(run_dir, 'aflp', 'workflow', 'workunits')}")
    return sources


def _write_tar(path, folder, members):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tarfile.open(path, "w:gz") as tf:
        info = tarfile.TarInfo(folder)
        info.type = tarfile.DIRTYPE
        info.mode = 0o755
        tf.addfile(info)
        for name, data in members:
            data = data if isinstance(data, bytes) else data.encode()
            info = tarfile.TarInfo(f"{folder}/{name}")
            info.size = len(data)
            info.mode = 0o644
            tf.addfile(info, io.BytesIO(data))


def build_tranche_library(run_dir, state):
    """The prepared forms regrouped into AdaptiveFlow's tranche layout:

      dock/collections/<tranche>/<7 digits>.tar.gz       every form, in collections of at most 500 of one tranche
      prescreen/collections/SFL/<7 digits>.tar.gz        the representatives of every tranche, with a .listing
      prep/tranches.tsv, prep/ligand_tranches.tsv        what is where

    AFLP does not do this (it only names files after a tranche, and its tranche properties need ChemAxon)."""
    import zlib

    plan = state["plan"]
    properties = tuple(plan.get("tranche_properties") or DEFAULT_TRANCHE_PROPERTIES)
    reps = int(plan.get("representatives") or DEFAULT_REPRESENTATIVES)
    names = read_names(run_dir)
    dimensions = tranche_dimensions()
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    tranche_of_molecule = {}
    by_tranche = {}
    for _key, src in aflp_sources(run_dir):
        with tarfile.open(src) as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                form = os.path.basename(member.name).rsplit(".", 1)[0]
                ident = form.split("_")[0]
                if ident not in tranche_of_molecule:
                    mol = Chem.MolFromSmiles(names[ident][1])
                    tranche_of_molecule[ident] = tranche_of(mol, properties, dimensions) if mol is not None else None
                tranche = tranche_of_molecule[ident]
                if tranche is None:
                    continue
                lines = tf.extractfile(member).read().decode().splitlines(keepends=True)
                text = "".join(line for line in lines if not line.startswith("COMPND"))
                by_tranche.setdefault(tranche, []).append((form, zlib.compress(text.encode(), 1)))
    if not by_tranche:
        raise StagedError("AFLP prepared no molecule at all. See the logs in the aflp/workflow folder.")

    for sub in ("dock/collections", "prescreen/collections"):
        shutil.rmtree(os.path.join(run_dir, sub), ignore_errors=True)
    rows, sparse = [], []
    with open(os.path.join(run_dir, "prep", "ligand_tranches.tsv"), "w") as fh:
        fh.write("ligand\tid\ttranche\n")
        for tranche in sorted(by_tranche):
            items = sorted(by_tranche[tranche])
            for form, _data in items:
                fh.write(f"{form}\t{form.split('_')[0]}\t{tranche}\n")
            where = {}
            for number, start in enumerate(range(0, len(items), TRANCHE_COLLECTION_SIZE), 1):
                chunk = items[start:start + TRANCHE_COLLECTION_SIZE]
                num = f"{number:07d}"
                _write_tar(os.path.join(run_dir, "dock", "collections", tranche, f"{num}.tar.gz"), num,
                           [(f"{form}.pdbqt", zlib.decompress(data)) for form, data in chunk])
                rows.append((tranche, num, len(chunk)))
                for form, _data in chunk:
                    where[form] = f"{tranche}_{num}"
            # the representatives: a fixed pseudo-random pick (by hash of the name), not the first in input order
            ranked = sorted(items, key=lambda item: hashlib.md5(item[0].encode()).hexdigest())[:reps]
            for index, (form, data) in enumerate(ranked):
                sparse.append((where[form], form, index, data))
    with open(os.path.join(run_dir, "prep", "tranches.tsv"), "w") as fh:
        fh.write("tranche\tcollection\tligands\n")
        for tranche, num, count in rows:
            fh.write(f"{tranche}\t{num}\t{count}\n")
    todo = []
    for number, start in enumerate(range(0, len(sparse), SPARSE_COLLECTION_SIZE), 1):
        chunk = sparse[start:start + SPARSE_COLLECTION_SIZE]
        num = f"{number:07d}"
        members = [(f"{form}.pdbqt", zlib.decompress(data)) for _c, form, _i, data in chunk]
        members.append((".listing", "".join(f"{c},{form},{i}\n" for c, form, i, _d in chunk)))
        _write_tar(os.path.join(run_dir, "prescreen", "collections", "SFL", f"{num}.tar.gz"), num, members)
        todo.append(f"SFL_{num} {len(chunk)}")
    with open(os.path.join(run_dir, "prescreen", "todo.all"), "w") as fh:
        fh.write("\n".join(todo) + "\n")
    return {"forms": sum(len(v) for v in by_tranche.values()), "tranches": len(by_tranche),
            "collections": len(rows), "prescreen_forms": len(sparse)}


def begin_prescreen(run_dir, state):
    """After AFLP: build the tranche library, then queue the prescreen (AFVS on the representatives)."""
    plan = state["plan"]
    counts = build_tranche_library(run_dir, state)
    state["counts"].update(forms=counts["forms"], tranches=counts["tranches"], prescreen_forms=counts["prescreen_forms"])
    log_event(state, f"AFLP prepared {counts['forms']:,} forms for {state['counts']['molecules']:,} molecules; "
                     f"{counts['tranches']:,} tranches; {counts['prescreen_forms']:,} representatives for the prescreen.")
    sizes = stage_sizes(counts["prescreen_forms"], plan)
    work = configure_afvs(run_dir, plan, sizes, "afvs_prescreen", os.path.join(run_dir, "prescreen", "collections"),
                          representatives=int(plan.get("representatives") or DEFAULT_REPRESENTATIVES))
    shutil.copyfile(os.path.join(run_dir, "prescreen", "todo.all"), os.path.join(work, "tools", "templates", "todo.all"))
    upstream(work, "afvs_prepare_folders.py")
    state["prescreen"]["workunits"] = count_workunits(upstream(work, "afvs_prepare_workunits.py"))
    state["stage"] = "prescreen"
    log_event(state, f"Prescreen (AFVS): {state['prescreen']['workunits']} work unit(s).")
    submit_next_workunit(run_dir, state, "prescreen")


def select_tranches(run_dir, state):
    """Score every tranche by the mean docking score of its prescreened forms, take the best tranches until the primary
    screen has about `primary_size` molecules (the 'tranche_ave_score' selection of AdaptiveFlow), and write the todo
    file. Whole collections are taken, so the selection can be a little larger than asked for."""
    plan = state["plan"]
    scores = {}
    files = sorted(glob.glob(os.path.join(run_dir, "afvs_prescreen", "output-files", SCENARIO, "csv", "*", "*.csv.gz")))
    if not files:
        raise StagedError("The prescreen wrote no result tables (afvs_prescreen/output-files/%s/csv)." % SCENARIO)
    for path in files:
        with gzip.open(path, "rt") as fh:
            for row in csv.DictReader(fh):
                try:
                    score = float(row["score_min"])
                except (KeyError, ValueError):
                    continue
                scores.setdefault(row["collection_key"].split("_")[0], []).append(score)
    if not scores:
        raise StagedError("No prescreened ligand has a docking score; nothing to select tranches from.")
    collections = []
    with open(os.path.join(run_dir, "prep", "tranches.tsv")) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            collections.append((row["tranche"], row["collection"], int(row["ligands"])))
    mean = {t: sum(v) / len(v) for t, v in scores.items()}
    ranked = sorted((c for c in collections if c[0] in mean), key=lambda c: (mean[c[0]], c[0], c[1]))
    primary_forms = int(math.ceil(int(plan.get("primary_size") or state["counts"]["molecules"])
                                  * state["sizes"]["forms_per_molecule"]))
    chosen, total = [], 0
    for tranche, num, count in ranked:
        if total >= primary_forms:
            break
        chosen.append((tranche, num, count))
        total += count
    selected = {t for t, _n, _c in chosen}
    with open(os.path.join(run_dir, "dock", "todo.all"), "w") as fh:
        fh.write("\n".join(f"{t}_{n} {c}" for t, n, c in chosen) + "\n")
    in_library = {}
    for tranche, _num, count in collections:
        in_library[tranche] = in_library.get(tranche, 0) + count
    with open(os.path.join(run_dir, "results", "tranche_scores.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["tranche", "prescreened", "mean_score", "min_score", "forms_in_library", "selected"])
        for tranche in sorted(in_library, key=lambda t: (mean.get(t, 1e9), t)):
            v = scores.get(tranche, [])
            writer.writerow([tranche, len(v), f"{mean[tranche]:.3f}" if v else "", f"{min(v):.3f}" if v else "",
                             in_library[tranche], "yes" if tranche in selected else "no"])
    state["counts"].update(tranches_scored=len(mean), tranches_selected=len(selected), primary_forms=total)
    return total


def prescreen_training_data(run_dir):
    """(form SMILES, best score) of every prescreened form, from the prescreen's result tables."""
    best = {}
    files = sorted(glob.glob(os.path.join(run_dir, "afvs_prescreen", "output-files", SCENARIO, "csv", "*", "*.csv.gz")))
    for path in files:
        with gzip.open(path, "rt") as fh:
            for row in csv.DictReader(fh):
                try:
                    score = float(row["score_min"])
                except (KeyError, ValueError):
                    continue
                smiles = (row.get("attr_smi") or "").strip()
                if smiles and (row["ligand"] not in best or score < best[row["ligand"]][1]):
                    best[row["ligand"]] = (smiles, score)
    return [v[0] for v in best.values()], [v[1] for v in best.values()]


_ML = {}


def _ml_init(model_path):
    _ML["model"] = vs_mlfilter.Model.load(model_path)
    _ML["generator"] = vs_mlfilter._generator()


def _form_smiles(text):
    match = re.search(r"^REMARK\s+SMILES_current:\s*(\S+)", text, re.M)
    return match.group(1) if match else ""


def _filter_collection(job):
    """One collection of the selected tranches: score every form with the model and write a collection with the forms
    that reach the cutoff (a form whose SMILES cannot be read is kept: it cannot be judged). Returns
    (key, forms in, forms kept, [(form, probability, kept)])."""
    src, dst, folder, key, cutoff = job
    with tarfile.open(src) as old:
        items = [(os.path.basename(m.name), old.extractfile(m).read()) for m in old.getmembers() if m.isfile()]
    smiles = [_form_smiles(data.decode(errors="replace")) for _name, data in items]
    unique = sorted({smi for smi in smiles if smi})
    probability = dict(zip(unique, _ML["model"].predict(unique, _ML["generator"]))) if unique else {}
    kept, rows = [], []
    for (name, data), smi in zip(items, smiles):
        p = float(probability.get(smi, float("nan")))
        keep = not p < cutoff                                  # NaN (unreadable) is kept
        rows.append((name.rsplit(".", 1)[0], p, keep))
        if keep:
            kept.append((name, data))
    if kept:
        _write_tar(dst, folder, kept)
    return key, len(items), len(kept), rows


def _write_filter_outcomes(outcomes, writer, todo):
    """Write the table of probabilities and collect the todo lines; returns (forms in, forms kept)."""
    forms = kept = 0
    for key, n_in, n_kept, rows in outcomes:
        forms, kept = forms + n_in, kept + n_kept
        if n_kept:
            todo.append(f"{key} {n_kept}")
        for form, p, keep in rows:
            writer.writerow([form, key.split("_")[0], "" if p != p else f"{p:.4f}", "yes" if keep else "no"])
    return forms, kept


def apply_ml_filter(run_dir, state):
    """Train the neural-network filter on the prescreen and apply it to the forms of the selected collections. Writes
    dock_ml/ (collections with the forms kept, and todo.all), results/ml_filter.csv (probability of every form) and
    results/ml_filter.json (how good the model was). Returns {"forms": ..., "kept": ...}, or None when the filter was
    not used (too little data, or a model no better than chance), in which case the whole selection is docked."""
    plan, counts = state["plan"], state["counts"]
    cutoff = float(plan.get("ml_cutoff") or DEFAULT_ML_CUTOFF)
    report_path = os.path.join(run_dir, "results", "ml_filter.json")

    def skip(reason, report=None):
        counts["ml_skipped"] = reason
        with open(report_path, "w") as fh:
            json.dump({"used": False, "reason": reason, **(report or {})}, fh, indent=1)
        log_event(state, f"Neural-network filter not used: {reason} The whole selection is docked.")
        return None

    smiles, scores = prescreen_training_data(run_dir)
    if len(scores) < ML_MIN_TRAIN:
        return skip(f"the prescreen has {len(scores):,} scored forms; at least {ML_MIN_TRAIN} are needed to train it.")
    try:
        model = vs_mlfilter.train(smiles, scores, cutoff)
    except ValueError as exc:
        return skip(f"it could not be trained ({exc}).")
    report = model.report
    log_event(state, f"Neural-network filter trained on {report['trained_on']:,} of {report['forms']:,} prescreened forms "
                     f"(area under the curve {report['auc']:.2f} on {report['held_back']:,} forms held back).")
    if report["auc"] is None or report["auc"] < ML_MIN_AUC:
        return skip(f"it is hardly better than chance on forms it has not seen (area under the curve "
                    f"{report['auc'] if report['auc'] is None else round(report['auc'], 2)}, at least {ML_MIN_AUC} "
                    "is needed).", report)
    os.makedirs(os.path.join(run_dir, "prep"), exist_ok=True)
    model_path = os.path.join(run_dir, "prep", "ml_model.npz")
    model.save(model_path)

    shutil.rmtree(os.path.join(run_dir, "dock_ml"), ignore_errors=True)
    jobs, todo = [], []
    with open(os.path.join(run_dir, "dock", "todo.all")) as fh:
        for line in fh:
            if line.strip():
                key = line.split()[0]
                tranche, num = key.split("_")
                jobs.append((os.path.join(run_dir, "dock", "collections", tranche, f"{num}.tar.gz"),
                             os.path.join(run_dir, "dock_ml", "collections", tranche, f"{num}.tar.gz"), num, key, cutoff))
    workers = max(1, min(4, os.cpu_count() or 1))
    forms = kept = 0
    # Fresh processes, not forks: numpy's BLAS threads are already running in this one, and forking such a process can
    # deadlock the children.
    pool = multiprocessing.get_context("spawn").Pool(workers, _ml_init, (model_path,)) if workers > 1 else None
    if pool is None:
        _ml_init(model_path)
    try:
        with open(os.path.join(run_dir, "results", "ml_filter.csv"), "w", newline="") as table:
            writer = csv.writer(table)
            writer.writerow(["ligand", "tranche", "probability", "kept"])
            outcomes = pool.imap(_filter_collection, jobs) if pool else map(_filter_collection, jobs)
            forms, kept = _write_filter_outcomes(outcomes, writer, todo)
    finally:
        if pool:
            pool.close()
            pool.join()
    if not kept:
        raise StagedError("The neural-network filter removed every form. Lower ml_cutoff, or turn the filter off.")
    with open(os.path.join(run_dir, "dock_ml", "todo.all"), "w") as fh:
        fh.write("\n".join(todo) + "\n")
    report.update(used=True, pool_forms=forms, kept_forms=kept)
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=1)
    counts.update(ml_pool_forms=forms, ml_kept_forms=kept, ml_auc=report["auc"])
    counts.pop("ml_skipped", None)
    return {"forms": forms, "kept": kept}


def begin_primary(run_dir, state):
    """After the prescreen: pick the tranches, filter them with the neural network if asked, then queue the primary screen
    on what is left."""
    plan = state["plan"]
    total = select_tranches(run_dir, state)
    log_event(state, f"Selected {state['counts']['tranches_selected']:,} of {state['counts']['tranches']:,} tranches "
                     f"({total:,} forms, {100 * total / state['counts']['forms']:.0f}% of the library).")
    collections, todo = os.path.join(run_dir, "dock", "collections"), os.path.join(run_dir, "dock", "todo.all")
    if plan.get("ml_filter"):
        result = apply_ml_filter(run_dir, state)
        if result:
            total = result["kept"]
            collections, todo = os.path.join(run_dir, "dock_ml", "collections"), os.path.join(run_dir, "dock_ml", "todo.all")
            log_event(state, f"The filter keeps {total:,} of {result['forms']:,} forms ({100 * total / result['forms']:.0f}%) "
                             "for the primary screen.")
    sizes = stage_sizes(total, plan)
    work = configure_afvs(run_dir, plan, sizes, "afvs", collections)
    shutil.copyfile(todo, os.path.join(work, "tools", "templates", "todo.all"))
    upstream(work, "afvs_prepare_folders.py")
    state["dock"]["workunits"] = count_workunits(upstream(work, "afvs_prepare_workunits.py"))
    state["stage"] = "dock"
    log_event(state, f"Primary screen (AFVS): {state['dock']['workunits']} work unit(s).")
    submit_next_workunit(run_dir, state, "dock")


def read_names(run_dir):
    names = {}
    with open(os.path.join(run_dir, "input", "names.tsv")) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            names[row["id"]] = (row["name"], row["smiles"])
    return names


FORM_SUFFIX = re.compile(r"_S\d+_T\d+$")
RESULT_COLUMNS = ["index", "name", "status", "best_score", "n_forms", "best_form", "note", "smiles"]


TRANCHE_RESULT_COLUMNS = RESULT_COLUMNS + ["tranche"]


def read_ligand_tranches(run_dir):
    """{molecule id: tranche} of every molecule with a prepared form (written when the tranche library was built)."""
    tranches = {}
    with open(os.path.join(run_dir, "prep", "ligand_tranches.tsv")) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            tranches[row["id"]] = row["tranche"]
    return tranches


def rank(run_dir, state):
    """Read the AFVS summaries (the prescreen too, with the tranche strategy), keep the best form of every molecule,
    write results/results.csv and ranking.csv. Returns (docked, without a score)."""
    tranche_mode = state.get("plan", {}).get("strategy") == "tranche"
    best, forms = {}, {}
    folders = ["afvs"] + (["afvs_prescreen"] if tranche_mode else [])
    files = []
    for folder in folders:
        files += sorted(glob.glob(os.path.join(run_dir, folder, "output-files", SCENARIO, "csv", "*", "*.csv.gz")))
    if not files:
        raise StagedError("AFVS wrote no result tables (output-files/%s/csv)." % SCENARIO)
    for path in files:
        with gzip.open(path, "rt") as fh:
            for row in csv.DictReader(fh):
                ident = FORM_SUFFIX.sub("", row["ligand"])
                forms.setdefault(ident, set()).add(row["ligand"])
                try:
                    score = float(row["score_min"])
                except (KeyError, ValueError):
                    continue
                if ident not in best or score < best[ident][0]:
                    best[ident] = (score, row["ligand"], row.get("attr_smi", ""))
    names = read_names(run_dir)
    prepared = read_ligand_tranches(run_dir) if tranche_mode else {}
    rows = []
    for index, (ident, (name, smiles)) in enumerate(names.items()):
        row = {"index": index, "name": name, "status": "", "best_score": "", "n_forms": len(forms.get(ident, ())),
               "best_form": "", "note": "", "smiles": smiles, "tranche": prepared.get(ident, "")}
        if ident in best:
            score, form, form_smiles = best[ident]
            row.update(status="docked", best_score=f"{score:.3f}", best_form=form, smiles=form_smiles or smiles)
        elif tranche_mode and ident in prepared:
            row.update(status="not_docked", note="not selected for the primary screen, or removed by the filter")
        else:
            row.update(status="failed", note="no prepared form was docked")
        rows.append(row)
    out = os.path.join(run_dir, "results")
    os.makedirs(out, exist_ok=True)
    columns = TRANCHE_RESULT_COLUMNS if tranche_mode else RESULT_COLUMNS
    for filename, ordered in (("results.csv", rows),
                              ("ranking.csv", sorted(rows, key=lambda r: (r["status"] != "docked",
                                                                           float(r["best_score"] or 0))))):
        with open(os.path.join(out, filename), "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(ordered)
    docked = sum(1 for r in rows if r["status"] == "docked")
    not_docked = sum(1 for r in rows if r["status"] == "not_docked")
    state["counts"].update(docked=docked, failed=len(rows) - docked - not_docked)
    if tranche_mode:
        state["counts"]["not_docked"] = not_docked
    return docked, len(rows) - docked


def finish(run_dir, state):
    docked, without = rank(run_dir, state)
    state["stage"], state["status"] = "done", "done"
    c = state["counts"]
    if state["plan"].get("strategy") == "tranche":
        text = (f"Finished: {docked:,} of {c['molecules']:,} molecules docked ({c.get('not_docked', 0):,} were "
                f"not selected or were removed by the filter, {c['failed']:,} have no prepared form).")
    else:
        text = f"Finished: {docked:,} molecules docked, {without:,} without a score."
    log_event(state, f"{text} Results: {os.path.join(run_dir, 'results')}")


def advance(run_dir):
    """The next step of a run: check what the last stage produced and submit the next one."""
    run_dir = os.path.abspath(run_dir)
    with Lock(run_dir):
        state = load_state(run_dir)
        if state["status"] in ("done", "failed"):
            print(f"This run is {state['status']}." + (f" {state.get('reason', '')}" if state["status"] == "failed" else ""))
            return state
        try:
            stage = state["stage"]
            if stage not in ("prepare", "prescreen", "dock"):
                raise StagedError(f"Unknown stage '{stage}'.")
            tranche = state["plan"].get("strategy") == "tranche"
            part = state.setdefault(stage, {"workunits": 0, "submitted": 0, "jobs": {}})
            if part["workunits"] == 0:                       # the stage was never set up
                {"prepare": begin_prepare, "prescreen": begin_prescreen,
                 "dock": begin_primary if tranche else begin_dock}[stage](run_dir, state)
            else:
                check_workunit_job(state, stage)             # does nothing before the first work unit is submitted
                if part["submitted"] < part["workunits"]:
                    submit_next_workunit(run_dir, state, stage)
                elif stage == "prepare":
                    (begin_prescreen if tranche else begin_dock)(run_dir, state)
                elif stage == "prescreen":
                    begin_primary(run_dir, state)
                else:
                    finish(run_dir, state)
        except StagedError as exc:
            message = str(exc)
            if "still running or queued" in message:
                print(message)
                return state
            state["status"], state["reason"] = "failed", message
            log_event(state, f"STOPPED: {message}")
            save_state(run_dir, state)
            raise
        except Exception as exc:                                  # an unexpected error: keep the state, say so
            state["status"], state["reason"] = "failed", f"{type(exc).__name__}: {exc}"
            log_event(state, f"STOPPED: {state['reason']}")
            save_state(run_dir, state)
            raise
        save_state(run_dir, state)
        return state


def retry(run_dir):
    """Clear a stopped run's failed mark so that `advance` tries the step again (after the cause was fixed)."""
    run_dir = os.path.abspath(run_dir)
    with Lock(run_dir):
        state = load_state(run_dir)
        if state["status"] != "failed":
            raise StagedError(f"The run is {state['status']}, not failed.")
        state["status"] = "running"
        state.pop("reason", None)
        # a work unit whose job failed is submitted again; a step after the last job (building the library, selecting
        # tranches, ranking) is simply run again
        part = state[state["stage"]]
        if part["submitted"] and job_failures(part["jobs"][str(part["submitted"])]):
            part["submitted"] -= 1
        log_event(state, "Retrying the last step.")
        save_state(run_dir, state)


# --------------------------------------------------------------------------
# Status and results
# --------------------------------------------------------------------------


def status_text(run_dir):
    state = load_state(os.path.abspath(run_dir))
    c = state["counts"]
    sizes = state["sizes"]
    tranche = state["plan"].get("strategy") == "tranche"
    lines = [f"Run folder   {state['plan']['run_dir']}",
             f"Status       {state['status']}   (stage: {state['stage']})",
             f"Molecules    {c['molecules']:,} in {c['collections']:,} collections"
             + (f", {c['unreadable']:,} could not be read" if c.get("unreadable") else "")]
    if tranche:
        lines.append(f"Strategy     tranches: {state['plan']['representatives']} representative(s) per tranche, then the "
                     f"best tranches up to {state['plan']['primary_size']:,} molecules")
    if c.get("forms"):
        lines.append(f"Forms        {c['forms']:,} prepared ({c['forms'] / c['molecules']:.2f} per molecule)")
    if tranche and c.get("tranches"):
        lines.append(f"Tranches     {c['tranches']:,}; {c.get('prescreen_forms', 0):,} forms in the prescreen"
                     + (f"; {c['tranches_selected']:,} selected ({c['primary_forms']:,} forms)"
                        if c.get("tranches_selected") else ""))
    if tranche and c.get("ml_kept_forms"):
        lines.append(f"ML filter    keeps {c['ml_kept_forms']:,} of {c['ml_pool_forms']:,} forms of the selected tranches "
                     f"(area under the curve {c['ml_auc']:.2f})")
    elif tranche and c.get("ml_skipped"):
        lines.append(f"ML filter    not used: {c['ml_skipped']}")
    for stage, label in (("prepare", "Preparation"), ("prescreen", "Prescreen"), ("dock", "Docking")):
        part = state.get(stage) or {}
        if part.get("workunits"):
            lines.append(f"{label:<12} work unit {part['submitted']} of {part['workunits']} submitted")
    if state["status"] == "running":
        if state.get("advance_job"):
            lines.append(f"Next step    job {state['advance_job']} ({' '.join(job_states(state['advance_job'])[:1]) or 'unknown'})")
    if state["status"] == "failed":
        lines.append(f"Reason       {state.get('reason', '')}")
        lines.append("             Fix the cause, then run `status`/`advance --retry` to try the step again.")
    if state["status"] == "done":
        extra = f", {c['not_docked']:,} not docked (not selected, or removed by the filter)" if tranche else ""
        lines.append(f"Docked       {c.get('docked', 0):,}{extra}; without a score {c.get('failed', 0):,}")
        lines.append(f"Results      {os.path.join(state['plan']['run_dir'], 'results')}")
    lines.append(f"Plan         about {sizes['cpu_hours']:,.0f} CPU-hours, {sizes['forms_per_molecule']:.2f} forms per molecule")
    if state["events"]:
        lines.append("Last event   " + state["events"][-1]["time"] + "  " + state["events"][-1]["text"])
    return "\n".join(lines)


def results_text(run_dir, top=20):
    path = os.path.join(os.path.abspath(run_dir), "results", "ranking.csv")
    if not os.path.isfile(path):
        raise StagedError("There is no ranking yet; the run has not finished. See `status`.")
    lines = [f"{'rank':>5}  {'score':>7}  {'forms':>5}  name"]
    with open(path) as fh:
        for rank_number, row in enumerate(csv.DictReader(fh), 1):
            if rank_number > top or row["status"] != "docked":
                break
            lines.append(f"{rank_number:>5}  {row['best_score']:>7}  {row['n_forms']:>5}  {row['name']}")
    return "\n".join(lines) + f"\n\nFull table: {path}"


# --------------------------------------------------------------------------
# Poses for the best molecules: a full-mode run on the hits
# --------------------------------------------------------------------------


REDOCK_INPUT = "redock_input"   # the molecule file and run file written for the full-mode run
REDOCK_OUTPUT = "redock"        # the run folder of the full-mode run: poses and results
DEFAULT_REDOCK_TOP = 1000
DEFAULT_REDOCK_CPUS = 32
MAX_REDOCK_CPUS = 256           # what a full-mode run allows


def finished_ranking(run_dir):
    """The ranking of a finished staged run: (state, rows of results/ranking.csv, best first)."""
    state = load_state(run_dir)
    path = os.path.join(run_dir, "results", "ranking.csv")
    if state["status"] != "done" or not os.path.isfile(path):
        raise StagedError("The staged run has not finished, so there is no ranking to take molecules from. See `status`.")
    with open(path) as fh:
        return state, list(csv.DictReader(fh))


def receptor_has_hydrogens(path):
    with open(path, errors="replace") as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")):
                element = line[76:78].strip() or line[12:16].strip().lstrip("0123456789")[:1]
                if element.upper() == "H":
                    return True
    return False


def redock_files(run_dir, top=DEFAULT_REDOCK_TOP, cpus=DEFAULT_REDOCK_CPUS, exhaustiveness=None, num_modes=None,
                 mode="submit"):
    """Write what a full-mode run needs to dock the best molecules of a finished staged run again and keep the poses:
    redock_input/hits.smi (the best form of every molecule, as AFLP prepared it), hits.tsv (what each came from) and
    run.yaml. The staged screen keeps scores only; this gives the poses. Returns (run file path, number of molecules)."""
    import vs_dock
    import vs_runfile

    run_dir = os.path.abspath(run_dir)
    if top < 1:
        raise StagedError("--top must be at least 1.")
    if not 1 <= cpus <= MAX_REDOCK_CPUS:
        raise StagedError(f"--cpus must be between 1 and {MAX_REDOCK_CPUS} for a full-mode run.")
    state, rows = finished_ranking(run_dir)
    plan = state["plan"]
    receptor = os.path.join(run_dir, "input", "receptor.pdb")
    if not os.path.isfile(receptor):
        receptor = plan.get("receptor") or ""
    if not os.path.isfile(receptor):
        raise StagedError("The protein file of this run is no longer available (runs started before this feature did "
                          "not keep a copy). Dock the molecules with a run file of your own instead.")
    if os.path.exists(os.path.join(run_dir, REDOCK_OUTPUT)) or os.path.exists(os.path.join(run_dir, REDOCK_INPUT)):
        raise StagedError(f"This run already has a `{REDOCK_OUTPUT}` folder. Move it away to dock again.")
    hits = [r for r in rows if r["status"] == "docked"][:top]
    if not hits:
        raise StagedError("No molecule of this run has a score, so there is nothing to dock again.")
    folder = os.path.join(run_dir, REDOCK_INPUT)
    os.makedirs(folder)
    with open(os.path.join(folder, "hits.smi"), "w") as smi, open(os.path.join(folder, "hits.tsv"), "w", newline="") as tsv:
        writer = csv.writer(tsv, delimiter="\t")
        writer.writerow(["index", "rank", "name", "staged_score", "form", "smiles"])
        for index, row in enumerate(hits):
            name = re.sub(r"[\s,;]+", "_", row["name"])
            smi.write(f"{row['smiles']} {name}\n")
            writer.writerow([index, index + 1, row["name"], row["best_score"], row["best_form"], row["smiles"]])
    docking = {"program": plan["program"], "exhaustiveness": exhaustiveness or plan["exhaustiveness"],
               "num_modes": num_modes or vs_dock.DEFAULT_NUM_MODES, "seed": plan["seed"], "box_size": plan["box_size"],
               "ionisation": "keep"}          # the forms are already protonated at the pH of the staged run
    values = {"receptor": receptor, "box_center": [round(c, 3) for c in plan["center"]],
              "ligands": os.path.join(folder, "hits.smi"), "output": os.path.join(run_dir, REDOCK_OUTPUT),
              "flagged_ligands": "include", "docking": docking,
              "run": {"mode": mode, "job_name": f"{plan.get('job_name', 'vs')}_redock"[:40], "cpus": min(cpus, len(hits)),
                      "partition": plan["partition"]}}
    if not receptor_has_hydrogens(receptor):
        values["protonation"] = "adfr" if plan.get("add_hydrogens") else "none"
    path = os.path.join(folder, "run.yaml")
    with open(path, "w") as fh:
        fh.write(vs_runfile.render(values, "vs"))
    return path, len(hits)


def redock_report(run_dir):
    """Join the poses run's scores with the staged scores: results/redock_comparison.csv. Returns a text summary."""
    run_dir = os.path.abspath(run_dir)
    hits_path = os.path.join(run_dir, REDOCK_INPUT, "hits.tsv")
    results_path = os.path.join(run_dir, REDOCK_OUTPUT, "results", "results.csv")
    if not os.path.isfile(hits_path):
        raise StagedError("This run has not been docked again. Start it with `redock <run folder> --top N`.")
    if not os.path.isfile(results_path):
        raise StagedError("The docking of the best molecules has not written results yet. Follow its job with "
                          f"`squeue`; its folder is {os.path.join(run_dir, REDOCK_OUTPUT)}.")
    with open(hits_path) as fh:
        hits = list(csv.DictReader(fh, delimiter="\t"))
    with open(results_path) as fh:
        results = {row["index"]: row for row in csv.DictReader(fh)}
    columns = ["rank", "name", "staged_score", "redock_score", "difference", "redock_status", "n_poses"]
    pairs, rows = [], []
    for hit in hits:
        result = results.get(hit["index"])
        status = result["status"] if result else "not_yet"
        score = result["best_score"] if result and status == "docked" and result["best_score"] else ""
        diff = ""
        if score != "":
            diff = f"{float(score) - float(hit['staged_score']):.3f}"
            pairs.append((float(hit["staged_score"]), float(score)))
        rows.append({"rank": hit["rank"], "name": hit["name"], "staged_score": hit["staged_score"],
                     "redock_score": score, "difference": diff, "redock_status": status,
                     "n_poses": result["n_poses"] if result else ""})
    out = os.path.join(run_dir, "results", "redock_comparison.csv")
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    lines = [f"{len(hits)} molecules were docked again; {len(pairs)} have a score "
             f"({sum(1 for r in rows if r['redock_status'] == 'not_yet')} not finished, "
             f"{sum(1 for r in rows if r['redock_status'] not in ('docked', 'not_yet'))} failed)."]
    if len(pairs) >= 2:
        diffs = [b - a for a, b in pairs]
        lines.append(f"Score now minus staged score: mean {sum(diffs) / len(diffs):+.2f} kcal/mol, "
                     f"{sum(1 for d in diffs if abs(d) <= 0.5)} of {len(diffs)} within 0.5.")
        a, b = [p[0] for p in pairs], [p[1] for p in pairs]
        if len(set(a)) > 1 and len(set(b)) > 1:
            import statistics
            lines.append(f"Correlation of the two scores: {statistics.correlation(a, b):.2f}.")
    lines += [f"Table: {out}",
              f"Poses: {os.path.join(run_dir, REDOCK_OUTPUT, 'results')} (top_poses.sdf has the best pose of every molecule)."]
    return "\n".join(lines)


def launch_redock(path, check_only=False):
    """Start the full-mode run described by the run file (the assistant's own run-file path)."""
    import vs_wizard

    vs_wizard.check_environment()
    vs_wizard.run_from_file(path, check_only)


def redock_command(args):
    if args.report:
        print(redock_report(args.run_dir))
        return
    run_dir = os.path.abspath(args.run_dir)
    path, n = redock_files(run_dir, args.top, args.cpus, args.exhaustiveness, args.num_modes,
                           "save" if args.save else "submit")
    print(f"{n:,} molecules (the best form of each) will be docked again with poses kept. Run file: {path}")
    try:
        launch_redock(path)
    except BaseException:                      # nothing started: leave the folder as it was, so it can be tried again
        output = os.path.join(run_dir, REDOCK_OUTPUT)
        if os.path.isdir(output) and not os.listdir(output):
            os.rmdir(output)
        if not os.path.exists(output):
            shutil.rmtree(os.path.join(run_dir, REDOCK_INPUT), ignore_errors=True)
        raise
    print(f"\nWhen it has finished:  vs redock {run_dir} --report")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="vs_staged.py")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("advance", "status", "results", "redock"):
        p = sub.add_parser(name)
        p.add_argument("run_dir")
        if name == "redock":
            p.add_argument("--top", type=int, default=DEFAULT_REDOCK_TOP,
                           help=f"how many of the best molecules to dock again (default {DEFAULT_REDOCK_TOP})")
            p.add_argument("--cpus", type=int, default=DEFAULT_REDOCK_CPUS, help="CPUs of the docking job")
            p.add_argument("--exhaustiveness", type=int, default=None, help="default: the one of the staged run")
            p.add_argument("--num-modes", type=int, default=None, help="poses kept per molecule (default 9)")
            p.add_argument("--save", action="store_true", help="only prepare the job; submit it yourself")
            p.add_argument("--report", action="store_true",
                           help="when it has finished: compare the new scores with the staged ones")
        if name == "advance":
            p.add_argument("--retry", action="store_true", help="try again a step that stopped")
        if name == "results":
            p.add_argument("--top", type=int, default=20)
    args = parser.parse_args(argv)
    try:
        if args.command == "advance":
            if args.retry:
                retry(args.run_dir)
            advance(args.run_dir)
        elif args.command == "redock":
            redock_command(args)
        elif args.command == "status":
            print(status_text(args.run_dir))
        else:
            print(results_text(args.run_dir, args.top))
    except StagedError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
