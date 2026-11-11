"""The interactive assistant for virtual screening, and the same steps driven by a run file.

Every step is written once. With cfg=None it asks the user; with cfg from a run file (vs_runfile.load) it takes the
answer from the file, and stops with an explanation (Stop) when the file leaves open a question that the data raises.
So `vs` and `vs run.yaml` prepare, check and run in exactly the same way.

The structure-cleaning and protonation steps follow dock_cli.py of the anchored_docking_workflow repository
(MIT License, see NOTICE.md).
"""

import atexit
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

import vs_dock
import vs_runfile
import vs_slurm
from vs_ui import (WARNINGS, ask, ask_menu, ask_number, ask_yes_no, bold, dim, fail, file_validator, fmt_minutes,
                   heading, note, ok, path_completer, red, say, table, time_to_minutes, time_validator, warn, yellow)

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
UNDER_DDSUITE = bool(os.environ.get("DDSUITE_RUN_COMMAND"))
RUN_COMMAND = os.environ.get("DDSUITE_RUN_COMMAND") or "pixi run python vs_cli.py"

TOTAL_STEPS = 5
WATER_NAMES = {"HOH", "WAT", "H2O", "DOD", "TIP3", "SOL"}
MIN_BOX = 8.0
LARGE_BOX = 40.0                 # above this, Vina-family programs need more search effort to stay reliable
MAX_BOX = 126.0
BOX_MARGIN = 10.0                # box edge = size of the reference ligand + this
BOX_FAR_FROM_PROTEIN = 8.0       # a box centre this far from every protein atom is suspicious
BIG_LIBRARY = 20000              # above this many molecules, warn about the number of files and the run time
MAX_LIGAND_HEAVY = 80
DEFAULT_CPUS = vs_slurm.MAX_CPUS_DEFAULT
RUN_HERE_CPUS = 4                # a terminal on a shared machine is for small tests
PROGRAM_ORDER = ["qvina", "AutodockVina_1.2", "vina", "qvina-w", "smina", "PSOVina"]


class Stop(Exception):
    """The run file leaves a question open that the assistant would have asked, or asks for something impossible."""


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------


@dataclass
class Settings:
    # protein
    structure_path: str = ""
    receptor: str = ""               # the protein PDB the job will use (a temporary file until the files are saved)
    original_receptor: str = ""      # the user's own receptor file, before protonation
    prepared: object = None          # structure_prep.Prepared when the receptor comes from a complex
    protonated: object = None        # protein_prep.ProteinPrep when the assistant protonated the receptor
    add_hydrogens: bool = False
    receptor_hydrogens: int = -1     # hydrogens the receptor had before any preparation
    kept_labels: list = field(default_factory=list)
    template_label: str = ""
    # binding site
    reference_ligand: str = ""
    box_center_given: object = None
    center: list = field(default_factory=list)
    extent: float = 0.0
    box_size: float = 20.0
    receptor_coords: object = None
    route: str = ""                  # "structure" (a complex to clean up) or "files" (a ready receptor)
    ref_xyz: object = None           # heavy-atom coordinates of the ligand that marks the binding site
    info: object = None              # analyse_receptor() result for the receptor
    # molecules
    ligands: str = ""
    ligand_stage: str = ""          # the reduced molecule file, when some molecules were left out
    library: list = field(default_factory=list)       # LigandResult
    kept: list = field(default_factory=list)
    flagged_choice: object = None
    # docking
    program: str = vs_dock.DEFAULT_PROGRAM
    exhaustiveness: int = vs_dock.DEFAULT_EXHAUSTIVENESS
    num_modes: int = vs_dock.DEFAULT_NUM_MODES
    seed: int = vs_dock.DEFAULT_SEED
    rescore: object = None
    clean: bool = False
    # how to run
    mode: str = "submit"
    rundir: str = ""
    outdir: str = ""
    jobname: str = "vs_run"
    partition: str = vs_slurm.DEFAULT_PARTITION
    cpus: int = DEFAULT_CPUS
    time_limit: str = "2:00:00"
    resume: bool = False
    force: bool = False
    remaining: int = 0
    script_path: str = ""


@dataclass
class LigandResult:
    index: int
    name: str
    mol: object
    usable: bool
    problems: list = field(default_factory=list)


# --------------------------------------------------------------------------
# The protein
# --------------------------------------------------------------------------


def analyse_receptor(path):
    info = {"atoms": 0, "hydrogens": 0, "waters": 0, "hetero": 0, "hetnames": set(),
            "models": 0, "altlocs": 0, "chains": set(), "coords": []}
    with open(path, errors="replace") as fh:
        for line in fh:
            record = line[:6]
            if record.startswith("MODEL"):
                info["models"] += 1
                continue
            if record not in ("ATOM  ", "HETATM"):
                continue
            resname = line[17:20].strip()
            name = line[12:16].strip()
            element = line[76:78].strip().upper() or re.sub(r"[^A-Za-z]", "", name)[:1].upper()
            if record == "HETATM" and resname in WATER_NAMES:
                info["waters"] += 1
                continue
            if line[16:17] not in (" ", "A", ""):
                info["altlocs"] += 1
                continue  # count each alternate location only once
            if record == "HETATM":
                info["hetero"] += 1
                info["hetnames"].add(resname)
            else:
                info["atoms"] += 1
                info["chains"].add(line[21:22])
            if element in ("H", "D"):
                info["hydrogens"] += 1
                continue
            try:
                info["coords"].append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
            except ValueError:
                pass
    return info


def report_receptor(info, check_hetero=True):
    """Print findings. Returns False if the file is unusable."""
    if info["atoms"] == 0:
        fail("No protein atoms (ATOM records) were found. Is this a PDB file?")
        return False
    chains = ", ".join(sorted(c.strip() or "-" for c in info["chains"]))
    ok(f"Protein with {info['atoms'] - info['hydrogens']} heavy atoms (chains: {chains}).")
    if info["hydrogens"] == 0:
        warn("The receptor has no hydrogen atoms.", record=False)
    else:
        ok(f"Hydrogens present ({info['hydrogens']}).")
    if info["hetero"] and check_hetero:
        names = ", ".join(sorted(info["hetnames"]))
        warn(f"The file also contains non-protein molecules ({names}). Waters are removed automatically, but "
             "leftovers such as the original ligand occupy the binding site and block docking. Remove them unless "
             "you need them.")
    if info["models"] > 1:
        warn(f"The file contains {info['models']} models (e.g. an NMR ensemble). Keep just one.")
    if info["altlocs"]:
        warn(f"{info['altlocs']} atoms have alternate locations. Keep one conformation.")
    return True


def stage_text(text, name, prefix="vs_"):
    """Write text to a temporary file (removed when the assistant exits); the job gets its own saved copy later."""
    folder = tempfile.mkdtemp(prefix=prefix)
    atexit.register(shutil.rmtree, folder, ignore_errors=True)
    path = os.path.join(folder, name)
    with open(path, "w") as fh:
        fh.write(text)
    return path


# ---- choosing the ligand of a complex (interactive) ------------------------------------------------------------


def show_inventory(inv):
    chains = ", ".join(f"{name} ({n} residues)" for name, n in inv.protein_chains.items())
    ok(f"Protein chain(s): {chains}.")
    if inv.nucleic_chains:
        note(f"Nucleic acid chain(s) {', '.join(inv.nucleic_chains)} will not be used.")
    others = [h for h in inv.hets if h.kind != "sugar"]
    kinds = {}
    for h in others:
        kinds.setdefault(h.kind, []).append(h.name)
    parts = [f"{len(names)} {kind}{'s' if len(names) > 1 and kind != 'additive' else ''} ({', '.join(sorted(set(names)))})"
             for kind, names in kinds.items()]
    sugars = sum(1 for h in inv.hets if h.kind == "sugar")
    note(f"Also in the file: {inv.waters} water(s)" + (f", {', '.join(parts)}" if parts else "")
         + (f", {sugars} sugar residue(s)" if sugars else "") + ".")


def molecule_rows(shown, components):
    """Table rows for the molecules offered as the reference ligand: what each one is, in terms a chemist recognises.
    Names and formulas come from the PDB's chemical component dictionary; offline, the formula of the heavy atoms in
    the file is shown instead, marked as having no hydrogens."""
    rows = []
    for n, het in enumerate(shown, 1):
        info = components.describe(het.name)
        formula = info.formula or (f"{het.heavy_formula} (no H)" if het.heavy_formula else "")
        rows.append((n, info.name or "(name unavailable)", formula, het.name, het.chain.strip() or "-", het.number,
                     het.kind, het.heavy, het.contacts))
    return rows


def choose_ligand_of_complex(inv, components):
    """Ask which molecule marks the binding site. Returns a structure_prep.Het, or None to choose another file."""
    import structure_prep as sp

    candidates = sp.template_candidates(inv)
    if not candidates:
        fail("No molecule in this file is big enough to be a ligand, so the binding site cannot be located from it.")
        return None
    suggestion = sp.suggested_template(inv)
    say("\n  Which molecule sits in the binding site? The search box will be centred on it, and it is removed from")
    say("  the protein so that it does not block the site.")
    shown = candidates[:12]
    found = components.prefetch([h.name for h in shown])
    table(molecule_rows(shown, components),
          headers=("#", "Molecule", "Formula", "Code", "Chain", "Residue No.", "Kind", "Heavy atoms",
                   "Protein contacts"))
    if shown and not found:
        note("The names could not be looked up at the PDB (no connection?), so only the formulas of the atoms in the "
             "file are shown. Hydrogens are not in the file, so the real formulas have more H.")
    if len(candidates) > len(shown):
        note(f"{len(candidates) - len(shown)} smaller or less likely molecules are not shown.")
    default = str(shown.index(suggestion) + 1) if suggestion in shown else None
    if default is None:
        warn("None of the molecules looks like a drug-like ligand; pick the one you want, if any.", record=False)

    def pick(text):
        if text.lower() in ("q", "none"):
            return None
        if text.isdigit() and 1 <= int(text) <= len(shown):
            return shown[int(text) - 1]
        raise ValueError(f"Enter a number from 1 to {len(shown)}, or q to choose another file.")

    return ask("Molecule number (q = choose another file)", default, pick)


def choose_chains(inv, distances):
    import structure_prep as sp

    defaults = sp.default_chains(distances)
    if len(inv.protein_chains) == 1:
        return list(inv.protein_chains)
    say("\n  The structure has several protein chains. Chains near the binding site are suggested")
    say(f"  (an atom within {sp.CHAIN_NEAR:g} A of the ligand).")
    table([(name, n, f"{distances[name]:.1f}", "keep" if name in defaults else "")
           for name, n in inv.protein_chains.items()],
          headers=("Chain", "Residues", "Nearest to ligand (A)", "Suggested"))

    def validate(text):
        names = text.replace(",", " ").split()
        unknown = [n for n in names if n not in inv.protein_chains]
        if unknown or not names:
            raise ValueError(f"Choose from: {', '.join(inv.protein_chains)} (separated by spaces).")
        return sorted(set(names), key=list(inv.protein_chains).index)

    return ask("Chains to keep", " ".join(defaults), validate)


def ask_molecules(plan):
    """Ask, for each ion, cofactor or other molecule near the ligand, whether to keep it in the receptor."""
    import structure_prep as sp

    if not plan.ask:
        return []
    say("\n  Waters, crystallisation additives and sugars are removed. For the molecules below, decide whether")
    say("  they belong to the receptor (for example a catalytic metal or a cofactor in the pocket).")
    kept = []
    if len(plan.ask) > sp.GROUP_ABOVE:
        groups = {}
        for het in plan.ask:
            groups.setdefault(het.name, []).append(het)
        say(f"  There are {len(plan.ask)} of them, so molecules of the same kind are asked about together.")
        for name, members in groups.items():
            nearest = min(h.distance for h in members)
            if ask_yes_no(f"Keep all {len(members)} {name} ({members[0].kind}, nearest {nearest:.1f} A from the ligand)?",
                          nearest <= sp.KEEP_DEFAULT_RADIUS):
                kept.extend(members)
        return kept
    for het in plan.ask:
        text = (f"Keep {het.label} ({het.kind}, {het.heavy} atom{'s' if het.heavy != 1 else ''}, "
                f"{het.distance:.1f} A from the ligand)?")
        if ask_yes_no(text, sp.default_keep(het)):
            kept.append(het)
    return kept


# ---- the same decisions, taken from a run file -------------------------------------------------------------------


def _norm(label):
    return re.sub(r"\s+", "", str(label)).upper()


def _molecule_lines(hets, suggestion=None, defaults=False):
    import structure_prep as sp

    lines = []
    for het in hets[:12]:
        text = (f"    {het.label:<12} {het.kind:<9} {het.heavy:>3} heavy atoms, {het.contacts} protein contacts"
                if not defaults else
                f"    {het.label:<12} {het.kind:<9} {het.heavy:>3} heavy atoms, {het.distance:.1f} A from the ligand"
                f" (the assistant suggests: {'keep' if sp.default_keep(het) else 'remove'})")
        lines.append(text + ("   <- the assistant suggests this one" if het is suggestion else ""))
    if len(hets) > 12:
        lines.append(f"    ... and {len(hets) - 12} more")
    return lines


def pick_template(inv, wanted):
    import structure_prep as sp

    candidates = sp.template_candidates(inv)
    if not candidates:
        raise Stop("No molecule in the structure file is big enough to be a ligand, so the binding site cannot be "
                   "located from it. Use 'receptor' with 'reference_ligand' or 'box_center' instead.")
    suggestion = sp.suggested_template(inv)
    listing = "\n".join(_molecule_lines(candidates, suggestion))
    if wanted is None:
        raise Stop("'template_molecule' is missing.\n"
                   "  The assistant asks which molecule in the structure sits in the binding site (the box is "
                   "centred on it). The candidates are:\n" + listing + "\n"
                   "  Add the code, chain and number of one of them, for example:  template_molecule: "
                   f"{(suggestion or candidates[0]).label}")
    for het in candidates:
        if _norm(het.label) == _norm(wanted):
            return het
    raise Stop(f"'template_molecule: {wanted}' is not a molecule that can mark the binding site.\n"
               "  The candidates are:\n" + listing)


def pick_chains(inv, wanted, distances):
    import structure_prep as sp

    names = list(inv.protein_chains)
    if wanted is not None:
        unknown = [c for c in wanted if c not in inv.protein_chains]
        if unknown or not wanted:
            raise Stop(f"'chains' names {', '.join(unknown) or 'no chain'}, but the protein chains in the "
                       f"structure are: {', '.join(names)}.")
    if len(names) == 1:
        return names
    suggested = sp.default_chains(distances)
    if wanted is None:
        rows = [f"    {name:<6} {n:>5} residues, nearest to the ligand {distances[name]:.1f} A"
                + ("   <- the assistant suggests keeping it" if name in suggested else "")
                for name, n in inv.protein_chains.items()]
        raise Stop("'chains' is missing.\n"
                   "  The structure has several protein chains and the assistant asks which to keep:\n"
                   + "\n".join(rows) + f"\n  Add the chains to keep, for example:  chains: [{', '.join(suggested)}]")
    return sorted(set(wanted), key=names.index)


def pick_kept(plan, wanted):
    if wanted is not None:
        asked = {_norm(h.label): h for h in plan.ask}
        unknown = [w for w in wanted if _norm(w) not in asked]
        if unknown:
            options = ", ".join(h.label for h in plan.ask) or "none near the ligand"
            raise Stop(f"'keep_molecules' names {', '.join(unknown)}, which is not among the molecules that could be "
                       f"kept. Those are: {options}.")
        return [asked[_norm(w)] for w in dict.fromkeys(wanted)]
    if not plan.ask:
        return []
    raise Stop("'keep_molecules' is missing.\n"
               "  Waters, additives and sugars are removed. The assistant asks, for each of these molecules near the "
               "ligand, whether to keep it in the receptor (for example a catalytic metal or a cofactor):\n"
               + "\n".join(_molecule_lines(plan.ask, defaults=True)) + "\n"
               "  List the ones to keep, or none:  keep_molecules: [ZN A601]   or   keep_molecules: []")


def prepare_from_complex(s, cfg):
    """Clean up a protein-ligand complex. Sets s.receptor, s.prepared and the binding-site ligand's coordinates.
    Returns the coordinates of the ligand that marks the binding site."""
    import structure_prep as sp

    components = sp.Components()
    if cfg is None:
        say("  A PDB (.pdb) or mmCIF (.cif) file, for example downloaded from rcsb.org. It is only read: your")
        say("  file is not changed. Press Tab to complete file names.\n")
        while True:
            s.structure_path = ask("Structure file", validate=file_validator(sp.STRUCTURE_EXTENSIONS),
                                   completer=path_completer)
            try:
                structure, n_models, altlocs = sp.load_structure(s.structure_path)
                inv = sp.inventory(structure, n_models, altlocs, s.structure_path)
            except sp.PrepError as exc:
                fail(str(exc))
                continue
            say("")
            show_inventory(inv)
            template = choose_ligand_of_complex(inv, components)
            if template is None:
                continue
            break
        distances = sp.measure_distances(inv, template)
        chains = choose_chains(inv, distances)
        plan = sp.plan_molecules(inv, template, chains)
        keep = ask_molecules(plan)
    else:
        s.structure_path = cfg["structure"]
        try:
            structure, n_models, altlocs = sp.load_structure(s.structure_path)
            inv = sp.inventory(structure, n_models, altlocs, s.structure_path)
        except sp.PrepError as exc:
            raise Stop(f"The structure file cannot be used: {exc}")
        show_inventory(inv)
        template = pick_template(inv, cfg.get("template_molecule"))
        distances = sp.measure_distances(inv, template)
        chains = pick_chains(inv, cfg.get("chains"), distances)
        plan = sp.plan_molecules(inv, template, chains)
        keep = pick_kept(plan, cfg.get("keep_molecules"))
    try:
        prepared = sp.prepare(structure, inv, template, chains, keep, ccd=None)   # no hydrogens needed for display
    except sp.PrepError as exc:
        if cfg is None:
            fail(str(exc))
            raise SystemExit(1)
        raise Stop(f"The structure could not be prepared: {exc}")
    say("\n  What was done with the structure:")
    for line in prepared.summary:
        say(f"    - {line}")
    if keep:
        warn(f"Kept in the receptor: {', '.join(h.label for h in keep)}. Ions and cofactors may lack AutoDock "
             "partial charges (ADFR reports this for metals such as zinc), which weakens their effect.")
    s.prepared = prepared
    s.kept_labels = [h.label for h in keep]
    s.template_label = template.label
    s.receptor = stage_text(prepared.receptor_pdb, "receptor.pdb")
    return template.xyz


# ---- protonation ---------------------------------------------------------------------------------------------------


def apply_protonation(s, prep):
    """Use the protonated receptor from PDB2PQR for the rest of the run, and report what it did."""
    import protein_prep as pp

    s.protonated = prep
    s.original_receptor = s.original_receptor or s.receptor
    s.receptor = stage_text(prep.pdb, "receptor.pdb")
    if s.prepared is not None:
        s.prepared.receptor_pdb = prep.pdb
    for line in prep.lines():
        ok(line)
    notable = [p for p in prep.pocket if p.notable]
    if notable:
        say(f"\n  Residues within {pp.POCKET_DISTANCE:g} A of the binding-site ligand that deserve a look:")
        table([(p.label, f"{p.pka:.1f}", p.state) for p in notable], headers=("Residue", "pKa", "State"))
    elif prep.pocket:
        note(f"The {len(prep.pocket)} titratable residue(s) near the ligand all keep their usual state.")
    for problem in prep.problems:
        warn(f"PDB2PQR: {problem}.")
    if prep.kept_without_hydrogens:
        warn(f"{', '.join(prep.kept_without_hydrogens)} kept in the receptor got no hydrogens (PDB2PQR only "
             "protonates protein).")
    note("The full record is saved in the run folder as protonation_report.txt.")


def step_protonation(s, cfg, info, ref_xyz):
    """Report gaps and missing atoms, then protonate at a chosen pH if the receptor has no hydrogens."""
    import protein_prep as pp

    say("\n  Protein preparation")
    with open(s.receptor, errors="replace") as fh:
        text = fh.read()
    notes, urgent = pp.quality_lines(pp.read_residues(text), ref_xyz)
    for line in notes:
        note(line)
    for line in urgent:
        warn(line)
    if not notes and not urgent:
        ok("No gaps in the chains and no residues with missing heavy atoms.")

    choice = None if cfg is None else cfg.get("protonation")
    if info["hydrogens"] > 0:
        ok(f"The receptor already has hydrogens ({info['hydrogens']}); they are used as they are.")
        if cfg is None:
            note("Protonation states (histidine form, charged or neutral acids) are not checked. To have them "
                 "predicted, start from a receptor without hydrogens.")
        elif choice is not None:
            note(f"'protonation: {choice}' is ignored because the receptor already has hydrogens.")
        return

    if cfg is None:
        say("\n  Docking needs hydrogens on the receptor. PDB2PQR with PROPKA predicts the charge state of each")
        say("  residue at a pH you choose (histidine form, aspartate and glutamate, termini), places hydrogens so")
        say("  that hydrogen bonds are satisfied, and rebuilds missing side-chain atoms. It takes about a minute.")
        if ask_yes_no("Protonate the protein with PDB2PQR and PROPKA?", True):
            choice = "pdb2pqr"
            ph = ask_number("pH", pp.DEFAULT_PH, minimum=0.0, maximum=14.0)
        else:
            say("\n  ADFR can add hydrogens with a quick rule-based guess (histidine protonation is not optimised).")
            choice = "adfr" if ask_yes_no("Add hydrogens automatically that way?", True) else "none"
            ph = pp.DEFAULT_PH
    else:
        if choice is None:
            raise Stop("'protonation' is missing.\n"
                       "  The receptor has no hydrogens and docking needs them. The assistant asks how to add them; "
                       "choose one:\n"
                       "    protonation: pdb2pqr    predict the charge state of each residue at a pH, place hydrogens "
                       "(takes about a minute; set 'ph:' to change the default 7.4)\n"
                       "    protonation: adfr       a quick rule-based guess; histidine protonation is not optimised\n"
                       "    protonation: none       add nothing (the results will be poor)")
        ph = cfg.get("ph", pp.DEFAULT_PH)

    if choice == "pdb2pqr":
        say(f"  Protonating with PDB2PQR and PROPKA at pH {ph:g}...")
        try:
            prep = pp.protonate(text, ref_xyz, ph)
        except pp.ProteinPrepError as exc:
            if cfg is not None:
                raise Stop(f"Protonation with PDB2PQR did not work: {exc}\n  Set 'protonation: adfr' to use ADFR's "
                           "quicker hydrogen placement instead.")
            warn(f"Protonation with PDB2PQR did not work: {exc}")
            note("Falling back to ADFR's quicker rule-based hydrogen placement.")
            choice = "adfr"
        else:
            apply_protonation(s, prep)
            return
    if choice == "adfr":
        s.add_hydrogens = True
        warn("Hydrogens will be added to the receptor by a quick guess; check histidine protonation if it matters "
             "for your target.")
    else:
        warn("The receptor has no hydrogens and none will be added; the results will be poor.")


# --------------------------------------------------------------------------
# Step 1 and 2: the protein and the binding site
# --------------------------------------------------------------------------


def heavy_coords(mol):
    import numpy as np

    return np.array([p for p, a in zip(mol.GetConformer().GetPositions(), mol.GetAtoms()) if a.GetAtomicNum() > 1])


def load_reference(path):
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    try:
        return Chem.SDMolSupplier(path)[0]
    except (IndexError, OSError):
        return None


def count_sdf_records(path):
    with open(path, errors="replace") as fh:
        return sum(1 for line in fh if line.startswith("$$$$"))


def check_reference(mol, receptor_coords, n_molecules=1):
    """Print findings about a reference ligand. Returns its heavy-atom coordinates, or None if unusable."""
    import numpy as np
    from scipy.spatial import cKDTree

    if mol is None:
        fail("RDKit could not read a molecule from this SDF file.")
        return None
    if mol.GetNumConformers() == 0 or not mol.GetConformer().Is3D():
        fail("The reference ligand has no 3D coordinates. It must be a ligand positioned in the binding site.")
        return None
    xyz = heavy_coords(mol)
    if len(xyz) and np.ptp(xyz, axis=0).min() < 0.01:
        fail("All atoms of the reference ligand lie in one plane, so these are 2D coordinates. It must be a 3D "
             "ligand positioned in the binding site.")
        return None
    ok(f"Reference ligand: {mol.GetNumHeavyAtoms()} heavy atoms.")
    if n_molecules > 1:
        warn(f"The file contains {n_molecules} molecules; only the first is used.")
    if len(receptor_coords):
        distances, _ = cKDTree(np.array(receptor_coords)).query(xyz)
        close = float(np.mean(distances <= 4.5))
        if distances.min() < 1.0:
            warn("Reference ligand atoms sit directly on top of receptor atoms. Is the ligand still inside the "
                 "receptor PDB file?")
        elif close < 0.5:
            warn(f"Only {close:.0%} of the reference ligand's atoms are near the receptor. The ligand and receptor "
                 "are probably in different coordinate systems. Export both from the same structure.")
        else:
            ok(f"The ligand sits in the receptor ({close:.0%} of its atoms are within 4.5 A of protein atoms).")
    return xyz


def parse_centre(text):
    try:
        values = [float(t) for t in re.split(r"[\s,;]+", text.strip().strip("[]()")) if t]
    except ValueError:
        raise ValueError("Enter three numbers, for example  12.5 -3.0 40.2")
    if len(values) != 3:
        raise ValueError("Enter three numbers (x y z), for example  12.5 -3.0 40.2")
    return values


def step_protein(s, cfg):
    heading(1, TOTAL_STEPS, "Protein and binding site")
    if cfg is None:
        say("  What do you have?")
        route = ask_menu([("structure", "A structure file of the protein-ligand complex (PDB or mmCIF). The assistant "
                                        "cleans it up and centres the box on the ligand."),
                          ("files", "A protein PDB file that is ready to use (you say where the binding site is "
                                    "next).")], 1)
        say("")
    else:
        route = cfg["route"]
    ref_xyz = None
    if route == "structure":
        ref_xyz = prepare_from_complex(s, cfg)
        info = analyse_receptor(s.receptor)
        say("")
        report_receptor(info, check_hetero=False)
    else:
        if cfg is None:
            say("  A PDB file of the protein, without the ligand. Press Tab to complete file names.\n")
            mark = len(WARNINGS)
            while True:
                del WARNINGS[mark:]               # forget warnings about a file the user is replacing
                s.receptor = ask("Receptor PDB file", validate=file_validator((".pdb",)), completer=path_completer)
                info = analyse_receptor(s.receptor)
                if report_receptor(info):
                    break
        else:
            s.receptor = cfg["receptor"]
            info = analyse_receptor(s.receptor)
            if not report_receptor(info):
                raise Stop(f"The receptor file cannot be used: {s.receptor}")
        s.original_receptor = s.receptor
    s.receptor_coords = info["coords"]
    s.receptor_hydrogens = info["hydrogens"]
    s.route, s.ref_xyz, s.info = route, ref_xyz, info


def step_site(s, cfg):
    """Where the binding site is, how big the search box is, and then protonation of the receptor."""
    import numpy as np

    say("\n  Binding site and search box")
    ref_xyz = s.ref_xyz
    if s.route == "structure":
        say(f"  Taken from the structure: the box is centred on {s.template_label}.\n")
    else:
        if cfg is None:
            say("  Where is the binding site?")
            how = ask_menu([("ligand", "A ligand that sits in the site (SDF file, e.g. from a crystal structure). "
                                       "The box is centred on it."),
                            ("centre", "I know the coordinates of the centre.")], 1)
            say("")
        else:
            how = "ligand" if cfg.get("reference_ligand") else "centre"
        if how == "ligand":
            if cfg is None:
                mark = len(WARNINGS)
                while True:
                    del WARNINGS[mark:]
                    s.reference_ligand = ask("Ligand SDF file", validate=file_validator((".sdf",)),
                                             completer=path_completer)
                    mol = load_reference(s.reference_ligand)
                    ref_xyz = check_reference(mol, s.receptor_coords, count_sdf_records(s.reference_ligand))
                    if ref_xyz is not None:
                        break
            else:
                s.reference_ligand = cfg["reference_ligand"]
                mol = load_reference(s.reference_ligand)
                ref_xyz = check_reference(mol, s.receptor_coords, count_sdf_records(s.reference_ligand))
                if ref_xyz is None:
                    raise Stop(f"The reference ligand cannot be used (see above): {s.reference_ligand}")
        else:
            if cfg is None:
                s.box_center_given = ask("Box centre: x y z in Angstrom, in the receptor's coordinates",
                                         validate=parse_centre)
            else:
                s.box_center_given = cfg["box_center"]
            ref_xyz = None
    if ref_xyz is not None and len(ref_xyz):
        s.center = [float(c) for c in np.asarray(ref_xyz).mean(axis=0)]
        s.extent = float(np.ptp(np.asarray(ref_xyz), axis=0).max())
    else:
        s.center = [float(c) for c in s.box_center_given]
        s.extent = 0.0
    note(f"Box centre: {s.center[0]:.2f} {s.center[1]:.2f} {s.center[2]:.2f}")

    if len(s.receptor_coords):
        from scipy.spatial import cKDTree

        distance = float(cKDTree(np.array(s.receptor_coords)).query(np.array(s.center))[0])
        if distance > BOX_FAR_FROM_PROTEIN:
            warn(f"The box centre is {distance:.1f} A from the nearest protein atom. Check that the coordinates "
                 "belong to this receptor.")

    suggested = max(20.0, float(math.ceil(s.extent + BOX_MARGIN))) if s.extent else 20.0
    if cfg is None:
        if s.extent:
            say(f"  The reference ligand spans {s.extent:.1f} A. A box of the ligand's size plus {BOX_MARGIN:g} A lets")
            say("  ligands of a similar size move and turn; the suggestion is at least 20 A.")
        s.box_size = ask_number("Search box edge length (Angstrom)", f"{suggested:g}", float, MIN_BOX, True, MAX_BOX)
    else:
        s.box_size = cfg["docking"].get("box_size", suggested)
    if s.extent and s.box_size <= s.extent:
        warn(f"The search box ({s.box_size:g} A) is not larger than the reference ligand ({s.extent:.1f} A at its "
             "longest). Use a bigger box.")
    if s.box_size > LARGE_BOX:
        warn(f"A {s.box_size:g} A box is large: the search becomes less reliable unless the exhaustiveness is raised "
             "(docking settings), and docking takes longer.")
    step_protonation(s, cfg, s.info, ref_xyz)


# --------------------------------------------------------------------------
# Step 2: the molecules to dock
# --------------------------------------------------------------------------


def check_library(path, program):
    """Read the ligand file and run the quick checks on every molecule. Returns a list of LigandResult.

    Only cheap checks are made here (they must work for libraries of many thousands). Whether RDKit can generate 3D
    coordinates for a molecule is found out in the job, where a failure is reported for that molecule only.
    """
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    rows = vs_dock.read_ligands(path)
    results = []
    total = len(rows)
    for i, (name, mol, error) in enumerate(rows):
        if total > 500 and i % 200 == 0:
            sys.stdout.write(f"\r  Checking molecule {i + 1} of {total}...   ")
            sys.stdout.flush()
        if mol is None:
            results.append(LigandResult(i, name, None, False, [error or "could not be read"]))
            continue
        usable, problems = vs_dock.check_ligand(name, mol, program)
        results.append(LigandResult(i, name, mol, usable, problems))
    if total > 500:
        sys.stdout.write("\r" + " " * 50 + "\r")
    return results


def flagged(results):
    return [r for r in results if r.problems]


def report_library(results, program, shown=12):
    unusable = [r for r in results if not r.usable]
    noted = [r for r in results if r.usable and r.problems]
    ok(f"{len(results) - len(unusable)} of {len(results)} molecule(s) can be docked with "
       f"{vs_dock.PROGRAMS[program]['title']}.")
    if unusable:
        say(f"\n  {yellow(str(len(unusable)) + ' molecule(s) cannot be docked:')}")
        table([(r.index + 1, r.name[:30], "; ".join(r.problems)[:90]) for r in unusable[:shown]],
              headers=("#", "Molecule", "Problem"))
        if len(unusable) > shown:
            note(f"... and {len(unusable) - shown} more.")
    if noted:
        say(f"\n  {len(noted)} molecule(s) can be docked but need a note:")
        table([(r.index + 1, r.name[:30], "; ".join(r.problems)[:90]) for r in noted[:shown]],
              headers=("#", "Molecule", "Note"))
        if len(noted) > shown:
            note(f"... and {len(noted) - shown} more.")


def write_library(path, results):
    """Write the given molecules to a file the job can read: SDF if the input was SDF, SMILES otherwise."""
    from rdkit import Chem

    if path.lower().endswith(".sdf"):
        with Chem.SDWriter(path) as writer:
            for r in results:
                r.mol.SetProp("_Name", r.name)
                writer.write(r.mol)
    else:
        with open(path, "w") as fh:
            for r in results:
                name = re.sub(r"\s+", "_", r.name)
                fh.write(f"{Chem.MolToSmiles(r.mol)} {name}\n")


def step_ligands(s, cfg):
    heading(3, TOTAL_STEPS, "Molecules to dock")
    if cfg is None:
        say("  A file with the molecules to dock: an SDF file, or a text file with one SMILES per line")
        say("  (\"SMILES name\", separated by a space, comma or tab; a header line is fine).")
        say("  Any number of molecules works; the job docks several at the same time.\n")
        while True:
            s.ligands = ask("Molecule file", validate=file_validator(vs_runfile.LIGAND_EXTENSIONS), completer=path_completer)
            say("")
            results = check_library(s.ligands, s.program)
            if not results:
                fail("No molecules were found in that file. Choose another.")
                continue
            report_library(results, s.program)
            if not [r for r in results if r.usable]:
                fail("None of these molecules can be docked. Choose another file.")
                continue
            break
    else:
        s.ligands = cfg["ligands"]
        results = check_library(s.ligands, s.program)
        if not results:
            raise Stop(f"No molecules were found in {s.ligands}.")
        report_library(results, s.program)
        if not [r for r in results if r.usable]:
            raise Stop("None of the molecules can be docked (see above).")
    s.library = results
    s.kept = [r for r in results if r.usable]
    bad = [r for r in results if not r.usable]
    if bad:
        remaining = [r for r in results if r.usable]
        if cfg is None:
            say("")
            if ask_yes_no(f"Leave out the {len(bad)} molecule(s) that cannot be docked and dock the other "
                          f"{len(remaining)}?", True):
                s.flagged_choice = "exclude"
                say(f"  {len(remaining)} molecule(s) will be docked.")
            else:
                s.flagged_choice = "include"
                s.kept = results
                warn("Molecules that cannot be docked are included; the job reports each of them as failed, and "
                     "the others are still docked.")
        else:
            choice = cfg.get("flagged_ligands")
            if choice is None:
                lines = [f"    #{r.index + 1} {r.name[:40]}: {'; '.join(r.problems)}" for r in bad[:15]]
                if len(bad) > 15:
                    lines.append(f"    ... and {len(bad) - 15} more")
                raise Stop("'flagged_ligands' is missing.\n"
                           f"  {len(bad)} of {len(results)} molecule(s) cannot be docked. The assistant asks whether to "
                           "leave them out:\n" + "\n".join(lines) + "\n"
                           "    flagged_ligands: exclude    dock only the molecules that can be docked\n"
                           "    flagged_ligands: include    dock all of them; the ones that cannot be docked are "
                           "reported as failed, the others are still docked")
            s.flagged_choice = choice
            if choice == "include":
                s.kept = results
                warn("Molecules that cannot be docked are included; the job reports each of them as failed.")
            else:
                say(f"  {len(remaining)} molecule(s) will be docked ({len(bad)} left out).")
    s.remaining = len(s.kept)
    s.clean = len(s.kept) >= 500
    if len(s.kept) > BIG_LIBRARY:
        warn(f"{len(s.kept):,} molecules is a large library: the job writes a folder per molecule, so intermediate "
             "files are deleted as it goes, and a run may need several jobs (finished molecules are skipped when "
             "you resume).")
    if any("several fragments" in p for r in s.kept for p in r.problems):
        note("Molecules with several fragments (salts) are reduced to their largest fragment, and simple charges "
             "are neutralised, before docking.")
    note("Stereocentres and tautomers are used as they are in your file; they are not enumerated. If a centre is "
         "undefined, one arbitrary form is docked.")


# --------------------------------------------------------------------------
# Step 3: docking settings
# --------------------------------------------------------------------------


def step_settings(s, cfg):
    heading(2, TOTAL_STEPS, "Docking settings")
    settings = {} if cfg is None else cfg["docking"]
    if cfg is None:
        say("  Which docking program? All six were tested here on a real receptor and ligand.\n")
        options = [(p, f"{p:<17} {vs_dock.PROGRAMS[p]['title']:<22} {vs_dock.PROGRAMS[p]['text']}")
                   for p in PROGRAM_ORDER]
        new = ask_menu(options, 1)
    else:
        new = settings.get("program", vs_dock.DEFAULT_PROGRAM)
    s.program = new

    s.exhaustiveness = settings.get("exhaustiveness", vs_dock.DEFAULT_EXHAUSTIVENESS)
    s.num_modes = settings.get("num_modes", vs_dock.DEFAULT_NUM_MODES)
    s.seed = settings.get("seed", vs_dock.DEFAULT_SEED)
    s.rescore = settings.get("rescore")
    if cfg is None:
        say("\n  Search effort per molecule (exhaustiveness) is "
            f"{s.exhaustiveness} by default: a good balance for screening. A bigger number searches more")
        say("  thoroughly and takes proportionally longer.")
        if s.box_size > LARGE_BOX:
            s.exhaustiveness = max(s.exhaustiveness, 16)
            note(f"The box is large ({s.box_size:g} A), so the suggestion is raised to {s.exhaustiveness}.")
        say("\n  NNScore2 can re-score the poses with a neural network (it adds a predicted pKd per pose; about 4 s")
        say("  per molecule). It is a second opinion, not a replacement for the docking score.")
        s.rescore = "nnscore2" if ask_yes_no("Re-score the poses with NNScore2?", False) else None
        if ask_yes_no("Change the search effort, number of poses or random seed?", False):
            s.exhaustiveness = ask_number("Exhaustiveness", s.exhaustiveness, int, 1, True, 128)
            s.num_modes = ask_number("Poses kept per molecule", s.num_modes, int, 1, True, 20)
            s.seed = ask_number("Random seed", s.seed, int, 0)
    elif s.box_size > LARGE_BOX and "exhaustiveness" not in settings:
        s.exhaustiveness = max(s.exhaustiveness, 16)
        note(f"The box is large ({s.box_size:g} A), so the exhaustiveness is raised to {s.exhaustiveness}.")
    seconds = vs_dock.PROGRAMS[s.program]["seconds_per_ligand"] * s.exhaustiveness / vs_dock.DEFAULT_EXHAUSTIVENESS
    say(f"\n  {vs_dock.PROGRAMS[s.program]['title']}, exhaustiveness {s.exhaustiveness}, {s.num_modes} poses per "
        f"molecule, box {s.box_size:g} A"
        f"{', NNScore2 re-scoring' if s.rescore else ''}. About {seconds:.0f} s of one CPU per molecule.")


# --------------------------------------------------------------------------
# Files the job needs, and the engine's command line
# --------------------------------------------------------------------------

MB_PER_LIGAND = 0.04             # a ligand's folder with all intermediates (measured: about 40 KB)
MB_PER_LIGAND_CLEAN = 0.02       # with --clean: only poses.sdf stays


def input_paths(s):
    """Where the job's own copies of its inputs live: inside the run folder, so the run is self-contained."""
    ext = os.path.splitext(s.ligands)[1].lower() or ".smi"
    return os.path.join(s.rundir, "input", "receptor.pdb"), os.path.join(s.rundir, "input", f"molecules{ext}")


def staged_ligands(s):
    """The molecule file as the job will see it (a reduced copy if some molecules were left out)."""
    if len(s.kept) == len(s.library):
        return s.ligands
    if not s.ligand_stage:
        ext = os.path.splitext(s.ligands)[1].lower() or ".smi"
        folder = tempfile.mkdtemp(prefix="vs_ligands_")
        atexit.register(shutil.rmtree, folder, ignore_errors=True)
        s.ligand_stage = os.path.join(folder, f"molecules{ext}")
        write_library(s.ligand_stage, s.kept)
    return s.ligand_stage


def engine_args(s, final=False):
    """Command-line arguments of vs_dock.py for this run. final=False points at the staged files (for the resume
    check), final=True at the copies inside the run folder."""
    receptor, ligands = input_paths(s) if final else (s.receptor, staged_ligands(s))
    args = ["--receptor", receptor, "--ligands", ligands, "--outdir", s.outdir,
            "--center"] + [f"{c:.3f}" for c in s.center] + ["--size"] + [f"{s.box_size:.3f}"] * 3 + [
            "--program", s.program, "--exhaustiveness", str(s.exhaustiveness), "--num_modes", str(s.num_modes),
            "--seed", str(s.seed)]
    if s.rescore:
        args += ["--rescore", s.rescore]
    for flag, on in (("--add_hydrogens", s.add_hydrogens), ("--clean", s.clean), ("--resume", s.resume),
                     ("--force", s.force and not s.resume)):
        if on:
            args.append(flag)
    return args


def previous_run(s):
    """Compare an earlier run in s.outdir with the current choices. Returns (differences, ligands_done) or None if
    the folder does not hold a run of this tool."""
    path = os.path.join(s.outdir, vs_dock.SIGNATURE_FILE)
    if not os.path.isfile(path):
        return None
    done = sum(1 for r in vs_dock.read_results(s.outdir).values() if r.get("status") == "docked")
    try:
        with open(path) as fh:
            old = json.load(fh)
    except (OSError, ValueError):
        return ["the earlier run's settings file, which cannot be read"], done
    args = vs_dock.build_parser().parse_args(engine_args(s))
    return vs_dock.signature_differences(old, vs_dock.run_signature(args)), done


def validate_rundir(path):
    """Return a message if `path` cannot be used as the run folder, otherwise None."""
    if os.path.isfile(path):
        return "That path is a file, not a folder."
    parent = path
    while not os.path.exists(parent):
        nxt = os.path.dirname(parent)
        if nxt == parent:
            break
        parent = nxt
    if not os.path.isdir(parent) or not os.access(parent, os.W_OK | os.X_OK):
        return f"You do not have permission to write in {parent}."
    return None


def results_state(s):
    """Look at s.outdir: ('empty'|'ours'|'foreign')."""
    if not os.path.isdir(s.outdir) or not os.listdir(s.outdir):
        return "empty"
    return "ours" if os.path.isfile(os.path.join(s.outdir, vs_dock.SIGNATURE_FILE)) else "foreign"


def default_jobname(ligands):
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", os.path.splitext(os.path.basename(ligands))[0])
    return ("vs_" + stem)[:40]


def time_limit_validator(text):
    time_validator(text)
    if time_to_minutes(text) > vs_slurm.MAX_TIME_MINUTES:
        raise ValueError("The partition allows at most 7 days (7-00:00:00).")
    return text


# --------------------------------------------------------------------------
# Step 4: how to run it
# --------------------------------------------------------------------------


def choose_existing(s, cfg, earlier):
    """The run folder already holds results. Decide between resuming, overwriting and another folder.
    Returns True if the folder can be used, False if the user wants another one."""
    state = results_state(s)
    if state == "foreign":
        text = (f"{s.outdir} already exists and was not made by this tool, so it will not be touched. "
                "Choose another run folder.")
        if cfg is not None:
            raise Stop(f"'output': {text}")
        fail(text)
        return False
    if cfg is None:
        warn(f"{s.outdir} already holds results.", record=False)
        differences, done = earlier
        if differences:
            say("       They are from a run whose inputs or settings differ from your current choices "
                f"({', '.join(differences)}), so it cannot be resumed.")
        else:
            say(f"       They are from a run with the same inputs and settings: {done} of {len(s.kept)} molecule(s) "
                "are already docked.")
            if ask_yes_no("Resume it? Finished molecules are skipped, the rest are docked", True):
                s.resume, s.force = True, False
                s.remaining = max(0, len(s.kept) - done)
                return True
        if ask_yes_no("Delete these results and write new ones there?", False):
            s.force, s.resume = True, False
            return True
        return False
    choice = cfg.get("existing_output")
    if choice is None:
        differences, done = earlier
        status = (f"They are from a run whose inputs or settings differ from this run file ({', '.join(differences)}), "
                  "so it cannot be resumed." if differences else
                  f"They are from a run with the same inputs and settings: {done} of {len(s.kept)} molecule(s) are "
                  "already docked.")
        raise Stop("'existing_output' is missing.\n"
                   f"  {s.outdir} already holds results. {status}\n"
                   "  The assistant asks what to do about it:\n"
                   "    existing_output: resume       skip the finished molecules and dock the rest (needs the same "
                   "inputs and settings)\n"
                   "    existing_output: overwrite    delete these results and write new ones there\n"
                   "  Or choose another 'output' folder.")
    differences, done = earlier
    if choice == "resume":
        if differences:
            raise Stop("'existing_output: resume' is not possible: these differ from the earlier run: "
                       f"{', '.join(differences)}. Use 'overwrite' or another folder.")
        s.resume, s.force = True, False
        s.remaining = max(0, len(s.kept) - done)
    else:
        s.force, s.resume = True, False
    return True


def step_run(s, cfg):
    heading(4, TOTAL_STEPS, "How to run it")
    run = {} if cfg is None else cfg["run"]
    has_slurm = shutil.which("sbatch") is not None
    s.jobname = run.get("job_name") or default_jobname(s.ligands)
    if cfg is None:
        options = []
        if has_slurm:
            options.append(("submit", "Submit to the cluster queue (SLURM, partition ncpu). Recommended."))
        options += [("run", f"Run now in this terminal (only for small tests: it uses this machine's CPUs)"),
                    ("save", "Only save the job script, do not run anything")]
        say("  Where should the docking run?")
        if not has_slurm:
            warn("SLURM (sbatch) was not found on this computer, so cluster submission is unavailable.")
        s.mode = ask_menu(options, 1)
    else:
        s.mode = run["mode"]
        if s.mode == "submit" and not has_slurm:
            raise Stop("run.mode is 'submit', but SLURM (sbatch) was not found on this computer. Use 'save' to only "
                       "write the job script, or 'run' to run here.")

    # the run folder: inputs, results, job script and log of this run all live in it
    if cfg is None:
        say("\n  Everything for this run (a copy of the inputs, the job script, the log and a results/ folder) is")
        say("  kept in one run folder.")
        default_dir = os.path.join(os.getcwd(), s.jobname)
        while True:
            s.resume, s.force = False, False
            path = ask("Run folder", default_dir, lambda t: os.path.abspath(os.path.expanduser(t.strip("'\""))))
            problem = validate_rundir(path)
            if problem:
                fail(problem)
                continue
            s.rundir, s.outdir = path, os.path.join(path, "results")
            if results_state(s) == "empty":
                break
            earlier = previous_run(s) if results_state(s) == "ours" else None
            if choose_existing(s, cfg, earlier):
                break
            default_dir = path.rstrip("/") + "_new"
    else:
        path = cfg["output"]
        problem = validate_rundir(path)
        if problem:
            raise Stop(f"'output' {path}: {problem}")
        s.rundir, s.outdir = path, os.path.join(path, "results")
        s.resume, s.force = False, False
        if results_state(s) != "empty":
            choose_existing(s, cfg, previous_run(s) if results_state(s) == "ours" else None)
    s.remaining = len(s.kept) if not s.resume else s.remaining

    n = max(1, s.remaining)
    mb = n * (MB_PER_LIGAND_CLEAN if s.clean else MB_PER_LIGAND)
    free = shutil.disk_usage(os.path.dirname(s.rundir) if os.path.isdir(os.path.dirname(s.rundir)) else os.getcwd()).free / 1e6
    if mb > free:
        warn(f"This run may need about {mb / 1000:.1f} GB but only {free / 1000:.1f} GB is free where the results go.")
    elif mb > 0.5 * free:
        warn(f"This run may need about {mb / 1000:.1f} GB, over half of the {free / 1000:.1f} GB free where the "
             "results go.")

    if cfg is None:
        s.jobname = ask("Job name", s.jobname, lambda t: re.sub(r"[^A-Za-z0-9_.-]", "_", t))

    # CPUs and time
    if s.mode == "run":
        available = vs_dock.available_cpus()
        s.cpus = run.get("cpus", min(RUN_HERE_CPUS, available, n))
        if cfg is None:
            say(f"\n  This machine has {available} CPU(s) available to you. A terminal on a shared machine is meant "
                "for small tests.")
            s.cpus = ask_number("Molecules to dock at the same time", s.cpus, int, 1, True, available)
        elif s.cpus > available:
            raise Stop(f"run.cpus is {s.cpus}, but only {available} CPU(s) are available here.")
        return
    seconds = vs_dock.PROGRAMS[s.program]["seconds_per_ligand"]
    s.cpus = run.get("cpus", min(DEFAULT_CPUS, n))
    if cfg is None:
        say(f"\n  Cluster job: each molecule uses one CPU, so more CPUs finish sooner. The default is {s.cpus}; the "
            "partition has up to 256 per node.")
        s.cpus = ask_number("CPUs (molecules docked at the same time)", s.cpus, int, 1, True, 256)
    suggested = vs_slurm.estimate_minutes(n, seconds, s.exhaustiveness, s.cpus, bool(s.rescore))
    s.time_limit = run.get("time") or fmt_minutes(suggested)
    if cfg is None:
        say(f"  Time limit: estimated from {seconds} s of one CPU per molecule at the default exhaustiveness, "
            "doubled for safety.")
        s.time_limit = ask("Time limit (hours:minutes:seconds)", s.time_limit, time_limit_validator)
    else:
        try:
            time_limit_validator(s.time_limit)
        except ValueError as exc:
            raise Stop(f"run.time: {exc}")
    if time_to_minutes(s.time_limit) < suggested / vs_slurm.SAFETY_FACTOR:
        warn(f"The time limit {s.time_limit} is shorter than the estimated need (about "
             f"{fmt_minutes(suggested / vs_slurm.SAFETY_FACTOR)} without the safety margin); the job may be stopped "
             "early. It can be resumed.")
    if suggested >= vs_slurm.MAX_TIME_MINUTES:
        warn("Even the longest allowed time (7 days) may not be enough for this many molecules on this many CPUs. "
             "Use more CPUs, or resume the run in a second job (finished molecules are skipped).")

    s.partition = run.get("partition", vs_slurm.DEFAULT_PARTITION)
    if cfg is None:
        say(f"\n  The job will use: partition {s.partition}, {s.cpus} CPUs, {vs_slurm.MEM_PER_CPU} memory per CPU, "
            f"time limit {s.time_limit}.")
        if ask_yes_no("Change the partition?", False):
            s.partition = ask("Partition", s.partition)
    if s.mode != "run" and has_slurm:
        ensure_partition_access(s, cfg)
    if cfg is not None:
        say(f"  {s.mode.capitalize()}: {s.cpus} molecule(s) at a time"
            + (f", partition {s.partition}, time limit {s.time_limit}" if s.mode != "run" else "")
            + f". Run folder {s.rundir}.")


def ensure_partition_access(s, cfg):
    """Make sure this account can use the chosen partition for such a job, before the script is written."""
    while True:
        result = vs_slurm.check_submission(s.partition, s.cpus, s.time_limit)
        if result.ok is True:
            return
        if result.ok is None:
            note(f"Could not check whether you may use the partition {s.partition}: {result.reason}.")
            return
        if cfg is not None:
            raise Stop(f"You cannot submit to the partition {s.partition}: {result.reason}.\n"
                       "  Set 'partition' in the 'run' section.")
        fail(f"You cannot submit to the partition {s.partition}: {result.reason}.")
        if ask_yes_no("Use it anyway (the job will be refused when it is submitted)?", False):
            warn(f"Slurm will refuse jobs for the partition {s.partition}: {result.reason}.")
            return
        s.partition = ask("Partition", vs_slurm.DEFAULT_PARTITION)


# --------------------------------------------------------------------------
# Step 5: review
# --------------------------------------------------------------------------


def step_review(s, confirm=True):
    heading(5, TOTAL_STEPS, "Review")
    names = ", ".join(r.name for r in s.kept[:3]) + (" ..." if len(s.kept) > 3 else "")
    if s.prepared is not None:
        protein = f"{s.structure_path} (chains {', '.join(s.prepared.kept_chains)}, cleaned)"
        site = f"centred on {s.template_label}: {s.center[0]:.2f} {s.center[1]:.2f} {s.center[2]:.2f}"
    else:
        protein = s.original_receptor or s.receptor
        what = "reference ligand " + os.path.basename(s.reference_ligand) if s.reference_ligand else "given centre"
        site = f"{what}: {s.center[0]:.2f} {s.center[1]:.2f} {s.center[2]:.2f}"
    rows = [("Protein", protein), ("Binding site", site), ("Search box", f"{s.box_size:g} A edge"),
            ("Molecules", f"{len(s.kept)} of {len(s.library)} in {s.ligands} ({names})"),
            ("Program", f"{vs_dock.PROGRAMS[s.program]['title']}, exhaustiveness {s.exhaustiveness}, "
                        f"{s.num_modes} poses per molecule, seed {s.seed}")]
    if s.rescore:
        rows.append(("Re-scoring", "NNScore2 (predicted pKd per pose)"))
    if s.protonated is not None:
        rows.append(("Protonation", f"PDB2PQR + PROPKA at pH {s.protonated.ph:g} ({s.protonated.hydrogens} hydrogens)"))
    elif s.add_hydrogens:
        rows.append(("Hydrogens", "added to the receptor by ADFR (quick rule-based guess)"))
    note_text = " (resuming: finished molecules are skipped)" if s.resume else (
        " (existing results will be deleted)" if s.force else "")
    rows.append(("Run folder", s.rundir + note_text))
    rows.append(("Delete intermediates", "yes" if s.clean else "no"))
    if s.mode == "run":
        rows.append(("Run here", f"{s.cpus} molecule(s) at a time"))
    else:
        rows += [("Partition / CPUs", f"{s.partition} / {s.cpus} ({vs_slurm.MEM_PER_CPU} each)"),
                 ("Time limit", s.time_limit), ("Job name", s.jobname)]
    action = {"submit": "Submit the job to the queue", "run": "Run the docking now",
              "save": "Save the job script"}[s.mode]
    rows.append(("Action", action))
    table(rows, indent="  ")
    if WARNINGS:
        say(f"\n  {yellow('Warnings to consider:')}")
        for w in dict.fromkeys(WARNINGS):
            say(f"   - {w}")
    say("")
    if not confirm:
        return True
    return ask_yes_no(f"{action}?", True)


# --------------------------------------------------------------------------
# Running, submitting, results
# --------------------------------------------------------------------------


def write_files(s):
    """Save everything the job needs inside the run folder: copies of the inputs, the report, the job script."""
    receptor, ligands = input_paths(s)
    os.makedirs(os.path.dirname(receptor), exist_ok=True)
    shutil.copyfile(s.receptor, receptor)
    shutil.copyfile(staged_ligands(s), ligands)
    if s.protonated is not None:
        with open(os.path.join(s.rundir, "protonation_report.txt"), "w") as fh:
            fh.write(s.protonated.report_text())
    if s.mode != "run":
        s.script_path = os.path.join(s.rundir, f"submit_{s.jobname}.sh")
        with open(s.script_path, "w") as fh:
            fh.write(vs_slurm.build_script(s.jobname, s.rundir, s.partition, s.time_limit, s.cpus,
                                           engine_args(s, final=True), REPO_DIR))
        os.chmod(s.script_path, 0o755)
        ok(f"Job script saved: {s.script_path}")


def run_file_values(s):
    """The answers of this run, in the form vs_runfile.render() writes."""
    values = {}
    if s.prepared is not None:
        values.update(structure=s.structure_path, template_molecule=s.template_label,
                      chains=list(s.prepared.kept_chains), keep_molecules=list(s.kept_labels))
    else:
        values["receptor"] = s.original_receptor or s.receptor
        if s.reference_ligand:
            values["reference_ligand"] = s.reference_ligand
        else:
            values["box_center"] = [round(c, 3) for c in s.center]
    values.update(ligands=s.ligands, output=s.rundir)
    if s.resume:
        values["existing_output"] = "resume"     # 'overwrite' is deliberately not saved: running the file again must ask
    if s.receptor_hydrogens == 0:
        values["protonation"] = "pdb2pqr" if s.protonated is not None else ("adfr" if s.add_hydrogens else "none")
        if s.protonated is not None:
            values["ph"] = s.protonated.ph
    values["flagged_ligands"] = s.flagged_choice
    values["docking"] = {"program": s.program, "exhaustiveness": s.exhaustiveness, "num_modes": s.num_modes,
                         "seed": s.seed, "box_size": s.box_size, "rescore": s.rescore or "none"}
    run = {"mode": s.mode, "job_name": s.jobname, "cpus": s.cpus}
    if s.mode != "run":
        run.update(time=s.time_limit, partition=s.partition)
    values["run"] = run
    return values


def write_run_file(s):
    """Save this run's answers in the run folder, so it can be repeated with `<command> <file>`."""
    path = os.path.join(s.rundir, "run.yaml")
    try:
        os.makedirs(s.rundir, exist_ok=True)
        with open(path, "w") as fh:
            fh.write(vs_runfile.render(run_file_values(s), RUN_COMMAND))
    except OSError as exc:
        warn(f"Could not save the run file: {exc}", record=False)
        return None
    ok(f"Run file saved: {path}")
    note(f"Repeat this run, or change a value and run it again, with:  {RUN_COMMAND} {path}")
    return path


def result_rows(s):
    return vs_dock.read_results(s.outdir)


def show_results(s, job_id=None):
    rows = result_rows(s)
    if not rows:
        say(f"\n  No results have been written to {s.outdir} yet.")
        return
    docked = [r for r in rows.values() if r.get("status") == "docked"]
    failed = [r for r in rows.values() if r.get("status") != "docked"]
    docked.sort(key=vs_dock.sort_key)
    say("\n  Best results (score in kcal/mol: more negative is better):")
    headers = ("Rank", "Molecule", "Score") + (("NNScore2 pKd",) if s.rescore else ())
    table([(n, r["name"][:32], r["best_score"]) + ((r.get("pkd_best", ""),) if s.rescore else ())
           for n, r in enumerate(docked[:10], 1)], headers=headers)
    say(f"\n  {len(docked)} of {len(s.kept)} molecule(s) docked in {s.outdir}")
    if failed:
        say(f"\n  {red(str(len(failed)) + ' molecule(s) failed.')} The others were still docked:")
        for r in sorted(failed, key=lambda r: int(r["index"]))[:10]:
            say(f"    #{int(r['index']) + 1} {r['name'][:30]}: {r.get('note', '')}")
        if len(failed) > 10:
            say(f"    ... and {len(failed) - 10} more (see results.csv)")
    if docked:
        say(f"\n  Files in {s.outdir}:")
        say("    ranking.csv                 all molecules, best score first")
        say("    results.csv                 the same in input order")
        say("    top_poses.sdf               the best pose of every molecule (with all hydrogens)")
        say("    all_poses.sdf               every pose of every molecule")
        say("    ligands/<group>/<number>/   one folder per molecule, with its poses.sdf")
        say("  Docking scores are estimates of binding strength. Use them to rank and to choose molecules to look at,")
        say("  and check the poses; a good score does not prove that a molecule binds.")
    if job_id and len(docked) + len(failed) < len(s.kept):
        log = os.path.join(s.rundir, f"{s.jobname}_{job_id}.log")
        if os.path.isfile(log):
            say(f"\n  Last lines of {log}:")
            with open(log, errors="replace") as fh:
                for line in fh.readlines()[-12:]:
                    say("    " + line.rstrip())


def run_here(s):
    command = [sys.executable, "-W", "ignore", os.path.join(REPO_DIR, "vs_dock.py")] + engine_args(s, final=True) + [
        "--cpus", str(s.cpus)]
    say("\n  Running... (Ctrl+C to stop; finished molecules are kept and can be resumed)\n")
    try:
        return subprocess.run(command, cwd=s.rundir).returncode
    except KeyboardInterrupt:
        fail("Stopped.")
        return None


def execute(s, ask_watch=True, wait=False):
    """Do what the review said: run the docking here, or write the job script and submit it."""
    os.makedirs(s.rundir, exist_ok=True)
    write_files(s)
    if s.mode == "run":
        code = run_here(s)
        if code is None:
            sys.exit(130)
        show_results(s)
        if code not in (0, 1):
            fail("The run stopped with an error. See the messages above.")
            sys.exit(code)
        if code == 1:
            fail("Some molecules failed. See the results above.")
            sys.exit(1)
        return
    if s.mode == "save":
        say(f"\n  Submit it later with:  sbatch {s.script_path}")
        return
    say("")
    job_id = vs_slurm.submit(s.script_path, s.jobname)
    if job_id is None:
        fail("The job was not submitted.")
        why = vs_slurm.check_submission(s.partition, s.cpus, s.time_limit)
        if why.ok is False:
            say(f"  Reason: {why.reason}. Run the assistant again and choose a partition you can use.")
        sys.exit(1)
    ok(f"Submitted job {job_id}.")
    log = os.path.join(s.rundir, f"{s.jobname}_{job_id}.log")
    for line in vs_slurm.follow_up_lines(job_id, log):
        say(f"  {line}")
    say(f"\n  Results will be written to {s.outdir}.")
    say(f"  If the job is stopped early, run {os.path.join(s.rundir, 'run.yaml')} again and choose to resume.")
    say("")
    if (ask_yes_no("Watch the job until it finishes?", True) if ask_watch else wait):
        if vs_slurm.watch(job_id, say, warn):
            show_results(s, job_id)


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------


def adfr_ready(path=None):
    exe = os.path.join(path or vs_dock.DEFAULT_ADFR_PATH, "bin", "prepare_receptor")
    return os.path.isfile(exe) and os.access(exe, os.X_OK)


def check_environment():
    try:
        import gemmi  # noqa: F401
        import meeko  # noqa: F401
        import rdkit  # noqa: F401
        import scipy  # noqa: F401
        import yaml  # noqa: F401
    except ImportError:
        print("This tool must be started through pixi so that RDKit and Meeko are available:\n\n"
              f"    {RUN_COMMAND}\n")
        sys.exit(1)
    if not adfr_ready():
        if UNDER_DDSUITE:
            print("ADFRsuite, which prepares the receptor, is not installed in this copy of the tool. "
                  "Ask the maintainer of the suite to reinstall it.")
        else:
            print("ADFRsuite, which prepares the receptor, is not installed yet. Install everything once with:\n\n"
                  "    pixi install --all\n\n"
                  f"then start this tool again with `{RUN_COMMAND}`.")
        sys.exit(1)


def run_steps(s, cfg):
    step_protein(s, cfg)
    step_site(s, cfg)
    step_settings(s, cfg)
    step_ligands(s, cfg)
    step_run(s, cfg)


def run_interactive():
    s = Settings()
    say(bold("\nVirtual screening"))
    say("Dock a library of molecules into a protein with a Vina-family program, and rank them by score.")
    say("Answer each question, or press Enter to accept the suggestion in [brackets]. Ctrl+C quits.")
    say(f"Everything you answer is saved as a run file, so the same run can be repeated with `{RUN_COMMAND} <file>`.")
    submitted = False
    try:
        run_steps(s, None)
        if not step_review(s):
            say("\nCancelled. Nothing was run.")
            return
        submitted = True
        os.makedirs(s.rundir, exist_ok=True)
        write_run_file(s)
        execute(s)
    except (KeyboardInterrupt, EOFError):
        say("\n\nStopped." + ("" if submitted else " Nothing was run."))
        sys.exit(130)


def run_from_file(path, check_only=False):
    try:
        cfg = vs_runfile.load(path)
    except vs_runfile.RunFileError as exc:
        print(f"\n{exc}\n\nNothing was run. Print an annotated example with:  {RUN_COMMAND} --example")
        sys.exit(2)
    s = Settings()
    say(bold("\nVirtual screening"))
    say(f"Run file: {cfg['file']}")
    try:
        run_steps(s, cfg)
    except Stop as exc:
        print(f"\n{red('Stopped.')} {exc}\n\n  Nothing was run. The run file: {cfg['file']}")
        sys.exit(2)
    except (KeyboardInterrupt, EOFError):
        say("\n\nStopped. Nothing was run.")
        sys.exit(130)
    step_review(s, confirm=False)
    if check_only:
        say("\nThe run file is complete and everything checks out. Nothing was run (--check).")
        return
    os.makedirs(s.rundir, exist_ok=True)
    if os.path.join(s.rundir, "run.yaml") != cfg["file"]:
        write_run_file(s)       # a copy with every default filled in, in the run folder
    execute(s, ask_watch=False, wait=cfg["run"].get("watch", False))
