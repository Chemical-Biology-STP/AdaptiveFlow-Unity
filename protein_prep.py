"""Protein preparation: protonation at a chosen pH, and checks for gaps and missing atoms.

Why this exists. ADFR can add hydrogens, but only by a fixed rule: on a test structure it made all
six histidines it handled cationic (both ring nitrogens carrying a hydrogen), which leaves them unable to
accept a hydrogen bond during docking, and it gave gap ends artificial charged termini. PDB2PQR with
PROPKA instead predicts pKa values from the structure, assigns each histidine its tautomer or charge,
protonates the other titratable groups at the chosen pH, optimises the hydrogen-bond network and rebuilds
missing heavy atoms.

The functions only analyse and transform; asking the user is dock_cli.py's job.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

import numpy as np

DEFAULT_PH = 7.4
POCKET_DISTANCE = 6.0        # residues this close to the template line the binding site
GAP_NEAR = 10.0              # a gap end this close to the template is worth a warning
PEPTIDE_BOND_MAX = 2.0       # C(i)-N(i+1) is 1.33 A in a peptide bond; longer means residues are missing
FLIP_SHIFT = 0.5             # a heavy atom that moved more than this was repositioned (a side-chain flip)
NOTABLE_PKA_WINDOW = 1.5     # pocket residues with a pKa this close to the pH are flagged as borderline
TIMEOUT_SECONDS = 900

# Side-chain heavy atoms of the standard residues (the backbone is N, CA, C, O for all).
BACKBONE = ("N", "CA", "C", "O")
SIDE_CHAIN = {
    "ALA": ("CB",), "ARG": ("CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"), "ASN": ("CB", "CG", "OD1", "ND2"),
    "ASP": ("CB", "CG", "OD1", "OD2"), "CYS": ("CB", "SG"), "GLN": ("CB", "CG", "CD", "OE1", "NE2"),
    "GLU": ("CB", "CG", "CD", "OE1", "OE2"), "GLY": (), "HIS": ("CB", "CG", "ND1", "CD2", "CE1", "NE2"),
    "ILE": ("CB", "CG1", "CG2", "CD1"), "LEU": ("CB", "CG", "CD1", "CD2"), "LYS": ("CB", "CG", "CD", "CE", "NZ"),
    "MET": ("CB", "CG", "SD", "CE"), "PHE": ("CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"), "PRO": ("CB", "CG", "CD"),
    "SER": ("CB", "OG"), "THR": ("CB", "OG1", "CG2"),
    "TRP": ("CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
    "TYR": ("CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"), "VAL": ("CB", "CG1", "CG2"),
}
# Names the AMBER force field uses for protonation variants, mapped to the residue they are a state of.
VARIANT_OF = {"HID": "HIS", "HIE": "HIS", "HIP": "HIS", "HSD": "HIS", "HSE": "HIS", "HSP": "HIS",
              "ASH": "ASP", "GLH": "GLU", "LYN": "LYS", "CYX": "CYS", "CYM": "CYS", "TYM": "TYR"}
STATE_TEXT = {
    "HID": "neutral, hydrogen on ND1", "HIE": "neutral, hydrogen on NE2", "HIP": "positive, both nitrogens protonated",
    "ASH": "neutral, protonated carboxylic acid", "GLH": "neutral, protonated carboxylic acid",
    "LYN": "neutral amine", "CYX": "in a disulfide bond", "CYM": "thiolate, negative", "TYM": "phenolate, negative",
}
DEFAULT_STATE = {"ASP": "negative (carboxylate)", "GLU": "negative (carboxylate)", "LYS": "positive (ammonium)",
                 "ARG": "positive (guanidinium)", "CYS": "neutral thiol", "TYR": "neutral phenol"}
TITRATABLE = frozenset({"ASP", "GLU", "LYS", "ARG", "HIS", "CYS", "TYR"})


class ProteinPrepError(Exception):
    """Protonation could not be done; the message says why in plain words."""


# --------------------------------------------------------------------------
# Reading a PDB file as residues
# --------------------------------------------------------------------------


@dataclass
class Residue:
    chain: str
    num: int
    icode: str
    name: str
    atoms: dict            # atom name -> xyz (numpy array)
    hetero: bool = False

    @property
    def key(self):
        return (self.chain, self.num, self.icode)

    @property
    def label(self):
        return f"{self.name} {self.chain}{self.num}{self.icode.strip()}"

    @property
    def base(self):
        return VARIANT_OF.get(self.name, self.name)


def read_residues(pdb_text):
    """The residues of a PDB file, in file order (first alternate location only, hydrogens included)."""
    residues = {}
    for line in pdb_text.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if line[16:17] not in (" ", "A", ""):
            continue
        try:
            key = (line[21:22], int(line[22:26]), line[26:27])
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        except ValueError:
            continue
        res = residues.setdefault(key, Residue(key[0], key[1], key[2], line[17:20].strip(), {}, line.startswith("HETATM")))
        res.atoms.setdefault(line[12:16].strip(), xyz)
    return list(residues.values())


def is_hydrogen(name):
    return bool(re.match(r"^\d?H", name))


def template_distance(residue, template_xyz, heavy_only=True):
    """Smallest distance from any atom of the residue to any template atom."""
    if template_xyz is None or len(template_xyz) == 0:
        return None
    xyz = np.array([p for n, p in residue.atoms.items() if not (heavy_only and is_hydrogen(n))])
    if len(xyz) == 0:
        return None
    return float(np.linalg.norm(xyz[:, None, :] - template_xyz[None, :, :], axis=2).min())


# --------------------------------------------------------------------------
# Checks that do not need PDB2PQR: gaps and missing atoms
# --------------------------------------------------------------------------


@dataclass
class Gap:
    chain: str
    before: str            # label of the residue before the gap
    after: str
    bond_length: float     # between them (a real peptide bond is 1.33 A)
    distance: float = None  # from the nearer end to the template


@dataclass
class Incomplete:
    label: str
    missing: tuple
    distance: float = None


def find_gaps(residues, template_xyz=None):
    """Places where consecutive residues are not bonded, so residues are missing between them."""
    gaps = []
    protein = [r for r in residues if r.base in SIDE_CHAIN and not r.hetero]
    for a, b in zip(protein, protein[1:]):
        if a.chain != b.chain or "C" not in a.atoms or "N" not in b.atoms:
            continue
        length = float(np.linalg.norm(a.atoms["C"] - b.atoms["N"]))
        if length > PEPTIDE_BOND_MAX:
            near = [d for d in (template_distance(a, template_xyz), template_distance(b, template_xyz)) if d is not None]
            gaps.append(Gap(a.chain, a.label, b.label, round(length, 1), min(near) if near else None))
    return gaps


def find_incomplete(residues, template_xyz=None):
    """Standard residues that lack heavy atoms (the terminal OXT is not required)."""
    found = []
    for r in residues:
        if r.hetero or r.base not in SIDE_CHAIN:
            continue
        missing = tuple(n for n in BACKBONE + SIDE_CHAIN[r.base] if n not in r.atoms)
        if missing:
            found.append(Incomplete(r.label, missing, template_distance(r, template_xyz)))
    return found


# --------------------------------------------------------------------------
# PDB2PQR + PROPKA
# --------------------------------------------------------------------------


def pdb2pqr_command():
    exe = shutil.which("pdb2pqr")
    return [exe] if exe else [sys.executable, "-m", "pdb2pqr"]


def run_pdb2pqr(pdb_text, ph):
    """Run PDB2PQR with PROPKA. Returns (protonated PDB text, log text)."""
    with tempfile.TemporaryDirectory(prefix="pdb2pqr_") as folder:
        source = os.path.join(folder, "in.pdb")
        with open(source, "w") as fh:
            fh.write(pdb_text)
        command = pdb2pqr_command() + ["--ff", "AMBER", "--ffout", "AMBER", "--with-ph", f"{ph:g}",
                                       "--titration-state-method", "propka", "--keep-chain",
                                       "--pdb-output", os.path.join(folder, "out.pdb"), source,
                                       os.path.join(folder, "out.pqr")]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=TIMEOUT_SECONDS, check=False)
        except subprocess.TimeoutExpired:
            raise ProteinPrepError(f"PDB2PQR did not finish within {TIMEOUT_SECONDS // 60} minutes.")
        except OSError as exc:
            run_command = os.environ.get("DDSUITE_RUN_COMMAND") or "pixi run dock"
            raise ProteinPrepError(f"PDB2PQR could not be started ({exc}). Run through `{run_command}`.")
        log = result.stdout + result.stderr
        log_file = os.path.join(folder, "out.log")
        if os.path.isfile(log_file):
            with open(log_file, errors="replace") as fh:
                log += fh.read()
        out = os.path.join(folder, "out.pdb")
        if result.returncode != 0 or not os.path.isfile(out):
            tail = " | ".join(l.strip() for l in log.splitlines() if "ERROR" in l or "Error" in l)[-300:]
            raise ProteinPrepError(f"PDB2PQR failed (exit code {result.returncode}). {tail}".strip())
        with open(out) as fh:
            return fh.read(), log


_PKA_LINE = re.compile(r"^\s*([A-Z][A-Z0-9+\-]{1,3})\s+(-?\d+)([A-Za-z]?)\s+([A-Za-z0-9])\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)")


def parse_propka(log):
    """pKa values from the PROPKA summary block of the log.

    Returns {(chain, residue number): (group name, pKa)}. Termini appear as groups named N+ and C-.
    """
    values = {}
    inside = False
    for line in log.splitlines():
        if "SUMMARY OF THIS PREDICTION" in line:
            inside = True
            continue
        if not inside:
            continue
        if line.startswith("----") or line.startswith("Free energy"):
            break
        m = _PKA_LINE.match(line)
        if m:
            values[(m.group(4), int(m.group(2)))] = (m.group(1), float(m.group(5)))
    return values


def parse_warnings(log):
    """The things PDB2PQR reports that matter to the user: repairs and residues it could not handle."""
    repaired, problems = [], []
    seen = set()
    for line in log.splitlines():
        m = re.search(r"Added atom (\S+) to residue (\S+) (\S+) (\S+)", line)
        if m:
            text = f"{m.group(1)} added to {m.group(2)} {m.group(3)}{m.group(4)}"
            if text not in repaired:
                repaired.append(text)
            continue
        m = re.search(r"Unable to debump (\S+) (\S+) (\S+)", line)
        if m:
            text = f"{m.group(1)} {m.group(2)}{m.group(3)}: hydrogens could not be placed without a clash"
            if text not in seen:
                seen.add(text)
                problems.append(text)
            continue
        m = re.search(r"Missing atoms or failed protonation for (\S+) (\d+) (\S+)", line)
        if m:
            text = f"{m.group(1)} {m.group(3)}{m.group(2)}: protonation failed"
            if text not in seen:
                seen.add(text)
                problems.append(text)
    return repaired, problems


@dataclass
class PocketResidue:
    label: str
    pka: float
    state: str            # what the residue became
    notable: bool         # a non-default state, or a pKa close to the pH


@dataclass
class ProteinPrep:
    pdb: str
    ph: float
    changed: list = field(default_factory=list)        # (label, from, to) for residues given another state
    flipped: list = field(default_factory=list)        # Asn/Gln/His whose side chain was flipped to fit the H-bond network
    pocket: list = field(default_factory=list)         # PocketResidue for titratable residues near the template
    gaps: list = field(default_factory=list)
    incomplete: list = field(default_factory=list)     # residues that lacked heavy atoms in the input
    repaired: list = field(default_factory=list)       # heavy atoms PDB2PQR added
    problems: list = field(default_factory=list)
    kept_without_hydrogens: list = field(default_factory=list)
    hydrogens: int = 0
    log: str = ""

    def lines(self):
        """A short account for the wizard."""
        out = [f"Protonated at pH {self.ph:g} with PDB2PQR and PROPKA: {self.hydrogens} hydrogens, "
               "hydrogen-bond network optimised."]
        counts = {}
        for _, was, now in self.changed:
            counts[now] = counts.get(now, 0) + 1
        if counts:
            out.append("Residues given another state: " + ", ".join(f"{n} {k}" for k, n in sorted(counts.items())))
        if self.flipped:
            out.append(f"Side chains flipped to improve hydrogen bonding (heavy atoms moved): {len(self.flipped)} "
                       f"({', '.join(self.flipped[:4])}{', ...' if len(self.flipped) > 4 else ''})")
        if self.repaired:
            out.append(f"Missing heavy atoms rebuilt: {len(self.repaired)} ({', '.join(self.repaired[:4])}"
                       f"{', ...' if len(self.repaired) > 4 else ''})")
        return out

    def report_text(self):
        """The full record, saved beside the results."""
        r = [f"Protein preparation report", f"pH: {self.ph:g}", "Tools: PDB2PQR (AMBER names) with PROPKA", ""]
        r.append(f"Hydrogens in the prepared protein: {self.hydrogens}")
        r.append("")
        r.append(f"Residues within {POCKET_DISTANCE:g} A of the template that can change protonation state:")
        if self.pocket:
            r.append(f"  {'residue':<12}{'pKa':>7}  state")
            for p in self.pocket:
                r.append(f"  {p.label:<12}{p.pka:>7.2f}  {p.state}{'   <-- look at this one' if p.notable else ''}")
        else:
            r.append("  (none)")
        r.append("")
        r.append(f"All residues given a non-default state ({len(self.changed)}):")
        r += [f"  {label}: {was} -> {now} ({STATE_TEXT.get(now, now)})" for label, was, now in self.changed] or ["  (none)"]
        r.append("")
        r.append(f"Side chains flipped by PDB2PQR (Asn/Gln amide or His ring turned 180 degrees so that hydrogen "
                 f"bonds fit; heavy atoms moved) ({len(self.flipped)}):")
        r += [f"  {label}" for label in self.flipped] or ["  (none)"]
        r.append("")
        r.append("Gaps in the chain (residues missing in the structure):")
        r += [gap_text(g) for g in self.gaps] or ["  (none)"]
        r.append("")
        r.append("Residues that lacked heavy atoms in the input:")
        r += [incomplete_text(i) for i in self.incomplete] or ["  (none)"]
        r.append("")
        r.append(f"Heavy atoms rebuilt by PDB2PQR ({len(self.repaired)}):")
        r += [f"  {a}" for a in self.repaired] or ["  (none)"]
        if self.problems:
            r += ["", "Problems reported by PDB2PQR:"] + [f"  {p}" for p in self.problems]
        if self.kept_without_hydrogens:
            r += ["", "Molecules kept in the receptor that were not given hydrogens (PDB2PQR only handles protein): "
                  + ", ".join(self.kept_without_hydrogens)]
        r += ["", "Notes: hydrogen placement and the histidine tautomer come from a hydrogen-bond network "
                  "optimisation, and pKa values are empirical predictions. Check the residues marked above "
                  "if your target depends on them."]
        return "\n".join(r) + "\n"


def gap_text(g):
    near = f", {g.distance:.1f} A from the template" if g.distance is not None else ""
    return f"  {g.before} -> {g.after} (chain {g.chain}): the C-N distance is {g.bond_length} A instead of 1.3{near}"


def incomplete_text(i):
    near = f", {i.distance:.1f} A from the template" if i.distance is not None else ""
    return f"  {i.label}: missing {', '.join(i.missing)}{near}"


def _titratable_pocket(after, pka, ph, template_xyz):
    rows = []
    for r in after:
        if r.hetero or r.base not in TITRATABLE:
            continue
        d = template_distance(r, template_xyz)
        if d is None or d > POCKET_DISTANCE:
            continue
        group = pka.get((r.chain, r.num))
        if group is None:
            continue
        default = r.name == r.base and r.base != "HIS"
        state = STATE_TEXT.get(r.name) or DEFAULT_STATE.get(r.name, r.name)
        notable = (not default) or abs(group[1] - ph) <= NOTABLE_PKA_WINDOW
        rows.append(PocketResidue(r.label, group[1], f"{r.name}: {state}" if r.name != r.base else state, notable))
    return rows


def protonate(pdb_text, template_xyz=None, ph=DEFAULT_PH, runner=None):
    """Protonate a protein at `ph`. Heavy atoms are kept as they are, missing ones are rebuilt.

    pdb_text: the protein (and any molecules to keep) as PDB text. Header records are dropped first,
    because PDB2PQR rejects the dummy cell record that cleaned structures carry.
    template_xyz: coordinates of the template ligand, used to report on the binding site.
    """
    if not 0.0 <= ph <= 14.0:
        raise ProteinPrepError("The pH must be between 0 and 14.")
    body = "".join(l + "\n" for l in pdb_text.splitlines() if l.startswith(("ATOM", "HETATM", "TER")))
    if "ATOM" not in body:
        raise ProteinPrepError("There is no protein in this file to protonate.")
    body += "END\n"
    before = read_residues(body)
    out_pdb, log = (runner or run_pdb2pqr)(body, ph)
    after = read_residues(out_pdb)

    if len(after) != len(before):
        raise ProteinPrepError(f"PDB2PQR changed the number of residues ({len(before)} -> {len(after)}); not using its output.")
    lost = []
    before_by_key = {r.key: r for r in before}
    after_by_key = {r.key: r for r in after}
    for r in before:
        new = after_by_key.get(r.key)
        if new is None:
            lost.append(r.label)
            continue
        gone = [n for n in r.atoms if not is_hydrogen(n) and n not in new.atoms]
        if gone and not r.hetero:
            lost.append(f"{r.label} ({', '.join(gone)})")
    if lost:
        raise ProteinPrepError("PDB2PQR dropped atoms (" + "; ".join(lost[:3]) + "); not using its output.")
    hydrogens = sum(1 for r in after for n in r.atoms if is_hydrogen(n))
    if hydrogens == 0:
        raise ProteinPrepError("PDB2PQR produced no hydrogens.")

    repaired, problems = parse_warnings(log)
    flipped = []
    for a in after:
        b = before_by_key.get(a.key)
        if b is not None and not b.hetero and any(
                not is_hydrogen(n) and n in a.atoms and np.linalg.norm(p - a.atoms[n]) > FLIP_SHIFT
                for n, p in b.atoms.items()):
            flipped.append(b.label)
    changed = [(f"{b.base} {b.chain}{b.num}{b.icode.strip()}", a.name, b.name)
               for a, b in zip(before, after) if a.name != b.name and not a.hetero]
    return ProteinPrep(
        pdb=out_pdb if out_pdb.rstrip().endswith("END") else out_pdb.rstrip() + "\nEND\n",
        ph=ph, changed=changed, flipped=flipped,
        pocket=_titratable_pocket(after, parse_propka(log), ph, template_xyz),
        gaps=find_gaps(before, template_xyz), incomplete=find_incomplete(before, template_xyz),
        repaired=repaired, problems=problems,
        kept_without_hydrogens=sorted({r.name for r in after if r.hetero and len(r.atoms) > 1
                                       and not any(is_hydrogen(n) for n in r.atoms)}),
        hydrogens=hydrogens, log=log)


def quality_lines(residues, template_xyz=None):
    """Warnings about gaps and incomplete residues, worded for the wizard. Returns (all lines, near-pocket lines)."""
    gaps = find_gaps(residues, template_xyz)
    incomplete = find_incomplete(residues, template_xyz)
    notes, urgent = [], []
    for g in gaps:
        line = (f"Chain gap between {g.before} and {g.after}: residues are missing in the structure "
                f"(C-N distance {g.bond_length} A).")
        if g.distance is not None and g.distance <= GAP_NEAR:
            urgent.append(line + f" The gap is {g.distance:.1f} A from the template, so the loose ends may touch the "
                          "binding site.")
        else:
            notes.append(line + (f" It is {g.distance:.1f} A from the template, so it should not matter for docking."
                                 if g.distance is not None else ""))
    pocket_incomplete = [i for i in incomplete if i.distance is not None and i.distance <= POCKET_DISTANCE]
    far = len(incomplete) - len(pocket_incomplete)
    if far:
        notes.append(f"{far} residue(s) away from the template lack heavy atoms in the structure.")
    for i in pocket_incomplete:
        urgent.append(f"Residue {i.label} next to the template lacks {', '.join(i.missing)}; the pocket shape may "
                      "be wrong there (protonation rebuilds missing atoms from standard geometry, "
                      "which is only a guess).")
    return notes, urgent
