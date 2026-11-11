"""Turn a protein-ligand complex (PDB or mmCIF) into a clean receptor and a template ligand.

The functions here only analyse and transform; asking the user is dock_cli.py's job.

A deposited complex holds more than the docking needs: the template ligand itself, waters,
crystallisation additives, ions, cofactors, sugars, other copies of the protein. ADFR's
prepare_receptor removes waters but keeps everything else, so a template left in the file
sits inside the receptor and blocks the pocket it is meant to define.
"""

import re
import urllib.request
from dataclasses import dataclass, field

import gemmi
import numpy as np
from rdkit import Chem, RDLogger
from scipy.spatial import cKDTree

RDLogger.DisableLog("rdApp.*")

STRUCTURE_EXTENSIONS = (".pdb", ".ent", ".cif", ".mmcif")

# Molecules added to get or protect the crystal. Removed without asking, but reported.
ADDITIVES = frozenset({
    "SO4", "PO4", "GOL", "EDO", "PEG", "PGE", "PG4", "PGO", "1PE", "P6G", "P33", "PG5", "PG6", "2PE", "PE4",
    "MPD", "MRD", "MRY", "DMS", "ACT", "ACY", "FMT", "TRS", "EPE", "MES", "BME", "IPA", "EOH", "MOH", "BU3",
    "TAR", "CIT", "FLC", "NO3", "SCN", "AZI", "IMD", "TLA", "BOG", "LDA", "DTT", "DTU", "CXS", "HEZ", "OCT",
    "NH4", "IOD", "BR", "SO3", "PI", "2HP", "PO3", "MLI", "MAL", "SUC", "GLC", "CAC", "CO3", "PGR", "15P",
})
# Cofactors and prosthetic groups: real, often relevant, but rarely what the user wants as template.
COFACTORS = frozenset({
    "HEM", "HEC", "HEA", "HEB", "NAD", "NAP", "NDP", "NAI", "FAD", "FMN", "ATP", "ADP", "AMP", "ANP", "GTP",
    "GDP", "GNP", "COA", "ACO", "SAM", "SAH", "PLP", "PMP", "TPP", "BTN", "FES", "SF4", "F3S", "CLA", "BCL",
    "RET", "B12", "COB", "MGD", "H4B", "THF", "FOL", "UDP", "UTP", "CDP", "CTP", "NMN", "PQQ", "TDP",
})

MIN_TEMPLATE_HEAVY = 5      # smaller molecules are not offered as templates
CONTACT_DISTANCE = 4.0      # protein atoms this close count as contacts of a ligand
CHAIN_NEAR = 8.0            # protein chains with an atom this close to the template are kept by default
ASK_RADIUS = 12.0           # molecules farther from the template cannot matter for the 20 A box: not asked about
KEEP_DEFAULT_RADIUS = 5.0   # within this distance of the template a molecule is suggested as "keep"
GROUP_ABOVE = 12            # with more questions than this, molecules of the same kind are asked about together

KIND_ORDER = {"ligand": 0, "cofactor": 1, "additive": 2, "ion": 3, "sugar": 4}


class PrepError(Exception):
    """The structure cannot be prepared; the message says why in plain words."""


@dataclass
class Het:
    chain: str
    name: str
    num: int
    icode: str
    heavy: int
    kind: str           # ligand, cofactor, additive, ion or sugar
    xyz: object         # heavy-atom coordinates, shape (n, 3)
    contacts: int = 0   # protein atoms within CONTACT_DISTANCE
    distance: float = None  # to the chosen template, once there is one
    heavy_formula: str = ""  # element counts of the heavy atoms in the file, e.g. C22N3O4 (hydrogens are rarely present)

    @property
    def label(self):
        return f"{self.name} {self.chain}{self.num}{self.icode.strip()}"

    @property
    def number(self):
        """The residue number as the file writes it, with its insertion code if it has one (e.g. 52A)."""
        return f"{self.num}{self.icode.strip()}"

    @property
    def key(self):
        return (self.chain, self.num, self.icode, self.name)


@dataclass
class Inventory:
    path: str
    n_models: int
    protein_chains: dict        # chain name -> number of residues
    nucleic_chains: list
    waters: int
    hets: list
    hydrogens: int              # hydrogen atoms in the protein chains
    altloc_atoms: int           # atoms with alternate locations in the file
    modified: list              # modified residue names inside the protein chains
    protein_xyz: object         # heavy atoms of all protein chains
    protein_xyz_by_chain: dict = field(default_factory=dict)


@dataclass
class Prepared:
    receptor_pdb: str
    template_molblock: str
    template: Het
    kept_chains: list
    summary: list               # one line per thing that was removed or kept, for the user and for a record
    template_view_molblock: str = None   # the template with hydrogens, for display only; None if unavailable
    template_view_note: str = ""         # why there are no hydrogens on it, when there are none


def _is_water(res):
    return res.entity_type == gemmi.EntityType.Water or res.is_water()


def _is_polymer(res):
    if res.entity_type == gemmi.EntityType.Polymer:
        return True
    return res.entity_type == gemmi.EntityType.Unknown and res.het_flag == "A"


def _heavy_formula(res):
    """Element counts of the non-hydrogen atoms of a residue in the usual order (C first, then alphabetical)."""
    counts = {}
    for atom in res:
        if not atom.is_hydrogen():
            counts[atom.element.name] = counts.get(atom.element.name, 0) + 1
    order = sorted(counts, key=lambda e: (e != "C", e))
    return "".join(f"{e}{counts[e] if counts[e] > 1 else ''}" for e in order)


def _heavy_xyz(res):
    return np.array([[a.pos.x, a.pos.y, a.pos.z] for a in res if not a.is_hydrogen()], dtype=float).reshape(-1, 3)


def _kind(res, heavy):
    if res.name in ADDITIVES:
        return "additive"
    if res.entity_type == gemmi.EntityType.Branched:
        return "sugar"
    if heavy <= 1:
        return "ion"
    if res.name in COFACTORS:
        return "cofactor"
    return "ligand"


def load_structure(path):
    """Read a PDB or mmCIF file. Returns (structure, number of models, atoms with alternate locations)."""
    try:
        st = gemmi.read_structure(str(path))
    except Exception as exc:  # gemmi raises RuntimeError and ValueError for bad files
        raise PrepError(f"Could not read this file as a PDB or mmCIF structure ({exc}).")
    if len(st) == 0 or sum(len(r) for c in st[0] for r in c) == 0:
        raise PrepError("The file holds no atoms.")
    st.setup_entities()
    altloc = sum(1 for chain in st[0] for res in chain for atom in res if atom.altloc != "\0")
    st.remove_alternative_conformations()
    return st, len(st), altloc


def inventory(st, n_models=1, altloc_atoms=0, path=""):
    protein, nucleic, hets = {}, [], []
    waters = hydrogens = 0
    modified = set()
    protein_atoms, by_chain = [], {}
    for chain in st[0]:
        amino = nuc = 0
        chain_xyz = []
        for res in chain:
            if _is_water(res):
                waters += 1
            elif _is_polymer(res):
                table = gemmi.find_tabulated_residue(res.name)
                if table is not None and table.is_nucleic_acid():
                    nuc += 1
                    continue
                amino += 1
                if res.het_flag == "H":
                    modified.add(res.name)
                hydrogens += sum(1 for a in res if a.is_hydrogen())
                chain_xyz.append(_heavy_xyz(res))
            else:
                xyz = _heavy_xyz(res)
                hets.append(Het(chain.name, res.name, res.seqid.num, (res.seqid.icode or " "), len(xyz),
                                _kind(res, len(xyz)), xyz, heavy_formula=_heavy_formula(res)))
        if amino and amino >= nuc:
            protein[chain.name] = amino
            by_chain[chain.name] = np.vstack(chain_xyz)
            protein_atoms.append(by_chain[chain.name])
        elif nuc:
            nucleic.append(chain.name)
    if not protein:
        raise PrepError("No protein chain was found in this structure.")
    protein_xyz = np.vstack(protein_atoms)
    tree = cKDTree(protein_xyz)
    for het in hets:
        near = set()
        for hits in tree.query_ball_point(het.xyz, CONTACT_DISTANCE):
            near.update(hits)
        het.contacts = len(near)
    return Inventory(str(path), n_models, protein, nucleic, waters, hets, hydrogens, altloc_atoms,
                     sorted(modified), protein_xyz, by_chain)


def template_candidates(inv):
    """Molecules that could be the template, best guess first.

    Real ligands rank above cofactors, additives and so on; within a kind, more contacts with the
    protein ranks higher (a bound inhibitor touches far more protein than a stray molecule).
    """
    pool = [h for h in inv.hets if h.heavy >= MIN_TEMPLATE_HEAVY and h.kind != "sugar"]
    return sorted(pool, key=lambda h: (KIND_ORDER[h.kind], -h.contacts, -h.heavy))


def suggested_template(inv):
    for het in template_candidates(inv):
        if het.kind == "ligand":
            return het
    return None


def measure_distances(inv, template):
    """Set each molecule's distance to the template; return the same for every protein chain."""
    tree = cKDTree(template.xyz)
    for het in inv.hets:
        het.distance = float(tree.query(het.xyz)[0].min()) if len(het.xyz) else None
    return {name: float(tree.query(xyz)[0].min()) for name, xyz in inv.protein_xyz_by_chain.items()}


def default_chains(chain_distances):
    """Protein chains that touch the template's neighbourhood; the nearest one if none does."""
    near = [name for name, d in chain_distances.items() if d <= CHAIN_NEAR]
    return near or [min(chain_distances, key=chain_distances.get)]


@dataclass
class Plan:
    """What to do with the molecules other than the protein and the template."""
    ask: list = field(default_factory=list)       # in a kept chain and close enough to matter
    removed: list = field(default_factory=list)   # (Het, reason) removed without asking


def plan_molecules(inv, template, keep_chains):
    plan = Plan()
    for het in inv.hets:
        if het.key == template.key:
            continue
        if het.chain not in keep_chains:
            plan.removed.append((het, "in a protein chain that is not kept"))
        elif het.kind == "sugar":
            plan.removed.append((het, "sugar chain"))
        elif het.kind == "additive":
            plan.removed.append((het, "crystallisation additive"))
        elif het.name == template.name:
            plan.removed.append((het, "another copy of the template ligand"))
        elif het.distance is not None and het.distance > ASK_RADIUS:
            plan.removed.append((het, f"more than {ASK_RADIUS:g} A from the template"))
        else:
            plan.ask.append(het)
    return plan


def default_keep(het):
    return het.distance is not None and het.distance <= KEEP_DEFAULT_RADIUS


def _find_residue(st, key):
    chain_name, num, icode, name = key
    for chain in st[0]:
        if chain.name != chain_name:
            continue
        for res in chain:
            if res.name == name and res.seqid.num == num and (res.seqid.icode or " ") == icode:
                return res
    raise PrepError(f"Could not find {name} {chain_name}{num} again in the structure.")


def fetch_ccd(name, timeout=15):
    """The PDB's chemical component dictionary entry for a residue name, or None if unavailable.

    It describes the ligand's real chemistry (bond orders, charges, hydrogens), which a structure file
    does not. The name comes from an untrusted file, so only plain residue names are looked up.
    """
    if not re.fullmatch(r"[A-Za-z0-9]{1,5}", name or ""):
        return None
    try:
        with urllib.request.urlopen(f"https://files.rcsb.org/ligands/download/{name.upper()}.cif",
                                    timeout=timeout) as response:
            text = response.read().decode("utf-8", errors="replace")
        return gemmi.cif.read_string(text)[0]
    except Exception:  # offline, unknown component, malformed reply: all mean "not available"
        return None


@dataclass
class Component:
    """What the PDB's chemical component dictionary says about a residue name, in terms a chemist uses."""
    code: str
    name: str = ""        # a recognisable name (Erlotinib, Heme, Glycerol); empty if unknown
    systematic: str = ""  # the dictionary's full chemical name, as written there
    formula: str = ""     # complete formula including hydrogens (C22H23N3O4); empty if unknown
    smiles: str = ""


NAME_MAX = 30   # longer systematic names are replaced by a common synonym when there is one


def _cif_text(block, tag):
    value = block.find_value(tag)
    if value is None:
        return ""
    text = gemmi.cif.as_string(value).strip()
    return "" if text in ("?", ".") else text


def friendly_name(code, systematic, synonyms):
    """A name a chemist recognises. Systematic names of drugs are very long ('[6,7-BIS(2-METHOXY-ETHOXY)QUINAZOLINE-4-YL]
    -(3-ETHYNYLPHENYL)AMINE'), so a short one is kept and a long one is replaced by the first common synonym
    (preferring one without digits, which skips lab codes such as STI-571 in favour of IMATINIB)."""
    names = [n.strip() for n in re.split(r"[;\n]", synonyms or "") if n.strip()]
    if systematic and len(systematic) <= NAME_MAX:
        chosen = systematic
    elif names:
        chosen = next((n for n in names if not re.search(r"\d", n)), names[0])
    else:
        chosen = systematic
    if not chosen:
        return ""
    chosen = re.sub(r"^\[\d+[RSrs]-\([^)]*\)\]-", "", chosen)   # a leading stereo-descriptor, e.g. [4R-(4ALPHA,5ALPHA)]-
    if chosen.isupper():                       # the dictionary writes many names in capitals
        chosen = chosen[0] + chosen[1:].lower()
    return chosen if len(chosen) <= NAME_MAX + 4 else chosen[:NAME_MAX + 1].rstrip() + "..."


def component_from_block(code, block):
    """A Component from a dictionary entry (a gemmi cif block), or an empty one if there is none."""
    if block is None:
        return Component(code)
    smiles = ""
    try:
        descriptors = {}
        for kind, text in block.find("_pdbx_chem_comp_descriptor.", ["type", "descriptor"]):
            descriptors.setdefault(gemmi.cif.as_string(kind), gemmi.cif.as_string(text))
        smiles = descriptors.get("SMILES_CANONICAL") or descriptors.get("SMILES") or ""
    except Exception:  # noqa: BLE001 - a missing or odd table only means no SMILES
        pass
    systematic = _cif_text(block, "_chem_comp.name")
    return Component(code,
                     name=friendly_name(code, systematic, _cif_text(block, "_chem_comp.pdbx_synonyms")),
                     systematic=systematic, formula=_cif_text(block, "_chem_comp.formula").replace(" ", ""), smiles=smiles)


LOOKUP_SECONDS = 6      # per molecule; offline, the answer should come quickly rather than stall the wizard


class Components:
    """Dictionary entries for the molecules of one structure, fetched once per residue name.

    fetch(name) has the signature of fetch_ccd, so it can also be handed to prepare() and the template's
    entry is not downloaded a second time. The download function is looked up when it is needed.
    """

    def __init__(self, fetch=None):
        self._fetch = fetch
        self.blocks = {}

    def fetch(self, name, timeout=15):
        key = (name or "").upper()
        if key not in self.blocks:
            self.blocks[key] = (self._fetch or fetch_ccd)(name)
        return self.blocks[key]

    def prefetch(self, names):
        """Fetch several entries at once. Returns how many were found."""
        from concurrent.futures import ThreadPoolExecutor

        todo = sorted({(n or "").upper() for n in names} - set(self.blocks))
        if todo:
            fetch = self._fetch or fetch_ccd
            with ThreadPoolExecutor(max_workers=min(8, len(todo))) as pool:
                for name, block in zip(todo, pool.map(lambda n: fetch(n, LOOKUP_SECONDS), todo)):
                    self.blocks[name] = block
        return sum(1 for n in names if self.blocks.get((n or "").upper()) is not None)

    def describe(self, name):
        return component_from_block(name, self.blocks.get((name or "").upper()))


def template_mol_from_residue(st, template):
    """The template residue as an RDKit molecule with PDB atom names. Bonds come from distances (all single)."""
    res = _find_residue(st, template.key)
    single = gemmi.Structure()
    model = gemmi.Model("1")
    chain = gemmi.Chain("A")
    chain.add_residue(res)
    model.add_chain(chain)
    single.add_model(model)
    mol = Chem.MolFromPDBBlock(single.make_pdb_string(), sanitize=False, removeHs=False, proximityBonding=True)
    if mol is None or mol.GetNumAtoms() == 0:
        raise PrepError(f"Could not read the template {template.label} as a molecule.")
    return mol


_BOND_ORDERS = {"SING": Chem.BondType.SINGLE, "DOUB": Chem.BondType.DOUBLE, "TRIP": Chem.BondType.TRIPLE}


def template_with_hydrogens(mol, ccd_block):
    """The template with its real bond orders, charges and all hydrogens, from the component dictionary.

    Returns (molecule, "") or (None, reason). The result is for display only: the docking itself keeps
    using the heavy-atom template. Atoms are matched by PDB atom name, and the result is only trusted if
    every heavy atom matches and the hydrogen count equals the dictionary's.
    """
    atoms = ccd_block.find("_chem_comp_atom.", ["atom_id", "type_symbol", "charge"])
    bonds = ccd_block.find("_chem_comp_bond.", ["atom_id_1", "atom_id_2", "value_order"])
    if not atoms or not bonds:
        return None, "the component dictionary entry has no atom or bond table"
    as_str = gemmi.cif.as_string
    heavy, expected_h = {}, 0
    for row in atoms:
        symbol = as_str(row[1]).upper()
        if symbol in ("H", "D"):
            expected_h += 1
            continue
        try:
            charge = int(as_str(row[2]))
        except ValueError:
            charge = 0
        heavy[as_str(row[0])] = charge
    names = {}
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None or atom.GetAtomicNum() == 1:
            continue
        names[info.GetName().strip()] = atom.GetIdx()
    if set(names) != set(heavy):
        missing = sorted(set(heavy) - set(names))
        extra = sorted(set(names) - set(heavy))
        detail = (f"atoms missing from the structure: {', '.join(missing[:5])}" if missing
                  else f"atoms not in the dictionary: {', '.join(extra[:5])}")
        return None, f"the ligand in the structure does not match the component dictionary ({detail})"
    rw = Chem.RWMol(mol)
    for bond in list(rw.GetBonds()):
        rw.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    for row in bonds:
        a, b, order = as_str(row[0]), as_str(row[1]), as_str(row[2]).upper()
        if a in names and b in names:
            if order not in _BOND_ORDERS:
                return None, f"unexpected bond order {order} in the component dictionary"
            rw.AddBond(names[a], names[b], _BOND_ORDERS[order])
    for name, charge in heavy.items():
        atom = rw.GetAtomWithIdx(names[name])
        atom.SetFormalCharge(charge)
        atom.SetNoImplicit(False)
        atom.SetNumExplicitHs(0)
    try:
        new = rw.GetMol()
        Chem.SanitizeMol(new)
        Chem.AssignStereochemistryFrom3D(new)
        with_h = Chem.AddHs(new, addCoords=True)
    except Exception as exc:
        return None, f"the dictionary chemistry could not be applied ({type(exc).__name__})"
    n_h = sum(1 for a in with_h.GetAtoms() if a.GetAtomicNum() == 1)
    if n_h != expected_h:
        return None, (f"it would get {n_h} hydrogens but the dictionary lists {expected_h} "
                      "(a different protonation state?)")
    return with_h, ""


def template_molblock(st, template):
    """The template as an SDF record. Bonds come from distances, so they are all single bonds.

    Anchoring needs only the connectivity and the coordinates; bond orders do not change the shared
    core or the anchor positions (checked against a template with real bond orders).
    """
    mol = template_mol_from_residue(st, template)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        Chem.SanitizeMol(mol, Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES ^ Chem.SANITIZE_KEKULIZE
                         ^ Chem.SANITIZE_SETAROMATICITY)
    mol.SetProp("_Name", template.name)
    block = Chem.MolToMolBlock(mol) + "$$$$\n"
    reread = Chem.MolFromMolBlock(block.replace("$$$$\n", ""))  # what the docking script will do
    if reread is None or reread.GetNumConformers() == 0:
        raise PrepError(f"The template {template.label} could not be turned into a molecule the docking can "
                        "read (unusual bonding?). Export it to an SDF file yourself and use the separate-files option.")
    return block


def prepare(st, inv, template, keep_chains, keep_hets, ccd=None):
    """Build the cleaned receptor and the template record, and a summary of what was done.

    ccd: a function name -> component dictionary entry (such as fetch_ccd), used to give the template
    its hydrogens for display. None means no lookup, so no network access.
    """
    keep_keys = {h.key for h in keep_hets}
    out = gemmi.Structure()
    out.name = "receptor"
    model = gemmi.Model("1")
    for chain in st[0]:
        if chain.name not in keep_chains:
            continue
        new = gemmi.Chain(chain.name)
        for res in chain:
            if _is_water(res):
                continue
            if _is_polymer(res):
                new.add_residue(res)
            else:
                key = (chain.name, res.seqid.num, (res.seqid.icode or " "), res.name)
                if key in keep_keys:
                    new.add_residue(res)
        model.add_chain(new)
    out.add_model(model)
    out.shorten_chain_names()
    receptor_pdb = out.make_pdb_string()
    if "ATOM" not in receptor_pdb:
        raise PrepError("Nothing was left of the protein after cleaning.")

    plan = plan_molecules(inv, template, keep_chains)
    summary = [f"Template ligand: {template.label} ({template.heavy} heavy atoms), taken from {inv.path or 'the structure'}"]
    summary.append("Protein chains kept: " + ", ".join(f"{c} ({inv.protein_chains[c]} residues)" for c in keep_chains))
    dropped = [c for c in inv.protein_chains if c not in keep_chains]
    if dropped:
        summary.append("Protein chains removed: " + ", ".join(dropped))
    if inv.nucleic_chains:
        summary.append("Nucleic acid chains removed: " + ", ".join(inv.nucleic_chains))
    if inv.waters:
        summary.append(f"Waters removed: {inv.waters}")
    if inv.n_models > 1:
        summary.append(f"Models: kept the first of {inv.n_models}")
    if inv.altloc_atoms:
        summary.append(f"Alternate locations: kept the first conformation ({inv.altloc_atoms} atoms had alternates)")
    kept_labels = [h.label for h in keep_hets]
    if kept_labels:
        summary.append("Other molecules kept in the receptor: " + ", ".join(kept_labels))
    asked_removed = [h for h in plan.ask if h.key not in keep_keys]
    if asked_removed:
        summary.append("Other molecules you chose to remove: " + ", ".join(h.label for h in asked_removed))
    by_reason = {}
    for het, reason in plan.removed:
        by_reason.setdefault(reason, []).append(het.label)
    for reason, labels in by_reason.items():
        summary.append(f"Removed ({reason}): " + ", ".join(labels))
    if inv.modified:
        summary.append("Modified residues inside the protein, kept as they are: " + ", ".join(inv.modified))

    view_block, view_note = None, "hydrogens were not looked up"
    if ccd is not None:
        block = ccd(template.name)
        if block is None:
            view_note = (f"the chemistry of {template.name} could not be fetched from the PDB's component "
                         "dictionary (offline, or not a registered component)")
        else:
            mol_h, why = template_with_hydrogens(template_mol_from_residue(st, template), block)
            if mol_h is None:
                view_note = why
            else:
                mol_h.SetProp("_Name", template.name)
                view_block = Chem.MolToMolBlock(mol_h) + "$$$$\n"
                view_note = ""
    if view_block:
        summary.append("Template hydrogens for display: added from the component dictionary entry for "
                       f"{template.name} (real bond orders; the docking itself uses the heavy atoms only)")
    elif ccd is not None:
        summary.append(f"Template hydrogens for display: not added, {view_note}")
    return Prepared(receptor_pdb, template_molblock(st, template), template, list(keep_chains), summary,
                    view_block, view_note)
