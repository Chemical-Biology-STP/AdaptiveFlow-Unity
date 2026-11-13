"""Shared fixtures for the tests.

Most tests need nothing but Python. The ones marked `slow` run the real docking programs on a small receptor and take
about 7 minutes in all, measured at 6m37s; `pixi run test` leaves them out and `pixi run test-all` includes them.
"""
import os
import shutil
import sys
import warnings

import pytest

warnings.filterwarnings("ignore")

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_DIR, "test", "data")
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: runs the real docking programs (about 7 minutes in all, measured at 6m37s)")


@pytest.fixture(scope="session")
def pocket():
    """A small receptor (110 residues around the BACE1 ligand), with hydrogens."""
    return os.path.join(DATA_DIR, "pocket.pdb")


@pytest.fixture(scope="session")
def pocket_noh():
    """The same receptor without hydrogens."""
    return os.path.join(DATA_DIR, "pocket_noh.pdb")


@pytest.fixture(scope="session")
def ligand_sdf():
    """The crystal ligand of the pocket: a 3D molecule sitting in the binding site."""
    return os.path.join(DATA_DIR, "crystal_ligand.sdf")


@pytest.fixture(scope="session")
def box_centre():
    """Centre of the crystal ligand, which is where the test dockings put their box."""
    import numpy as np
    from rdkit import Chem

    mol = Chem.SDMolSupplier(os.path.join(DATA_DIR, "crystal_ligand.sdf"))[0]
    xyz = np.array([p for p, a in zip(mol.GetConformer().GetPositions(), mol.GetAtoms()) if a.GetAtomicNum() > 1])
    return [round(float(c), 3) for c in xyz.mean(axis=0)]


@pytest.fixture
def library(tmp_path):
    """A SMILES library with a header, ordinary molecules, a salt, an unsupported element and an unreadable row."""
    path = tmp_path / "library.smi"
    path.write_text(
        "smiles name\n"
        "CC(=O)Nc1ccc(O)cc1 paracetamol\n"
        "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O ibuprofen_S\n"
        "c1ccccc1C(=O)[O-].[Na+] benzoate_salt\n"
        "OB(O)c1ccccc1 boronic_acid\n"
        "CCC1=[O+][Cu]2([O+]=C(CC)C1)[O+]=C(CC)CC(CC)=[O+]2 copper_complex\n"
        "not_a_smiles badrow\n")
    return str(path)


import vs_slurm as _vs_slurm

REAL_CHECK_SUBMISSION = _vs_slurm.check_submission      # the tests of the check itself use this one


@pytest.fixture(autouse=True)
def scheduler_allows_everything(monkeypatch):
    """The assistant asks Slurm whether the account may use a partition; tests must not depend on who runs them or
    on the cluster being reachable. Tests of that check call the original function they import themselves."""
    import vs_slurm

    monkeypatch.setattr(vs_slurm, "check_submission", lambda *args, **kwargs: vs_slurm.SubmitCheck(True))


@pytest.fixture
def adfr_available():
    """Skip a test when the ADFRsuite environment (pixi install --all) is not there."""
    import vs_dock

    if not os.access(os.path.join(vs_dock.DEFAULT_ADFR_PATH, "bin", "prepare_receptor"), os.X_OK):
        pytest.skip("ADFRsuite is not installed (pixi install --all)")


class Answers:
    """Feeds scripted answers to the assistant's prompts, and fails loudly if it asks more than expected."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.asked = []

    def __call__(self, prompt=""):
        self.asked.append(prompt)
        if not self.answers:
            raise AssertionError(f"the assistant asked an unexpected question: {prompt!r}")
        return self.answers.pop(0)


@pytest.fixture
def answers(monkeypatch):
    """answers([...]) makes input() return those strings in order; an empty string accepts the default."""
    def install(values):
        feed = Answers(values)
        monkeypatch.setattr("builtins.input", feed)
        return feed

    return install
