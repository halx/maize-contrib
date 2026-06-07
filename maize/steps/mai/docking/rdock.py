"""Docking with rDock"""

from pathlib import Path
from typing import Literal

from rdkit.Chem import rdMolAlign

from maize.core.node import Node
from maize.core.interface import Parameter, Flag, FileParameter, Suffix, Input, Output
from maize.utilities.chem import (
    Isomer,
    IsomerCollection,
    load_sdf_library,
    save_sdf_library,
)
from maize.utilities.validation import FileValidator

# All modes are encoded in their respective .prm files which are found
# through setting environment variable RBT_ROOT
MODES = Literal[
    "dock",
    "dock_solv",
    "minimise",
    "minimise_solv",
    "score",
    "score_solv",
]
INPUT_FILENAME = "input_mols.sdf"
OUTPUT_PREFIX = "rDock_output"  # will create SD file with .sd extensions


class rDock(Node):
    """ """

    tags = {"chemistry", "docking", "scorer", "tagger", "ensemble"}

    SCORE_TAG = "SCORE.INTER"
    SCORE_TAG_AGG = "min"

    required_callables = ["rdock"]

    # Inputs / Outputs

    inp: Input[list[IsomerCollection]] = Input()
    """List of molecules to dock"""

    out: Output[list[IsomerCollection]] = Output()
    """Docked molecules with conformations and scores attached"""

    mode: Parameter[Literal[MODES]] = Parameter(default="dock")
    """Docking, scoring, minimization"""

    sys_prm: Parameter[str] = Parameter(default=None)
    """System parameters"""

    num_runs: Parameter[int] = Parameter(default=10)
    """Number of docking runs"""

    tethered: Parameter[bool] = Parameter(default=False)
    """Whether tethered (scaffold/template) docking is requested.
    Requires pre-alignment of molecules to reference."""

    tethered_ref_mol: Input[Isomer | str] = Input(optional=True)
    """Reference molecule for pre-alignment of molecules to dock."""

    def run(self) -> None:
        mols = self.inp.receive()

        inputs = Path(INPUT_FILENAME)
        save_sdf_library(inputs, mols, split_strategy="none", conformers=True)

        # align molecules to reference
        if self.tethered.value:
            ref_mol = self.tethered_ref_mol.value
            pre_align_to_ref(mols, ref_mol)

        command = (
            f"{self.runnable['rdock']} "
            f"-i {INPUT_FILENAME} "
            f"-o {OUTPUT_PREFIX} "
            f"-r {self.sys_prm} "
            f"-p {self.mode}.rpm "
            f"-n {self.num_runs}"
        )

        res = self.run_command(
            command,
            verbose=True,
            raise_on_failure=False,
        )

        mols = load_sdf_library(
            Path(OUTPUT_PREFIX + ".sd"),
            split_strategy="schrodinger",
            sanitize=False,
            renumber=False,
        )

        self.out.send(mols)


def pre_align_to_ref(mols, ref_mol):
    ref_match = [i for i in range(ref_mol.GetNumAtoms())]

    for mol in mols:
        for iso in mol.molecules:
            iso_mol = iso._molecule
            mol_match = iso_mol.GetSubstructMatch(ref_mol)

            if not ref_match or not mol_match:
                raise ValueError("SMARTS not found")

            atom_map = list(zip(mol_match, ref_match))

            _ = rdMolAlign.AlignMol(mol, ref_mol, atomMap=atom_map)

            tethered_vals = [atom_idx + 1 for atom_idx in mol_match]
            mol.SetProp("TETHERED ATOMS", ",".join(map(str, tethered_vals)))

    return rmsd
