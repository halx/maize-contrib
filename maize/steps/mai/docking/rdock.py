"""Docking with rDock"""

from pathlib import Path
from typing import Literal

from rdkit.Chem import AllChem

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

    #SCORE_TAGS: tuple[str, ...] = ("SCORE.INTER", )
    #SCORE_TAGS_AGG: tuple[Literal["min", "max"], ...] = ("min",)
    SCORE_TAGS = "SCORE.INTER"
    SCORE_TAGS_AGG = "min"
    PRIMARY_SCORE_TAG = "SCORE.INTER"

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

    tethered_ref_mol: Input[Isomer] = Input(optional=True)
    """Reference molecule for pre-alignment of molecules to dock."""

    def run(self) -> None:
        mols = self.inp.receive()
        smilies = {mol.name: mol.smiles for mol in mols}

        inputs = Path(INPUT_FILENAME)

        # align molecules to reference
        if self.tethered.value:
            ref_mol = self.tethered_ref_mol.receive_optional()
            align_to_reference(mols, ref_mol)

        save_sdf_library(inputs, mols, split_strategy="none", conformers=True)

        command = (
            f"{self.runnable['rdock']} "
            f"-i {INPUT_FILENAME} "
            f"-o {OUTPUT_PREFIX} "
            f"-r {self.sys_prm.value} "
            f"-p {self.mode.value}.prm "
            f"-n {self.num_runs.value}"
        )

        self.logger.debug(f"Running rDock as: {command}")
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

        self._add_scores(mols)

        for mol in mols:
            for name, smiles in smilies.items():
                if mol.name == name:
                    mol.smiles = smiles
                    break

        self.out.send(mols)

    def _add_scores(self, mols):
        for mol in mols:
            for iso in mol.molecules:
                for conf in iso.conformers:
                    conf.add_score_tag(self.SCORE_TAGS, agg=self.SCORE_TAGS_AGG)

                iso.primary_score_tag = self.PRIMARY_SCORE_TAG
                iso.set_tag("score_type", "oracle")
                iso.set_tag("origin", self.name)


def align_to_reference(mols, ref_isomer):

    ref_mol = AllChem.RemoveHs(ref_isomer._molecule)

    for mol in mols:
        for iso in mol.molecules:
            iso_mol = iso._molecule
            AllChem.FastFindRings(iso_mol)  # unclear why this is needed
            mol_match = iso_mol.GetSubstructMatch(ref_mol)

            if not mol_match:
                raise ValueError(f"SMARTS not found: {AllChem.MolToSmiles(iso_mol)} {AllChem.MolToSmiles(ref_mol)}")

            iso_mol = AllChem.ConstrainedEmbed(iso_mol, ref_mol, useTethers=True)

            # parsed in lib/RbtModel.cxx: Each line is comma-separated list of atom IDs
            tethered_vals = [atom_idx + 1 for atom_idx in mol_match]
            iso.set_tag("TETHERED ATOMS", ",".join(map(str, tethered_vals)))
