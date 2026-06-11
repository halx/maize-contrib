"""Docking with rDock"""

from pathlib import Path
from typing import Literal, Callable

from rdkit import Chem
from rdkit.Chem.rdMolAlign import AlignMol
from rdkit.Chem.rdForceFieldHelpers import UFFGetMoleculeForceField

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

    # SCORE_TAGS: tuple[str, ...] = ("SCORE.INTER", )
    # SCORE_TAGS_AGG: tuple[Literal["min", "max"], ...] = ("min",)
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
        mols = hydrogens_last(mols)
        smilies = {mol.name: mol.smiles for mol in mols}
        #charges = get_formal_charges(mols)

        inputs = Path(INPUT_FILENAME)

        # align molecules to reference
        if self.tethered.value:
            ref_mol = self.tethered_ref_mol.receive_optional()
            align_to_reference(mols, ref_mol, self.logger)

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


def hydrogens_last(mols: IsomerCollection):
    """Reorder atoms such that hydrogens come last"""

    for mol in mols:
        for iso in mol.molecules:
            iso_mol = iso._molecule
            heavy = [a.GetIdx() for a in iso_mol.GetAtoms() if a.GetAtomicNum() > 1]
            hydrogens = [a.GetIdx() for a in iso_mol.GetAtoms() if a.GetAtomicNum() == 1]

            new_order = heavy + hydrogens

            iso_mol = Chem.RenumberAtoms(iso_mol, new_order)

    return mols


def get_formal_charges(mols: IsomerCollection) -> dict[float]:
    charges = {}

    # FIXME: formal charges on atoms!
    for mol in mols:
        for iso in mol.molecules:
            charges[mol.name] = iso._molecule.GetFormalCharge()

    return charges


def align_to_reference(mols: IsomerCollection, ref_isomer: Isomer, logger) -> None:
    """Align the molecules to the reference

    Updates molecules with new coordinates.

    :param mols: molecules to align
    :ref_isomer: refernce to align to
    """

    ref_mol = Chem.RemoveHs(ref_isomer._molecule)

    for mol in mols:
        for iso in mol.molecules:
            iso_mol = iso._molecule
            Chem.FastFindRings(iso_mol)  # unclear why this is needed
            mol_match = iso_mol.GetSubstructMatch(ref_mol)

            if not mol_match:
                raise ValueError(
                    f"SMARTS not found: {Chem.MolToSmiles(iso_mol)} {Chem.MolToSmiles(ref_mol)}"
                )

            # FIXME: use align followed by constraint minimization?
            try:
                iso_mol = constraindt_align(iso_mol, ref_mol, mol_match)
            except:
                logger.debug(f"{Chem.MolToSmiles(iso_mol)} failed to embed")
                continue

            # parsed in lib/RbtModel.cxx: Each line is comma-separated list of atom IDs
            tethered_vals = [atom_idx + 1 for atom_idx in mol_match]
            iso.set_tag("TETHERED ATOMS", ",".join(map(str, tethered_vals)))


def constraindt_align(
    mol: Chem.Mol,
    ref: Chem.Mol,
    match: list[int],
    get_forcefield: Callable = UFFGetMoleculeForceField,
):
    """Constraint alignment of a 3D molecule to a core

    Essentially the ConstraintEmebed code with the embedding because it mayy
    have a high failure rate and we expect the molecule to be 3D already
    anyway.  We assume mol has only one conformer.

    :param mol: the molecule to align
    :param ref: the reference to match to
    :param match: the matching indices between mol and core
    :param get_forcefield: optional forcefield getter
    :returns: molecule with new coordinates
    """

    align_mao = [(j, i) for i, j in enumerate(match)]

    AlignMol(mol, ref, atomMap=align_mao)
    forcefield = get_forcefield(mol, confId=0)

    conf = ref.GetConformer()

    for i in range(ref.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        idx = forcefield.AddExtraPoint(pos.x, pos.y, pos.z, fixed=True) - 1
        forcefield.AddDistanceConstraint(idx, match[i], 0, 0, 100.0)

    forcefield.Initialize()

    max_steps = 4
    success = 1

    while success and max_steps:
        success = forcefield.Minimize(energyTol=1e-4, forceTol=1e-3)
        max_steps -= 1

    AlignMol(mol, ref, atomMap=align_mao)

    return mol
