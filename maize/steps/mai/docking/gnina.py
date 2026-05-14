"""Docking with GNINA"""

from functools import partial, reduce
from pathlib import Path
from typing import Annotated, Literal, cast

import rdkit.Chem.AllChem as Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
import numpy as np
from numpy.typing import NDArray
import pytest

from maize.core.node import Node
from maize.core.interface import Parameter, Flag, FileParameter, Suffix, Input, Output
from maize.utilities.chem import Isomer, IsomerCollection, Conformer
from maize.utilities.chem.chem import find_mol, load_sdf_library, merge_libraries, save_sdf_library
from maize.utilities.testing import TestRig
from maize.utilities.validation import FileValidator
from maize.utilities.io import Config
from maize.utilities.resources import cpu_count
from maize.utilities.execution import GPU

ScoreType = Literal["default", "ad4_scoring", "dkoes_fast", "dkoes_scoring", "vina", "vinardo"]
CNNScoreType = Literal["none", "rescore", "refinement", "metrorescore", "metrorefine", "all"]
PDBFileType = Annotated[Path, Suffix("pdb", "pdbqt")]


def refresh_conformer_wrappers(iso) -> None:
    wrappers = [Conformer(rdconf, parent=iso) for rdconf in iso._molecule.GetConformers()]
    iso._conformers = wrappers


def _split_ligands(mols: IsomerCollection, scaffold: str) -> tuple:
    """Split each mol using the scaffold and compute atom SMARTS

    :param mols: molecules to split
    :param scaffold: scaffold SMARTS
    :returns: fragment mols, atom SMARTS
    """

    # Identify attachment atom and R-group atoms
    patt = Chem.MolFromSmarts(scaffold)
    mols_smarts = []

    for mol in mols:
        match = mol.GetSubstructMatcees(patt)

        if len(match) > 1:
            ...  # FIXME: what to do in case ofg multiple matches

        scaff_idxs = set(match)
        rgroup_idxs = [a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in scaff_idxs]

        attachment_atom = None

        for rg in rgroup_idxs:
            for nbr in mol.GetAtomWithIdx(rg).GetNeighbors():
                if nbr.GetIdx() in scaff_idxs:
                    attachment_atom = rg
                    break

        # Extract the R-group as a separate molecule, preserving attachment atom order
        order = [attachment_atom] + [i for i in rgroup_idxs if i != attachment_atom]
        rgroup = Chem.PathToSubmol(mol, order)

        # Convert the R-group to SMARTS with atom 0 first
        rgroup_smarts = Chem.MolToSmarts(rgroup)
        recursive_smarts = f"[$({rgroup_smarts})]"

        frag_mol = None

        # TODO: extract fragment molecule

        mols_smarts.append((frag_mol, recursive_smarts))

    return mols_smarts


class _GNINA(Node, register=False):
    """GNINA base"""

    SCORE_TAGS = ("minimizedAffinity", "minimizedRMSD", "CNNscore", "CNNaffinity", "CNN_VS")
    SCORE_TAGS_AGG: tuple[Literal["min", "max"], ...] = ("min", "max", "max", "max", "max")
    PRIMARY_SCORE_TAG = "minimizedAffinity"

    required_callables = ["gnina"]

    inp: Input[list[IsomerCollection]] = Input()
    """List of molecules to dock"""

    out: Output[list[IsomerCollection]] = Output()
    """Docked molecules with conformations and scores attached"""

    search_range: Parameter[tuple[float, float, float]] = Parameter(default=(15.0, 15.0, 15.0))
    """Range of the search space for docking"""

    autobox_add: Parameter[float] = Parameter(default=4.0)
    """Amount of buffer space to add around the ligand"""

    scoring: Parameter[ScoreType] = Parameter(default="default")
    """Scoring function to use"""

    cnn_scoring: Parameter[CNNScoreType] = Parameter(default="rescore")
    """CNN scoring method to use"""

    exhaustiveness: Parameter[int] = Parameter(default=8)
    """Exhaustiveness of the global search (roughly proportional to time)"""

    n_poses: Parameter[int] = Parameter(default=8)
    """Maximum number of poses to generate"""

    score_only: Flag = Flag(default=False)
    """
    If ``True``, will only score the provided pose without conformational search.
    With this option, neither a reference nor search_center need to be provided.
    """

    local_only: Flag = Flag(default=False)
    """
    If ``True``, will only do local only optimization and ligand minimization
    """

    minimize: Flag = Flag(default=False)
    """Whether to just minimize the passed-in conformation"""

    cnn_model: FileParameter[list[Annotated[Path, Suffix("pt")]]] = FileParameter(optional=True)
    """One or more alternative CNN scoring models to use"""

    cnn: Parameter[str] = Parameter(optional=True)
    """Name of a pre-trained CNN model or ensemble models to use"""

    covalent_ref: FileParameter[Annotated[Path, Suffix("sdf")]] = FileParameter(optional=True)
    """SDF of the fragment to remove from the molecule"""

    # NOTE: this could also be x,y,z coordinates, not optional if covalent_ref is set
    covalent_ap_fragment: Parameter[str] = Parameter(optional=True)
    """Attachment point of fragment as chain:resnum:atom_name"""

    covalent_kekulize: Flag = Flag(default=True)
    """Seems that in covalent docking structure may not kekulize"""

    local_opt_ref: FileParameter[Annotated[Path, Suffix("sdf")]] = FileParameter(optional=True)
    """Reference structure filename for local optimization: ligands will be aligned to it"""

    n_jobs: Parameter[int] = Parameter(default=cpu_count())
    """The number of CPUs to use per docking run"""

    gpu: Flag = Flag(default=True)
    """Whether to use the GPU for the CNN scoring step"""


class GNINAEnsemble(_GNINA):
    """
    Ensemble docks molecules with GNINA.

    You must specify multiple conformers of the receptor to dock to. Every molecule
    will be docked against all conformations, unless ``score_only`` is set, in which
    case conformer 1 will be scored with molecule 1, 2 with 2, etc.

    See the `repo <https://github.com/gnina/gnina>`_ and [#mcnutt2021]_ for more information.

    References
    ----------
    .. [#mcnutt2021] McNutt, A., Francoeur, P., Aggarwal, R., Masuda, T., Meli, R.,
       Ragoza, M., Sunseri, J. & Koes, D. R. GNINA 1.0: Molecular docking with deep learning.
       J. Cheminformatics 13, 43, (2021).

    """

    tags = {"chemistry", "docking", "scorer", "tagger", "ensemble"}

    inp_ref: Input[list[Isomer]] = Input(optional=True)
    """Reference pose input"""

    ensemble: FileParameter[list[PDBFileType]] = FileParameter(optional=True)
    """Paths to all receptor conformations"""

    ensemble_weights: Parameter[list[float]] = Parameter(optional=True)
    """Optional weights for each ensemble conformation"""

    search_center: Parameter[NDArray[np.float32]] = Parameter(optional=True)
    """Center of the search space for docking"""

    n_parallel: Parameter[int] = Parameter(default=1)
    """
    Number of parallel jobs to run for ensemble docking. Make sure that
    ``n_parallel * n_jobs`` does not exceed the number of available cores,
    or use batch processing.

    """

    def run(self) -> None:
        mols = self.inp.receive()
        protein_confs = self.ensemble.filepath
        weights = np.ones_like(protein_confs, dtype=np.float32) / len(protein_confs)

        if self.ensemble_weights.is_set:
            weights = np.array(self.ensemble_weights.value, dtype=np.float32)

        if (refs := self.inp_ref.receive_optional()) is not None:
            use_reference = True
            for i, ref in enumerate(refs):
                ref.to_sdf(Path(f"ref-{i}.sdf"))
        else:
            use_reference = False
            search_centers = self.search_center.value

        inputs = Path("mols.sdf")

        # Get GPU status
        mps_only = False
        gpus = GPU.from_system()
        gpu_ok = any(gpu.free for gpu in gpus)
        if not gpu_ok:
            mps_only = any(gpu.free_with_mps for gpu in gpus)
        self.logger.info(
            "GPU %savailable %s", "not " if not gpu_ok else "", ", MPS required" if mps_only else ""
        )

        if not self.score_only.value:
            save_sdf_library(inputs, mols, split_strategy="none")

        commands = []
        outputs = []
        for i, conf in enumerate(protein_confs):
            if self.score_only.value:
                inputs = Path(f"mols-{i}.sdf")
                mols[i].molecules[0].to_sdf(inputs)

            output = Path(f"output-{i}.sdf")
            command = (
                f"{self.runnable['gnina']} -l {inputs.as_posix()} -r {conf.as_posix()} "
                f"--scoring {self.scoring.value} --cnn_scoring {self.cnn_scoring.value} "
                f"--exhaustiveness {self.exhaustiveness.value} --num_modes {self.n_poses.value} "
                f"--cpu {self.n_jobs.value} --out {output.as_posix()} "
            )

            if use_reference:
                command += f"--autobox_ligand ref-{i}.sdf "
                command += f"--autobox_add {self.autobox_add.value} "

            elif self.score_only.value:
                command += "--score_only "

            elif self.minimize.value:
                command += "--minimize "

            else:
                x, y, z = search_centers[i]
                dx, dy, dz = self.search_range.value
                command += f"--center_x {x} --center_y {y} --center_z {z} "
                command += f"--size_x {dx} --size_y {dy} --size_z {dz} "

            if self.cnn.is_set:  # builtin models
                command += f"--cnn {self.cnn.value} "
            elif self.cnn_model.is_set:  # custom trained model
                command += f"--cnn_model {' '.join(p.as_posix() for p in self.cnn_model.filepath)} "

            # We have the following scenarios for our GPUs:
            # 1) No GPU in the system / user doesn't want GPU -> Run on CPU
            # 2) GPU available but blocked and no MPS -> Run on CPU
            # 3) GPU available but blocked, MPS running -> Run on GPU but use MPS
            # 4) GPU available -> Run on GPU
            if not self.gpu.value or not (gpu_ok or mps_only):
                command += "--no_gpu "

            commands.append(command)
            outputs.append(output)

        self.run_multi(
            commands,
            cuda_mps=mps_only and not self.batch_options.is_set,
            n_jobs=self.n_parallel.value,
        )

        libs = [
            load_sdf_library(output, split_strategy="inchi", renumber=False) for output in outputs
        ]
        for i, lib in enumerate(libs):
            for mol in lib:
                for iso in mol.molecules:
                    for score_tag, agg in zip(self.SCORE_TAGS, self.SCORE_TAGS_AGG):
                        value = float(cast(float, iso.get_tag(score_tag)))
                        iso.add_score(f"{score_tag}-{i}", value, agg=agg)

        mols = reduce(partial(merge_libraries, overwrite_conformers=False), libs)
        for mol in mols:
            for iso in mol.molecules:
                for score_tag, agg in zip(self.SCORE_TAGS, self.SCORE_TAGS_AGG):
                    all_scores = np.array(
                        [iso.scores[f"{score_tag}-{i}"] for i, _ in enumerate(outputs)]
                    )
                    iso.add_score(score_tag, float(weights @ all_scores), agg=agg)
                iso.primary_score_tag = self.PRIMARY_SCORE_TAG
                iso.set_tag("score_type", "oracle")
                iso.set_tag("origin", self.name)
                self.logger.info(
                    "Parsed isomer '%s', score %s", iso.name or iso.inchi, iso.primary_score
                )
            mol.primary_score_tag = self.PRIMARY_SCORE_TAG

        self.out.send(mols)


def find_attachment_point_index(match_idx: list[int], frag_mol: Chem.Mol) -> list[int] | None:
    """Find index of the dummy atom which will be the attachment point

    The indexes are the locations in the matching list of the dummy atom.
    The matching list contains the indexes of the heavy atoms of the molecule.

    :param match_idx: the subsstructure indexes in the molecule that match the fragment
    :param frag_mol: the matching fragment molecule
    :returns: location of dummy atom
    """

    ap_indexes = []

    for idx in range(len(match_idx)):
        frag_atom = frag_mol.GetAtomWithIdx(idx)

        if frag_atom.GetSymbol() == "*":
            ap_indexes.append(match_idx[idx])

    if len(ap_indexes) != 1:  # only one AP
        return None

    return ap_indexes[0]


def find_hydrogens(mol: Chem.Mol, heavy_idx: list[int]) -> list:
    """Find the hydrogen from the fragment in the molecule

    The fragment is expected to contain heavy atoms only

    :param mol: molecule
    :param heavy_idx: indexes of heavy atoms correspoding to fragment
    :returns: hydrogens attached to the match
    """

    hydrogen_idx = []

    for idx in heavy_idx:
        atom = mol.GetAtomWithIdx(idx)

        for neighbor_atom in atom.GetNeighbors():
            if neighbor_atom.GetAtomicNum() == 1:  # skip H attached to dummy
                hydrogen_idx.append(neighbor_atom.GetIdx())

    return hydrogen_idx


def delete_fragmemt_from_mol(mol: Chem.Mol, indexes: list[int]) -> Chem.Mol:
    """Delete fragment from molecule

    :param mol: molecule
    :param indexes: indexex of all atoms to delete
    :returns: molecule with remaining atoms
    """

    rwmol = Chem.RWMol(mol)
    indices_to_remove = set()

    for idx in indexes:
        indices_to_remove.add(idx)

    for idx in sorted(indices_to_remove, reverse=True):
        rwmol.RemoveAtom(idx)

    return rwmol.GetMol()


def reorder_atoms(mol: Chem.Mol, map_num: int) -> Chem.Mol | None:
    """Reorder atoms in molecule with chosen atom to come first

    Note: using isotope for tagging as this is also supported by OpenBabel

    :param mol: molecule
    :param map_num: atom map number of atom that needs to come first
    :retunrs: reordered molecule or None if there is not exactly one tagged atom
    """

    fields = mol.GetPropsAsDict()
    name = mol.GetProp("_Name")

    num_iso = 0

    for first_idx, atom in enumerate(mol.GetAtoms()):
        if atom.GetIsotope() == map_num:
            num_iso += 1
            break

    if num_iso != 1:
        return None

    order = list(range(mol.GetNumAtoms()))
    order[0], order[first_idx] = order[first_idx], order[0]

    reordered_mol = Chem.RenumberAtoms(mol, order)
    reordered_mol.SetProp("_Name", name)

    for key, value in fields.items():
        reordered_mol.SetProp(key, str(value))

    return reordered_mol


def has_one_dummy(mol: Chem.Mol) -> bool:
    """Check if molecule has exactly one dummy"""

    n_dummies = 0

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            n_dummies += 1

    if n_dummies != 1:
        return False

    return True


class GNINA(_GNINA):
    """
    Docks molecules with GNINA.

    See the `repo <https://github.com/gnina/gnina>`_ and [#mcnutt2021]_ for more information.

    References
    ----------
    .. [#mcnutt2021] McNutt, A., Francoeur, P., Aggarwal, R., Masuda, T., Meli, R.,
       Ragoza, M., Sunseri, J. & Koes, D. R. GNINA 1.0: Molecular docking with deep learning.
       J. Cheminformatics 13, 43, (2021).

    """

    tags = {"chemistry", "docking", "scorer", "tagger"}

    inp_ref: Input[Isomer | str] = Input(optional=True)
    """Reference pose input, or name of a compound"""

    flex_dist: Parameter[float] = Parameter(default=0.0)
    """Distance around the refeence pose for flexible residues"""

    n_cnn_rot: Parameter[int] = Parameter(default=0)
    """Number of rotations for each pose"""

    receptor: FileParameter[PDBFileType] = FileParameter(optional=True)
    """Path to the receptor structure"""

    search_center: Parameter[tuple[float, float, float]] = Parameter(optional=True)
    """Center of the search space for docking"""

    blind: Flag = Flag(default=False)
    """
    If ``True``, will attempt blind docking to the full protein,
    you should increase ``exhaustiveness`` in this case.

    """
    tag_nan_score: Parameter[str] = Parameter(optional=True)

    def run(self) -> None:
        mols = self.inp.receive()
        smilies = {mol.name: mol.smiles for mol in mols}

        protein = self.receptor.filepath
        inputs = Path("mols.sdf")
        output = Path("output.sdf")

        # Get GPU status
        mps_only = False
        gpus = GPU.from_system()
        gpu_ok = any(gpu.free for gpu in gpus)

        if not gpu_ok and gpus:
            mps_only = any(gpu.free_with_mps for gpu in gpus)

        self.logger.info(
            "GPU %savailable%s", "not " if not gpu_ok else "", ", MPS required" if mps_only else ""
        )

        command = (
            f"{self.runnable['gnina']} "
            f"--ligand {inputs.resolve().as_posix()} --receptor {protein.resolve().as_posix()} "
            f"--scoring {self.scoring.value} --cnn_scoring {self.cnn_scoring.value} "
            f"--exhaustiveness {self.exhaustiveness.value} --num_modes {self.n_poses.value} "
            f"--cpu {self.n_jobs.value} --out {output.resolve().as_posix()} "
        )

        rsef: Isomer | str | None
        kekulize = True
        map_num = 99
        is_covalent = False
        orig_dummy_loc = -1

        if self.covalent_kekulize.is_set:
            kekulize = self.covalent_kekulize.value

        if self.covalent_ref.is_set:
            kekulize = False
            is_covalent = True
            fragment_mol_ref = Chem.MolFromMolFile(self.covalent_ref.value, removeHs=False)

            # FIXME: assumes only one dummy with one neighbour
            for atom in fragment_mol_ref.GetAtoms():
                if atom.GetSymbol() == "*":
                    orig_dummy_loc = atom.GetIdx()

                    for neighbour_atom in atom.GetNeighbors():
                        ap_frag_idx = neighbour_atom.GetIdx()

                    break

            frag_num_atoms = fragment_mol_ref.GetNumAtoms()
            fragment_mol = Chem.RemoveHs(fragment_mol_ref)

            if not has_one_dummy(fragment_mol):
                msg = "Covalent SMARTS must have exactly one dummy atom"
                self.logger.critical(msg)
                raise ValueError(msg)

            if not self.covalent_ap_fragment.is_set:
                msg = "Covalent docking requires fragment attachment point"
                self.logger.critical(msg)
                raise ValueError(msg)

            covalent_ap = self.covalent_ap_fragment.value

            enumerator = rdMolStandardize.TautomerEnumerator()
            fragment_mol_cmp = enumerator.Canonicalize(fragment_mol)

            uncharger = rdMolStandardize.Uncharger()
            fragment_mol_cmp = uncharger.uncharge(fragment_mol_cmp)

            for mol in mols:
                for iso in mol.molecules:
                    iso_mol = iso._molecule
                    Chem.SanitizeMol(iso_mol)

                    # clean-up to deal with protonation, charge and tautomer states
                    iso_mol_noH = Chem.RemoveHs(iso_mol)
                    iso_mol_cmp = enumerator.Canonicalize(iso_mol_noH)
                    iso_mol_cmp = uncharger.uncharge(iso_mol_cmp)

                    match_idx = iso_mol_cmp.GetSubstructMatches(fragment_mol_cmp, useChirality=False)

                    # gypsum may generate non-matching variants e.g. tautomers
                    if not match_idx:
                        msg = "Fragment does not match molecule"
                        self.logger.debug(msg)
                        continue

                    if len(match_idx) > 1:
                        self.logger.warning("Fragment matches molecule more than once")

                    heavy_idx = list(match_idx[0])
                    dummy_loc = find_attachment_point_index(heavy_idx, fragment_mol)

                    if dummy_loc is None:
                        self.logger.warning(
                            f"Dummy location not found in {Chem.MolToSmiles(iso_mol)}"
                        )
                        continue

                    ap_atom = iso_mol.GetAtomWithIdx(dummy_loc)
                    ap_atom.SetIsotope(map_num)
                    heavy_idx.remove(dummy_loc)

                    hydrogen_idx = find_hydrogens(iso_mol, heavy_idx)
                    new_mol = delete_fragmemt_from_mol(iso_mol, heavy_idx + hydrogen_idx)

                    if not new_mol:
                        self.logger.warning(
                            f"Dummy location could not be assigned in {Chem.MolToSmiles(iso_mol)}"
                        )
                        continue

                    iso._molecule = reorder_atoms(new_mol, map_num)

            ref = self.inp_ref.receive_optional()

            if ref is None:
                msg = "SDF references is required"
                self.logger.critical(msg)
                raise ValueError(msg)

            ref_file = Path("ref.sdf")
            ref.to_sdf(ref_file)

            command += f"--autobox_ligand {ref_file.resolve().as_posix()} --autobox_add {self.autobox_add.value} "
            command += f"--covalent_rec_atom {covalent_ap} --covalent_lig_atom_pattern '*' "
        elif self.local_opt_ref.is_set:
            ref_mol = Chem.MolFromMolFile(self.local_opt_ref.value, removeHs=True)

            for mol in mols:
                for iso in mol.molecules:
                    iso_mol = iso._molecule
                    Chem.SanitizeMol(iso_mol)

                    try:  # 3D constraint embedding
                        iso_mol = Chem.ConstrainedEmbed(iso_mol, ref_mol, useTethers=True)
                    except ValueError:  # embedding based on 2D match
                        # FIXME: multiple matches
                        initial_match = iso_mol.GetSubstructMatch(ref_mol)
                        atom_map_initial = list(zip(initial_match, range(iso_mol.GetNumAtoms())))

                        try:
                            # in-place alignment!
                            rmsd = Chem.AlignMol(iso_mol, ref_mol, atomMap=atom_map_initial)
                            self.logger.debug(f"{rmsd=}")
                        except ValueError:  # Bad Conformer Id
                            pass

                    iso._molecule = iso_mol

            # only local optimization and minimization of the whole ligand
            command += "--local_only --minimize "
        elif (ref := self.inp_ref.receive_optional()) is not None:
            ref_file = Path("ref.sdf")
            if isinstance(ref, str):
                ref = find_mol(mols, value=ref)
            ref.to_sdf(ref_file)
            command += f"--autobox_ligand {ref_file.resolve().as_posix()} "
            command += f"--autobox_add {self.autobox_add.value} "

            if self.flex_dist.value > 0.1:
                command += f"--flexdist_ligand {ref_file.resolve().as_posix()} "
                command += f"--flexdist {self.flex_dist.value} "

        # Treat the whole protein as the search area if we don't know the pocket location
        elif self.blind.value:
            command += f"--autobox_ligand {protein.resolve().as_posix()} "

        # In this case we're supplying the complex, so no need for a search box
        elif self.score_only.value:
            command += "--score_only "

        # Same as above, start from the complex and just minimize without search
        elif self.minimize.value:
            command += "--minimize "

        elif self.local_only.value:
            command += "--local_only --minimize "

        # In all other cases we need to tell GNINA where to look for a pocket
        else:
            x, y, z = self.search_center.value
            dx, dy, dz = self.search_range.value
            command += f"--center_x {x} --center_y {y} --center_z {z} "
            command += f"--size_x {dx} --size_y {dy} --size_z {dz} "

        if self.cnn.is_set:  # builtin CNN models
            command += f"--cnn {self.cnn.value} "
        elif self.cnn_model.is_set:  # read CNN model from file
            command += (
                f"--cnn_model {' '.join(p.resolve().as_posix() for p in self.cnn_model.filepath)} "
            )

        command += f"--cnn_rotation {self.n_cnn_rot.value} "

        # We have the following scenarios for our GPUs:
        # 1) No GPU in the system / user doesn't want GPU -> Run on CPU
        # 2) GPU available but blocked and no MPS -> Run on CPU
        # 3) GPU available but blocked, MPS running -> Run on GPU but use MPS
        # 4) GPU available -> Run on GPU
        if not self.gpu.value or not (gpu_ok or mps_only):
            command += "--no_gpu "

        # NOTE: "schrodinger" splitting would change molecule name to "mol:iso"
        #       None leaves it unmodified
        save_sdf_library(inputs, mols, split_strategy=None, kekulize=kekulize)

        self.logger.debug(f"{command=}")
        self.run_command(
            command,
            validators=[FileValidator(output)],
            cuda_mps=(mps_only and not self.batch_options.is_set) and self.gpu.value,
            prefer_batch=True,
        )

        # FIXME: review splitting strategy
        # Edge case: REINVENT may generate the same molecule e.g.
        # "CN(C(=O)O)C(=O)c1ccc(F)cc1Br" vs "CN(C(=O)[O-])C(=O)c1ccc(F)cc1Br"
        mols = load_sdf_library(output, split_strategy="inchi", sanitize=False, renumber=False)

        for mol in mols:
            for iso in mol.molecules:
                if is_covalent:  # unclear why Gnina uses a prefix to the InChiKey
                    if iso.name.startswith("_"):
                        iso.name = iso.name[1:]

                    combined = Chem.CombineMols(iso._molecule, fragment_mol_ref)
                    rw_mol = Chem.RWMol(combined)
                    offset = iso._molecule.GetNumAtoms()

                    # FIXME: assumes AP is first atom
                    if offset > 0:
                        rw_mol.AddBond(0, ap_frag_idx + offset, Chem.BondType.SINGLE)
                        rw_mol.RemoveAtom(orig_dummy_loc + offset)

                    try:
                        Chem.SanitizeMol(rw_mol)
                    except (Chem.KekulizeException, Chem.AtomValenceException) as error:
                        self.logger.debug(f"Sanitization failed: {error}")
                        pass

                    iso._molecule = rw_mol.GetMol()

                for score_tag, agg in zip(self.SCORE_TAGS, self.SCORE_TAGS_AGG):
                    try:
                        iso.add_score_tag(score_tag, agg=agg)

                        for conf in iso.conformers:
                            conf.add_score_tag(score_tag, agg=agg)

                        self.logger.debug(f"Found score tag {score_tag}")
                    except KeyError:
                        continue

                iso.primary_score_tag = self.PRIMARY_SCORE_TAG
                iso.set_tag("score_type", "oracle")
                iso.set_tag("origin", self.name)
                self.logger.info(
                    "Parsed isomer '%s', score %s", iso.name or iso.inchi, iso.primary_score
                )

            mol.primary_score_tag = self.PRIMARY_SCORE_TAG

            # recover SMILES as they are the identifier for REINVENT
            for name, smiles in smilies.items():
                if mol.name == name:
                    mol.smiles = smiles
                    break

        self.out.send(mols)


# 1UYD previously published with Icolos (IcolosData/molecules/1UYD)
@pytest.fixture
def protein_path(shared_datadir: Path) -> Path:
    return shared_datadir / "1UYD_apo.pdb"


@pytest.fixture
def ligand_path(shared_datadir: Path) -> Path:
    return shared_datadir / "1UYD_ligand.sdf"


@pytest.fixture
def cnn_model_path(shared_datadir: Path) -> Path:
    return shared_datadir / "default2017.pt"


class TestSuiteGNINA:
    @pytest.mark.needs_node("gnina")
    def test_GNINA(self, temp_working_dir: Path, protein_path: Path, test_config: Config) -> None:
        """Test GNINA in isolation"""
        rig = TestRig(GNINA, config=test_config)
        params = {
            "search_center": (3.3, 11.5, 24.8),
            "receptor": protein_path,
            "n_poses": 4,
        }
        mol = IsomerCollection.from_smiles("Nc1nc(F)nc(c12)n(CCCC)c(n2)Cc3cc(OC)ccc3OC")
        mol.embed()
        n_atoms_in = mol.molecules[0].n_atoms
        res = rig.setup_run(parameters=params, inputs={"inp": [[mol]]})
        docked = res["out"].get()
        assert docked is not None
        assert len(docked) == 1
        assert docked[0].scored
        assert docked[0].molecules[0].n_conformers == 4
        assert -11.0 < docked[0].primary_score < -7.0
        n_atoms_out = docked[0].molecules[0].n_atoms
        assert n_atoms_in == n_atoms_out

    @pytest.mark.needs_node("gnina")
    def test_GNINA_custom_model(
        self, temp_working_dir: Path, protein_path: Path, test_config: Config, cnn_model_path: Path
    ) -> None:
        """Test GNINA in isolation"""
        rig = TestRig(GNINA, config=test_config)
        params = {
            "search_center": (3.3, 11.5, 24.8),
            "receptor": protein_path,
            "n_poses": 4,
            "cnn_model": [cnn_model_path],
        }
        mol = IsomerCollection.from_smiles("Nc1nc(F)nc(c12)n(CCCC)c(n2)Cc3cc(OC)ccc3OC")
        mol.embed()
        n_atoms_in = mol.molecules[0].n_atoms
        res = rig.setup_run(parameters=params, inputs={"inp": [[mol]]})
        docked = res["out"].get()
        assert docked is not None
        assert len(docked) == 1
        assert docked[0].scored
        assert docked[0].molecules[0].n_conformers == 4
        assert -11.0 < docked[0].primary_score < -7.0
        n_atoms_out = docked[0].molecules[0].n_atoms
        assert n_atoms_in == n_atoms_out

    @pytest.mark.needs_node("gninaensemble")
    def test_GNINA_ensemble(
        self, temp_working_dir: Path, protein_path: Path, test_config: Config
    ) -> None:
        """Test GNINA in isolation"""
        rig = TestRig(GNINAEnsemble, config=test_config)
        params = {
            "search_center": np.array([[3.3, 11.5, 24.8], [3.3, 11.5, 24.8]]),
            "ensemble": [protein_path, protein_path],
            "n_poses": 4,
        }
        mol = IsomerCollection.from_smiles("Nc1nc(F)nc(c12)n(CCCC)c(n2)Cc3cc(OC)ccc3OC")
        mol.embed()
        res = rig.setup_run(parameters=params, inputs={"inp": [[mol]]})
        docked = res["out"].get()
        assert docked is not None
        assert len(docked) == 1
        assert docked[0].scored
        assert docked[0].molecules[0].n_conformers == 8
        assert -11.0 < docked[0].primary_score < -7.0
        assert "minimizedAffinity" in docked[0].molecules[0].scores
        assert "minimizedAffinity-0" in docked[0].molecules[0].scores
        assert "minimizedAffinity-1" in docked[0].molecules[0].scores
        assert "CNNaffinity" in docked[0].molecules[0].scores
        assert "CNNaffinity-0" in docked[0].molecules[0].scores
        assert "CNNaffinity-1" in docked[0].molecules[0].scores

    @pytest.mark.needs_node("gnina")
    def test_GNINA_ref(
        self, temp_working_dir: Path, protein_path: Path, ligand_path: Path, test_config: Config
    ) -> None:
        """Test GNINA with reference"""
        rig = TestRig(GNINA, config=test_config)
        params = {
            "receptor": protein_path,
            "n_poses": 4,
        }
        mol = IsomerCollection.from_smiles("Nc1nc(F)nc(c12)n(CCCC)c(n2)Cc3cc(OC)ccc3OC")
        mol.embed()
        ref = Isomer.from_sdf(ligand_path)
        res = rig.setup_run(parameters=params, inputs={"inp": [[mol]], "inp_ref": [ref]})
        docked = res["out"].get()
        assert docked is not None
        assert len(docked) == 1
        assert docked[0].scored
        assert docked[0].molecules[0].n_conformers == 4
        assert -5.0 < docked[0].primary_score < 0.0

    @pytest.mark.needs_node("gnina")
    def test_GNINA_ref_string(
        self, temp_working_dir: Path, protein_path: Path, ligand_path: Path, test_config: Config
    ) -> None:
        """Test GNINA with reference"""
        rig = TestRig(GNINA, config=test_config)
        params = {
            "receptor": protein_path,
            "n_poses": 4,
        }
        mol = IsomerCollection.from_smiles("Nc1nc(F)nc(c12)n(CCCC)c(n2)Cc3cc(OC)ccc3OC")
        mol.embed()
        ref = Isomer.from_sdf(ligand_path)
        res = rig.setup_run(
            parameters=params,
            inputs={"inp": [[mol, IsomerCollection([ref])]], "inp_ref": [ref.name]},
        )
        docked = res["out"].get()
        assert docked is not None
        assert len(docked) == 2
        assert docked[0].scored
        assert docked[0].molecules[0].n_conformers == 4
        assert -5.0 < docked[0].primary_score < -1.0

    @pytest.mark.needs_node("gnina")
    def test_GNINA_score_only(
        self, temp_working_dir: Path, protein_path: Path, ligand_path: Path, test_config: Config
    ) -> None:
        """Test GNINA with reference"""
        rig = TestRig(GNINA, config=test_config)
        params = {"receptor": protein_path, "score_only": True}
        ref = IsomerCollection.from_sdf(ligand_path)
        res = rig.setup_run(parameters=params, inputs={"inp": [[ref]]})
        docked = res["out"].get()
        assert docked is not None
        assert len(docked) == 1
        assert docked[0].scored
        assert docked[0].molecules[0].n_conformers == 1
        assert -5.0 < docked[0].primary_score < 0.0

    @pytest.mark.needs_node("gnina")
    def test_GNINA_minimize(
        self, temp_working_dir: Path, protein_path: Path, ligand_path: Path, test_config: Config
    ) -> None:
        """Test GNINA with reference"""
        rig = TestRig(GNINA, config=test_config)
        params = {"receptor": protein_path, "minimize": True}
        ref = IsomerCollection.from_sdf(ligand_path)
        res = rig.setup_run(parameters=params, inputs={"inp": [[ref]]})
        docked = res["out"].get()
        assert docked is not None
        assert len(docked) == 1
        assert docked[0].scored
        assert docked[0].molecules[0].n_conformers == 1
        assert -5.0 < docked[0].primary_score < 0.0

    @pytest.mark.needs_node("gnina")
    def test_GNINA_blind(
        self, temp_working_dir: Path, protein_path: Path, test_config: Config
    ) -> None:
        """Test GNINA in blind mode (no reference / pocket info)"""
        rig = TestRig(GNINA, config=test_config)
        params = {"receptor": protein_path, "n_poses": 4, "blind": True}
        mol = IsomerCollection.from_smiles("Nc1nc(F)nc(c12)n(CCCC)c(n2)Cc3cc(OC)ccc3OC")
        mol.embed()
        res = rig.setup_run(parameters=params, inputs={"inp": [[mol]]})
        docked = res["out"].get()
        assert docked is not None
        assert len(docked) == 1
        assert docked[0].scored
        assert docked[0].molecules[0].n_conformers == 4
        assert -11.0 < docked[0].primary_score < -7.0
