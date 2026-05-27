"""Docking with GNINA: unified single-receptor and ensemble implementation."""

from functools import partial, reduce
from pathlib import Path
from typing import Annotated, Literal, cast

import rdkit.Chem.AllChem as Chem

from maize.core.node import Node
from maize.core.interface import Parameter, Flag, FileParameter, Suffix, Input, Output
from maize.steps.mai.gnina.fragment_growing import (
    combine_iso_with_fragment,
    prepare_mols_for_covalent,
    prepare_mols_for_local,
)
from maize.utilities.chem import (
    Isomer,
    IsomerCollection,
    find_mol,
    load_sdf_library,
    merge_libraries,
    save_sdf_library,
)
from maize.utilities.validation import FileValidator
from maize.utilities.resources import cpu_count
from maize.utilities.execution import GPU

MODES = Literal[
    "dock_with_ref",
    "dock_no_ref",
    "local_only",
    "score_only",
    "minimize_only",
    "blind",
    "fragment_covalent",  # requires modified Gnina
    "fragment_local_only",
]
ScoreType = Literal["default", "ad4_scoring", "dkoes_fast", "dkoes_scoring", "vina", "vinardo"]
CNNScoreType = Literal["none", "rescore", "refinement", "metrorescore", "metrorefine", "all"]
PDBFileType = Annotated[Path, Suffix("pdb", "pdbqt")]
INPUT_FILENAME = "mols.sdf"
POSE_REF_FILENAME = "ref.sdf"
OUTPUT_FILENAME = "output-{}.sdf"


class _GninaParameters(Node, register=False):
    """
    Collected parameters for Gnina.
    """

    tags = {"chemistry", "docking", "scorer", "tagger", "ensemble"}

    SCORE_TAGS = ("minimizedAffinity", "minimizedRMSD", "CNNscore", "CNNaffinity", "CNN_VS")
    SCORE_TAGS_AGG: tuple[Literal["min", "max"], ...] = ("min", "max", "max", "max", "max")
    PRIMARY_SCORE_TAG = "minimizedAffinity"

    required_callables = ["gnina"]

    # Inputs / Outputs

    inp: Input[list[IsomerCollection]] = Input()
    """List of molecules to dock"""

    out: Output[list[IsomerCollection]] = Output()
    """Docked molecules with conformations and scores attached"""

    inp_ref: Input[Isomer | str] = Input(optional=True)
    """Reference pose input. A single Isomer or compound name for single-receptor
    docking, or a list of Isomers (one per receptor) for ensemble docking."""

    # Receptor configuration

    receptors: FileParameter[list[PDBFileType]] = FileParameter()
    """Path(s) to receptor structure(s). A single-element list for standard docking
    or multiple paths for ensemble docking."""

    # Mode and search configuration

    mode: Parameter[Literal[MODES]] = Parameter()
    """Docking mode."""

    search_center: Parameter[tuple[float, float, float]] = Parameter(optional=True)
    """Center of the search space. A single (x, y, z) tuple (broadcast to all
    receptors) or a list of (x, y, z) tuples (one per receptor)."""

    search_range: Parameter[tuple[float, float, float]] = Parameter(default=(15.0, 15.0, 15.0))
    """Range of the search space for docking"""

    autobox_add: Parameter[float] = Parameter(default=4.0)
    """Amount of buffer space to add around the ligand"""

    # Docking parame

    exhaustiveness: Parameter[int] = Parameter(default=8)
    """Exhaustiveness of the global search (roughly proportional to time)"""

    n_poses: Parameter[int] = Parameter(default=8)
    """Maximum number of poses to generate"""

    scoring: Parameter[ScoreType] = Parameter(default="default")
    """Scoring function to use"""

    cnn_scoring: Parameter[CNNScoreType] = Parameter(default="rescore")
    """CNN scoring method to use"""

    cnn_model: FileParameter[list[Annotated[Path, Suffix("pt")]]] = FileParameter(optional=True)
    """One or more alternative CNN scoring models to use"""

    cnn: Parameter[str] = Parameter(optional=True)
    """Name of a pre-trained CNN model or ensemble models to use"""

    n_cnn_rot: Parameter[int] = Parameter(default=0)
    """Number of rotations for each CNN pose"""

    n_jobs: Parameter[int] = Parameter(default=cpu_count())
    """The number of CPUs to use per docking run"""

    gpu: Flag = Flag(default=True)
    """Whether to use the GPU for the CNN scoring step"""

    n_parallel: Parameter[int] = Parameter(default=1)
    """Number of parallel jobs for ensemble docking. Ensure that
    ``n_parallel * n_jobs`` does not exceed the number of available cores."""

    # Advanced parameters

    flex_dist: Parameter[float] = Parameter(default=0.0)
    """Distance around the reference pose for flexible residues"""

    covalent_ref: FileParameter[Annotated[Path, Suffix("sdf")]] = FileParameter(optional=True)
    """SDF of the fragment (must have hydrogens and one dummy atom!) in the receptor"""

    covalent_ap_fragment: Parameter[str] = Parameter(optional=True)
    """Attachment point of fragment as chain:resnum:atom_name"""

    local_opt_ref: FileParameter[Annotated[Path, Suffix("sdf")]] = FileParameter(optional=True)
    """Reference structure filename for local optimization"""


class Gnina(_GninaParameters):  # FIXME: change class name back later when tested
    """
    Docks molecules with GNINA, supporting both single-receptor and ensemble docking.

    When a single receptor path is provided via ``receptors``, standard docking is
    performed. When multiple receptor conformations are provided via ``receptors``,
    ensemble docking is performed: the ligand(s) are docked against each conformation
    and results are merged.

    Single-receptor docking is treated as a degenerate case of ensemble docking with
    one conformation.

    See the `repo <https://github.com/gnina/gnina>`_ and [#mcnutt2021]_ for more information.

    References
    ----------
    .. [#mcnutt2021] McNutt, A., Francoeur, P., Aggarwal, R., Masuda, T., Meli, R.,
       Ragoza, M., Sunseri, J. & Koes, D. R. GNINA 1.0: Molecular docking with deep learning.
       J. Cheminformatics 13, 43, (2021).

    """

    def run(self) -> None:
        receptors = self.receptors.filepath
        n_receptors = len(receptors)
        is_ensemble = n_receptors > 1

        mols = self.inp.receive()
        self.logger.debug("Molecules in: %d", len(mols))
        smilies = {mol.name: mol.smiles for mol in mols}

        ref = self._resolve_ref(mols)
        mode = self.mode.value

        gpu_ok, mps_only = self._get_gpu_status()

        is_covalent, fragment_mol_ref, ap_frag_idx, orig_dummy_loc = self._prepare_mode(
            mode, mols, ref
        )

        kekulize = mode != "fragment_covalent"
        inputs = Path(INPUT_FILENAME)
        save_sdf_library(inputs, mols, split_strategy="none", conformers=True, kekulize=kekulize)
        self.logger.debug("%d molecules saved to %s", len(mols), inputs)

        commands: list[str] = []
        outputs: list[Path] = []

        for i, receptor in enumerate(receptors):
            output = Path(OUTPUT_FILENAME.format(i))

            command = self._build_base_command(inputs, receptor, output)
            command = self._append_mode_flags(command, mode, i, receptors, ref)

            command = self._append_cnn_flags(command)
            command += f"--cnn_rotation {self.n_cnn_rot.value} "
            command = self._append_gpu_flags(command, gpu_ok, mps_only)

            commands.append(command)
            outputs.append(output)

        cuda_mps_flag = mps_only and not self.batch_options.is_set and self.gpu.value

        self.logger.debug("command=%s", commands[0])

        if is_ensemble:
            self.run_multi(commands, cuda_mps=cuda_mps_flag, n_jobs=self.n_parallel.value)
        else:
            self.run_command(
                commands[0],
                validators=[FileValidator(outputs[0])],
                cuda_mps=cuda_mps_flag,
                prefer_batch=True,
            )

        mols = self._process_results(
            outputs=outputs,
            smilies=smilies,
            is_covalent=is_covalent,
            fragment_mol_ref=fragment_mol_ref,
            ap_frag_idx=ap_frag_idx,
            orig_dummy_loc=orig_dummy_loc,
        )

        self.out.send(mols)

    def _resolve_ref(self, mols: list[IsomerCollection]) -> Isomer | None:
        """Resolve reference poses into a list matching the number of receptors."""
        raw_ref = self.inp_ref.receive_optional()

        if raw_ref is None:
            return None

        if isinstance(raw_ref, str):
            return find_mol(mols, value=raw_ref)

        return raw_ref

    def _get_gpu_status(self) -> tuple[bool, bool]:
        """Determine GPU availability and MPS requirement."""
        mps_only = False
        gpus = GPU.from_system()
        gpu_ok = any(gpu.free for gpu in gpus)
        if not gpu_ok and gpus:
            mps_only = any(gpu.free_with_mps for gpu in gpus)
        self.logger.info(
            "GPU %savailable%s",
            "not " if not gpu_ok else "",
            ", MPS required" if mps_only else "",
        )
        return gpu_ok, mps_only

    def _prepare_mode(
        self,
        mode: str,
        mols: list[IsomerCollection],
        ref: Isomer | None,
    ) -> tuple[bool, "Chem.Mol | None", "int | None", "int | None"]:
        """One-time preparation before the receptor loop. Modes that modify mols
        in-place (fragment_covalent, fragment_local_only) do so here."""
        is_covalent = False
        fragment_mol_ref = None
        ap_frag_idx = None
        orig_dummy_loc = None

        if mode == "fragment_covalent":
            is_covalent = True
            fragment_mol_ref, ap_frag_idx, orig_dummy_loc = self._prepare_covalent(mols, ref)
        elif mode == "fragment_local_only":
            ref_mol = Chem.MolFromMolFile(self.local_opt_ref.filepath, removeHs=True)
            prepare_mols_for_local(mols, ref_mol)

        return is_covalent, fragment_mol_ref, ap_frag_idx, orig_dummy_loc

    def _build_base_command(self, ligand_path: Path, receptor: Path, output: Path) -> str:
        """Build the common gnina command prefix."""
        return (
            f"{self.runnable['gnina']} "
            f"--ligand {ligand_path.resolve().as_posix()} "
            f"--receptor {receptor.resolve().as_posix()} "
            f"--scoring {self.scoring.value} --cnn_scoring {self.cnn_scoring.value} "
            f"--exhaustiveness {self.exhaustiveness.value} "
            f"--num_modes {self.n_poses.value} "
            f"--cpu {self.n_jobs.value} --out {output.resolve().as_posix()} "
        )

    def _append_mode_flags(
        self,
        command: str,
        mode: str,
        receptor_idx: int,
        receptors: list[Path],
        ref: Isomer | None,
    ) -> str:
        """Append per-receptor mode-specific flags to the command."""

        if mode == "dock_with_ref":
            ref_file = Path(POSE_REF_FILENAME)
            ref.to_sdf(ref_file)

            command += f"--autobox_ligand {ref_file.resolve().as_posix()} "
            command += f"--autobox_add {self.autobox_add.value} "

            if self.flex_dist.value > 0.1:
                command += f"--flexdist_ligand {ref_file.resolve().as_posix()} "
                command += f"--flexdist {self.flex_dist.value} "

        elif mode == "dock_no_ref":
            x, y, z = self.search_center.value
            dx, dy, dz = self.search_range.value
            command += f"--center_x {x} --center_y {y} --center_z {z} "
            command += f"--size_x {dx} --size_y {dy} --size_z {dz} "

        elif mode == "fragment_covalent":
            ref_file = Path(POSE_REF_FILENAME)
            ref.to_sdf(ref_file)

            covalent_ap = self.covalent_ap_fragment.value
            command += f"--autobox_ligand {ref_file.resolve().as_posix()} "
            command += f"--autobox_add {self.autobox_add.value} "
            command += f"--covalent_rec_atom {covalent_ap} --covalent_lig_atom_pattern '*' "

        elif mode in ("fragment_local_only", "local_only"):
            command += "--local_only --minimize "

        elif mode == "score_only":
            command += "--score_only "

        elif mode == "minimize_only":
            command += "--minimize "

        elif mode == "blind":
            command += f"--autobox_ligand {receptors[receptor_idx].resolve().as_posix()} "

        else:
            msg = f"Unknown mode '{mode}'"
            self.logger.critical(msg)
            raise ValueError(msg)

        return command

    def _append_cnn_flags(self, command: str) -> str:
        """Append CNN model selection flags."""
        if self.cnn.is_set:
            command += f"--cnn {self.cnn.value} "
        elif self.cnn_model.is_set:
            command += (
                f"--cnn_model "
                f"{' '.join(p.resolve().as_posix() for p in self.cnn_model.filepath)} "
            )
        return command

    def _append_gpu_flags(self, command: str, gpu_ok: bool, mps_only: bool) -> str:
        """Append GPU control flags."""
        if not self.gpu.value or not (gpu_ok or mps_only):
            command += "--no_gpu "
        return command

    def _process_results(
        self,
        outputs: list[Path],
        smilies: dict[str, str],
        is_covalent: bool,
        fragment_mol_ref,
        ap_frag_idx: int | None,
        orig_dummy_loc: int | None,
    ) -> None:
        """Load outputs, tag scores, merge libraries, and apply metadata."""

        libs = [
            load_sdf_library(
                output,
                split_strategy="schrodinger",
                sanitize=False,
                renumber=False,
            )
            for output in outputs
        ]

        is_ensemble = len(libs) > 1

        for i, lib in enumerate(libs):
            for mol in lib:
                for iso in mol.molecules:
                    if is_covalent:
                        iso._molecule = combine_iso_with_fragment(
                            iso, fragment_mol_ref, ap_frag_idx, orig_dummy_loc
                        )

                    if is_ensemble:
                        iso.set_tag("ensemble", i)

                    for score_tag, agg in zip(self.SCORE_TAGS, self.SCORE_TAGS_AGG):
                        try:
                            value = float(cast(float, iso.get_tag(score_tag)))
                            iso.add_score(score_tag, value, agg=agg)
                        except (KeyError, TypeError, ValueError):
                            continue

                    for conformer in iso.conformers:
                        self.logger.debug(f"=== {conformer=}")

        if is_ensemble:
            mols_out = reduce(partial(merge_libraries, overwrite_conformers=False), libs)
        else:
            mols_out = libs[0]

        found_smilies: set[str] = set()

        self.logger.debug(f"=== {mols_out=}")
        for mol in mols_out:
            for iso in mol.molecules:
                for score_tag, agg in zip(self.SCORE_TAGS, self.SCORE_TAGS_AGG):
                    for conf in iso.conformers:
                        try:
                            conf.add_score_tag(score_tag, agg=agg)
                        except (KeyError, TypeError):
                            continue

                iso.primary_score_tag = self.PRIMARY_SCORE_TAG
                iso.set_tag("score_type", "oracle")
                iso.set_tag("origin", self.name)
                self.logger.info(
                    "Parsed isomer '%s', score %s",
                    iso.name or iso.inchi,
                    iso.primary_score,
                )
                self.logger.debug(f"=== {iso=}")

            for name, smi in smilies.items():
                if mol.name == name:
                    mol.smiles = smi
                    found_smilies.add(smi)
                    break

            mol.primary_score_tag = self.PRIMARY_SCORE_TAG

        save_sdf_library(Path("_gnina2.sdf"), mols_out, split_strategy="none", conformers=True)
        not_found = set(smilies.values()) - found_smilies
        if not_found:
            self.logger.debug("SMILES not found: %s", not_found)

        self.logger.debug("Molecules out: %d", len(mols_out))

        return mols_out

    def _prepare_covalent(
        self, mols: list[IsomerCollection], ref: Isomer | None
    ) -> tuple["Chem.Mol", int, int]:
        """One-time preparation for covalent fragment docking: validate inputs,
        load the fragment reference, and modify mols in-place."""
        if not self.covalent_ap_fragment.is_set:
            msg = "Covalent docking requires fragment attachment point"
            self.logger.critical(msg)
            raise ValueError(msg)

        if ref is None or len(ref) == 0:
            msg = "Reference pose is required for covalent docking"
            self.logger.critical(msg)
            raise ValueError(msg)

        try:
            fragment_mol_ref = Chem.MolFromMolFile(self.covalent_ref.filepath, removeHs=False)
        except OSError:
            msg = "Covalent reference SDF cannot be read"
            self.logger.critical(msg)
            raise ValueError(msg)

        ap_frag_idx, orig_dummy_loc = prepare_mols_for_covalent(mols, fragment_mol_ref)

        return fragment_mol_ref, ap_frag_idx, orig_dummy_loc
