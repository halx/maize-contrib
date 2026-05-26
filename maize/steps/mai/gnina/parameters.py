"""Parameters for Gnina"""

from pathlib import Path
from typing import Literal, Annotated

from maize.steps.mai.gnina.gnina2 import PDBFileType, MODES, ScoreType, CNNScoreType
from maize.utilities.chem import IsomerCollection, Isomer


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

    inp_ref: Input[list[Isomer] | Isomer | str] = Input(optional=True)
    """Reference pose input. A single Isomer or compound name for single-receptor
    docking, or a list of Isomers (one per receptor) for ensemble docking."""

    # Receptor configuration

    receptors: FileParameter[list[PDBFileType]] = FileParameter()
    """Path(s) to receptor structure(s). A single-element list for standard docking
    or multiple paths for ensemble docking."""

    # Mode and search configuration

    mode: Parameter[Literal[MODES]] = Parameter()
    """Docking mode."""

    search_center: Parameter[list[tuple[float, float, float]] | tuple[float, float, float]] = (
        Parameter(optional=True)
    )
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
