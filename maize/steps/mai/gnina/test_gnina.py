from pathlib import Path

import numpy as np
import pytest

from maize.steps.mai.gnina import GNINA
from maize.steps.mai.gnina.gnina import GNINAEnsemble
from maize.utilities.chem import IsomerCollection, Isomer
from maize.utilities.testing import TestRig
from maize.utilities.io import Config


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
