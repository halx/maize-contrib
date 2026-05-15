"""Support routines for covalent docking with Gnina"""

import logging

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from maize.utilities.chem import IsomerCollection

logger = logging.getLogger("run")
MAP_NUM = 99


def find_attachment_point_index(match_idx: list[int], frag_mol: Chem.Mol) -> int | None:
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
    first_idx = -1

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


def combine_iso_with_fragment(iso, fragment_mol_ref, ap_frag_idx, orig_dummy_loc: int):
    if iso.name.startswith("_"):  # why does Gnina do that?
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
        pass

    iso._molecule = rw_mol.GetMol()


def prepare_mols_for_covalent(mols: list[IsomerCollection], fragment_mol_ref: Chem.Mol):
    orig_dummy_loc = ap_frag_idx = -1

    # FIXME: assumes only one dummy with one neighbour
    for atom in fragment_mol_ref.GetAtoms():
        if atom.GetSymbol() == "*":
            orig_dummy_loc = atom.GetIdx()

            for neighbour_atom in atom.GetNeighbors():
                ap_frag_idx = neighbour_atom.GetIdx()

            break

    fragment_mol = Chem.RemoveHs(fragment_mol_ref)

    if not has_one_dummy(fragment_mol):
        msg = "Covalent SMARTS must have exactly one dummy atom"
        raise ValueError(msg)

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
                continue

            if len(match_idx) > 1:
                continue

            heavy_idx = list(match_idx[0])
            dummy_loc = find_attachment_point_index(heavy_idx, fragment_mol)

            if dummy_loc is None:
                continue

            ap_atom = iso_mol.GetAtomWithIdx(dummy_loc)
            ap_atom.SetIsotope(MAP_NUM)
            heavy_idx.remove(dummy_loc)

            hydrogen_idx = find_hydrogens(iso_mol, heavy_idx)
            new_mol = delete_fragmemt_from_mol(iso_mol, heavy_idx + hydrogen_idx)

            if not new_mol:
                continue

            iso._molecule = reorder_atoms(new_mol, MAP_NUM)

    return ap_frag_idx, orig_dummy_loc


def prepare_mols_for_local(mols: list[IsomerCollection], ref_mol: Chem.Mol):
    for mol in mols:
        for iso in mol.molecules:
            iso_mol = iso._molecule
            Chem.SanitizeMol(iso_mol)

            try:
                iso_mol = Chem.ConstrainedEmbed(iso_mol, ref_mol, useTethers=True)
            except ValueError:  # embedding based on 2D match
                # FIXME: multiple matches
                initial_match = iso_mol.GetSubstructMatch(ref_mol)
                atom_map_initial = list(zip(initial_match, range(iso_mol.GetNumAtoms())))

                try:
                    _ = Chem.AlignMol(
                        iso_mol, ref_mol, atomMap=atom_map_initial
                    )  # in-place alignment!
                except ValueError:  # Bad Conformer Id
                    pass

            iso._molecule = iso_mol
