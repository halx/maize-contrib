"""Fragment growing using Gnina docking

Supports covalent docking and local-only optimization.

For covalent docking a fragment 3D molecule with hydrogen's, a single atom with
a single neighbour atom is needed.  The fragment is substructure-matched
against the molecule after tautomer canonicalization and uncharging to allow
comparison with prepared ligands.  The fragment is deleted from the molecule
for covalent docking (the receptor contains a copy of the fragment in the
binding site) ignoring the dummy atom and marking its neighbour as attachment
point (AP).  The second fragment is reordered such that the AP is the first
atom.

For local-only optimization, constraint conformer generation with the fragment
as refence is carried out to generate a 3D molecule keeping the reference
conformation.  The conformer is aligned to the reference coordinates (the
receptor contains a copy of the fragment in the binding site).
"""

import logging

from rdkit import Chem

from maize.utilities.chem import IsomerCollection, Isomer

logger = logging.getLogger("run")
MAP_NUM = 99


def prepare_mols_for_covalent(
    mols: list[IsomerCollection], fragment_mol_ref: Chem.Mol
) -> tuple[int, int]:
    """Prepare molecules for covalent docking

    Deletes reference fragment from molecule to obtain docking fragment and
    determines AP.

    :param mols: molecules to prepare
    :param fragment_mol_ref: reference fragment molecule
    :returns: index of atom which will be the AP in the docking fragment and
              dummy index in the reference fragment
    """

    ap_frag_idx, orig_dummy_loc = find_dummy(fragment_mol_ref)
    fragment_mol_noH = Chem.RemoveHs(fragment_mol_ref)
    _, heavy_dummy_loc = find_dummy(fragment_mol_noH)

    for mol in mols:
        for iso in mol.molecules:
            iso_mol = iso._molecule

            heavy_idx = get_heavy_substructure_indices(iso_mol, fragment_mol_noH, heavy_dummy_loc, MAP_NUM)
            hydrogen_idx = find_hydrogens(iso_mol, heavy_idx)
            new_mol = delete_fragmemt_from_mol(iso_mol, heavy_idx + hydrogen_idx)

            if not new_mol:
                continue

            iso._molecule = reorder_atoms(new_mol, MAP_NUM)

    return ap_frag_idx, orig_dummy_loc


def find_dummy(mol: Chem.Mol) -> tuple[int, int]:
    """Find dummy atom index and its neighbour atom index

    Expects exactly one dummy atom in the molecule attached to exactly one
    neighbour atom.

    :param mol: molecule with dummy
    :returns: indices of dummy atom and neighbour atom
    :raise: ValueError if not exactly one dummy and one neighbour
    """

    n_dummies = 0
    n_neighbours = 0
    orig_dummy_loc = ap_frag_idx = -1

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            n_dummies += 1
            orig_dummy_loc = atom.GetIdx()

            for neighbour_atom in atom.GetNeighbors():
                n_neighbours += 1
                ap_frag_idx = neighbour_atom.GetIdx()

    if n_dummies != 1 or n_neighbours != 1:
        raise ValueError(
            f"Must have exactly one dummy (found {n_dummies}) "
            f"and one neighbour atom (found {n_neighbours})"
        )

    return ap_frag_idx, orig_dummy_loc


def get_heavy_substructure_indices(mol: Chem.Mol, frag: Chem.Mol, dummy_loc: int, map_num: int) -> list[int]:
    """Heavy atom substructure match

    Removes the atom in the molecule that is the equivalent of the dummy atom
    in the fragment.

    :param mol: molecule to search for substructure
    :param frag: expected substructure
    :param dummy_loc: dummy atom index
    :param map_num: isotope label number
    :returns: matching indices of heavy atoms with dummy equivalent removed
    """

    params = Chem.AdjustQueryParameters()
    params.adjustDegree = False  # same as if frag constructed as SMARTS
    params.makeDummiesQueries = True  # the default
    params.makeBondsGeneric = True  # tautomers
    query = Chem.AdjustQueryProperties(frag, params)

    match_idx = mol.GetSubstructMatches(query, useChirality=False)

    if not match_idx or len(match_idx) != 1:
        raise ValueError("Molecule does not match fragment or matches more than once")

    heavy_idx = list(match_idx[0])

    ap_atom = mol.GetAtomWithIdx(heavy_idx[dummy_loc])
    ap_atom.SetIsotope(map_num)

    heavy_idx.pop(dummy_loc)

    return heavy_idx


def find_hydrogens(mol: Chem.Mol, heavy_idx: list[int]) -> list:
    """Find the hydrogens from the fragment in the molecule

    The fragment is expected to contain heavy atoms only

    :param mol: molecule
    :param heavy_idx: indexes of heavy atoms corresponding to fragment
    :returns: hydrogens attached to the match
    """

    hydrogen_idx = []

    for idx in heavy_idx:
        atom = mol.GetAtomWithIdx(idx)

        for neighbor_atom in atom.GetNeighbors():
            if neighbor_atom.GetAtomicNum() == 1:
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


def combine_iso_with_fragment(
    mol: Isomer, fragment_mol_ref, ap_frag_idx, orig_dummy_loc: int
) -> Chem.Mol:
    """Combine the docking fragment with the reference fragment

    Assumes AP in docking fragment is at index 0.

    :param mol: docking fragment
    :param fragment_mol_ref: reference fragment (needs hydrogens!)
    :param ap_frag_idx: index of the AP in the docking fragment
    :param orig_dummy_loc: index of the dummy atom in the reference fragment, to be deleted
    :returns: combined molecule
    """

    if mol.name.startswith("_"):  # why does Gnina do that?
        mol.name = mol.name[1:]

    combined = Chem.CombineMols(mol._molecule, fragment_mol_ref)
    rw_mol = Chem.RWMol(combined)
    offset = mol._molecule.GetNumAtoms()

    if offset > 0:
        rw_mol.AddBond(0, ap_frag_idx + offset, Chem.BondType.SINGLE)
        rw_mol.RemoveAtom(orig_dummy_loc + offset)

    try:
        Chem.SanitizeMol(rw_mol)
    except (Chem.KekulizeException, Chem.AtomValenceException):
        # FIXME: check why this happens
        pass

    return rw_mol.GetMol()


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
