"""Force field related helper functions"""

from rdkit import Chem
from rdkit.Chem import AllChem


def constraint_miminimzation(
    mol: Chem.Mol,
    logger,
    max_displ: float = 0.25,
    force_constant: float = 50.0,
    mmff_variant: str = "MMFF94",
    addHs: bool = False,
) -> Chem.Mol:
    """Constraint minimization of all conformers of a molecule.

    :param mol: molecule
    :param max_displ: maximum displacement in Angstrom
    :param force_constant: force constant
    :param mmff_variant: variant of the MMFF94 force field
    :paran addHs: whether to add hydrogens
    :returns: molecule with minimized conformers
    """

    if addHs:
        mol = Chem.AddHs(mol, addCoords=True)

    mp = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant=mmff_variant)

    if not mp:
        return mol

    for conf in mol.GetConformers():
        confId = conf.GetId()

        try:
            ff = AllChem.MMFFGetMoleculeForceField(mol, mp, confId=confId)
        except Chem.AtomValenceException:
            return mol

        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 1:
                continue

            idx = atom.GetIdx()
            pos = conf.GetAtomPosition(idx)

            ff.MMFFAddPositionConstraint(idx, maxDispl=max_displ, forceConstant=force_constant)

        ff.Minimize()

    return mol

def add_hs_to_smiles(smiles: str, mol: Chem.Mol) -> Chem.Mol | None:
    """Reconstruct hydrogens from SMILES

    :params: SMILES with hydrogens
    :mol: molecule with conformers
    """

    props = mol.GetPropsAsDict()

    full = Chem.AddHs(Chem.MolFromSmiles(smiles))
    template = Chem.RemoveHs(full)

    mol_heavy = Chem.RemoveHs(mol)
    match = template.GetSubstructMatch(mol_heavy)

    if not match:
        return None

    conf = Chem.Conformer(full.GetNumAtoms())

    for mol_conf in mol.GetConformers():
        for mol_idx, template_idx in enumerate(match):
            pos = mol_conf.GetAtomPosition(mol_idx)
            conf.SetAtomPosition(template_idx, pos)

    full.RemoveAllConformers()
    full.AddConformer(conf)

    full = Chem.AddHs(Chem.RemoveHs(full), addCoords=True)

    for name, prop in props.items():
        full.SetProp(name, str(prop))

    return full

