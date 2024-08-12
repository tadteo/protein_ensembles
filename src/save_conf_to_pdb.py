import numpy as np
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom

def save_conformations_to_pdb(conformations, template_pdb, output_prefix="conformation"):
    """
    Save conformations as PDB files, preserving original amino acid types.
    
    :param conformations: List of conformations (numpy arrays)
    :param template_pdb: Path to the original PDB file (used as a template)
    :param output_prefix: Prefix for output PDB files
    """
    # Parse the template PDB file
    parser = PDBParser()
    template_structure = parser.get_structure("template", template_pdb)
    
    # Get atom names, residue information, and amino acid types from the template
    atom_data = []
    for atom in template_structure.get_atoms():
        if atom.get_name() == "CA":  # We only need CA atoms
            residue = atom.get_parent()
            atom_data.append({
                "atom_name": atom.get_name(),
                "residue_id": residue.id,
                "chain_id": residue.get_parent().id,
                "residue_name": residue.get_resname()
            })
    
    # Ensure conformations is a list (in case of single conformation)
    if conformations.ndim == 2:
        conformations = [conformations]
    
    # Create PDB files for each conformation
    for i, conf in enumerate(conformations):
        # Create a new structure
        new_structure = Structure("conformation")
        new_model = Model(0)
        new_structure.add(new_model)
        new_chain = Chain("A")
        new_model.add(new_chain)
        
        # Add atoms to the new structure
        for atom_info, coord in zip(atom_data, conf):
            new_residue = Residue((" ", atom_info["residue_id"][1], " "), atom_info["residue_name"], atom_info["residue_id"][0])
            new_atom = Atom(atom_info["atom_name"], coord, 0, 1, " ", atom_info["atom_name"], None, element="C")
            new_residue.add(new_atom)
            new_chain.add(new_residue)
        
        # Save the new structure as a PDB file
        io = PDBIO()
        io.set_structure(new_structure)
        io.save(f"{output_prefix}_{i+1}.pdb")
        print(f"Saved conformation to {output_prefix}_{i+1}.pdb")
