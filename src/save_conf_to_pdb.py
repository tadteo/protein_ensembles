import numpy as np
import os
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom

def save_conformations_to_pdb(conformations, template_pdb, output_prefix="conformation", output_folder="."):
    """
    Save conformations as a multi-model PDB file, preserving original amino acid types.
    
    :param conformations: Numpy array of shape (n_conformations, n_atoms, 3) or (n_atoms, 3)
    :param template_pdb: Path to the original PDB file (used as a template)
    :param output_prefix: Prefix for the output PDB file
    :param output_folder: Folder to save the output PDB file (default is current directory)
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
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
    
    # Ensure conformations is a 3D array
    if conformations.ndim == 2:
        conformations = np.expand_dims(conformations, axis=0)
    
    # Create a new structure to hold all conformations
    new_structure = Structure("conformations")
    
    # Add each conformation as a separate model
    for i, conf in enumerate(conformations):
        new_model = Model(i)
        new_structure.add(new_model)
        
        # Create chains (preserving original chain IDs)
        chains = {}
        for atom_info, coord in zip(atom_data, conf):
            chain_id = atom_info["chain_id"]
            if chain_id not in chains:
                chains[chain_id] = Chain(chain_id)
                new_model.add(chains[chain_id])
            
            new_residue = Residue(atom_info["residue_id"], atom_info["residue_name"], atom_info["residue_id"][0])
            new_atom = Atom(atom_info["atom_name"], coord, 0, 1, " ", atom_info["atom_name"], None, element="C")
            new_residue.add(new_atom)
            chains[chain_id].add(new_residue)
    
    # Construct the full output file path
    output_file = os.path.join(output_folder, f"{output_prefix}.pdb")
    
    # Save the new structure as a multi-model PDB file
    io = PDBIO()
    io.set_structure(new_structure)
    io.save(output_file)
    print(f"Saved {len(conformations)} conformation(s) to {output_file}")

# Usage example
# conformations = ... # Your conformations array
# template_pdb = "path/to/your/template.pdb"
# save_conformations_to_pdb(conformations, template_pdb, "output_conformations", "path/to/output/folder")
