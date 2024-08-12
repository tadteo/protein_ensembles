import numpy as np
from Bio import PDB

def load_protein_conformations(pdb_file):
    # Create a PDB parser object
    parser = PDB.PDBParser(QUIET=True)
    
    # Parse the PDB file
    structure = parser.get_structure('protein', pdb_file)
    
    conformations = []
    
    # Iterate through models in the structure
    for model in structure:
        ca_atoms = []
        
        # Iterate through all atoms in the model
        for atom in model.get_atoms():
            # Check if the atom is an alpha carbon
            if atom.name == 'CA':
                ca_atoms.append(atom.coord)
        
        # Add the flattened coordinates of CA atoms to conformations
        if ca_atoms:
            conformations.append(np.array(ca_atoms))
    
    return np.array(conformations)

# Usage example:
# conformations = load_protein_conformations('your_protein.pdb')
