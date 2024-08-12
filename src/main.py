import time
import numpy as np
import os
from protein_ensemble_multivariate import ProteinEnsembleMultivariate, rmsd
from load_prot_conformation import load_protein_conformations
from save_conf_to_pdb import save_conformations_to_pdb

# External variable for template PDB
template_pdb = "/proj/berzelius-2021-29/users/x_matta/PED00034e000.pdb"

def get_protein_code(pdb_path):
    # Extract the filename without extension
    filename = os.path.basename(pdb_path)
    return os.path.splitext(filename)[0]

def main():
    protein_code = get_protein_code(template_pdb)
    
    print(f"Processing protein: {protein_code}")
    print("Loading protein conformation")
    start_time = time.time()
    conformations = load_protein_conformations(template_pdb)
    end_time = time.time()
    print(f"Time taken to load protein conformation: {end_time - start_time:.2f} seconds")
    print(f"Conformations shape: {conformations.shape}")

    print("Creating and fitting Multivariate Gaussian")
    start_time = time.time()
    model = ProteinEnsembleMultivariate(reg_coef=1e-4)
    model.fit(conformations)
    end_time = time.time()
    print(f"Time taken to fit Multivariate Gaussian: {end_time - start_time:.2f} seconds")

    print(f"Mean shape: {model.mean.shape}")
    print(f"Covariance shape: {model.cov.shape}")

    print("Generating new conformations")
    start_time = time.time()
    new_conformations = model.sample(n_samples=5)
    end_time = time.time()
    print(f"Time taken to generate new conformations: {end_time - start_time:.2f} seconds")
    print(f"New conformations shape: {new_conformations.shape}")

    print("Computing log-likelihood of first original conformation")
    log_likelihood = model.log_likelihood(conformations[0])
    print(f"Log-likelihood: {log_likelihood}")

    print("Calculating RMSD from mean structure")
    mean_structure = model.mean
    for i, conf in enumerate(new_conformations):
        print(f"RMSD for conformation {i+1}: {rmsd(conf, mean_structure):.2f} Å")

    print("Saving new conformations as PDB files")
    save_conformations_to_pdb(new_conformations, 
                              template_pdb=template_pdb,
                              output_prefix=f"{protein_code}_sampled_conformation")

    print("Extracting and saving mean protein conformation")
    mean_conformation = model.mean
    save_conformations_to_pdb(mean_conformation, 
                              template_pdb=template_pdb,
                              output_prefix=f"{protein_code}_mean_conformation")

if __name__ == "__main__":
    main()
