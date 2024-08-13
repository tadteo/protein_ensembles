import time
import numpy as np
import os
from protein_ensemble_BGMM import ProteinEnsembleBGMM, rmsd
from load_prot_conformation import load_protein_conformations
from save_conf_to_pdb import save_conformations_to_pdb
# from analyze_sampling_directions import analyze_sampling_directions

# External variable for template PDB
template_pdb = "/proj/berzelius-2021-29/users/x_matta/PED00044e000.pdb"
output_folder = "/proj/berzelius-2021-29/users/x_matta/predicted_conformations"

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
    model = ProteinEnsembleBGMM(n_components=5, reg_coef=1e-4)
    model.fit(conformations)
    end_time = time.time()
    print(f"Time taken to fit Multivariate Gaussian: {end_time - start_time:.2f} seconds")

    print("Generating new conformations")
    start_time = time.time()
    new_conformations = model.sample_conformations(n_samples=5)
    end_time = time.time()
    print(f"Time taken to generate new conformations: {end_time - start_time:.2f} seconds")
    print(f"New conformations shape: {new_conformations.shape}")

    print("Computing log-likelihood of first original conformation")
    log_likelihood = model.log_likelihood(conformations[0])
    print(f"Log-likelihood: {log_likelihood}")

    print("Calculating RMSD from mean of first component")
    mean_structure = model.bgmm.means_[0].reshape(model.n_atoms, model.n_dims)
    for i, conf in enumerate(new_conformations):
        print(f"RMSD for conformation {i+1}: {rmsd(conf, mean_structure):.2f} Å")

    print("Saving new conformations as PDB files")
    save_conformations_to_pdb(new_conformations, 
                              template_pdb=template_pdb,
                              output_prefix=f"bgmm_{protein_code}_sampled_conformation",
                              output_folder=output_folder)

    print("Sampling along multiple principal components")
    n_directions = 3  # Number of principal components to sample along
    directions = model.pca.components_[:n_directions]
    directional_samples = model.sample_along_directions(directions, n_samples=100, scale=2.0)

    print(f"Number of directional sample sets: {len(directional_samples)}")
    print(f"Shape of each directional sample set: {directional_samples[0].shape}")

    print("Calculating RMSD from mean structure")
    mean_structure = model.bgmm.means_[np.argmax(model.bgmm.weights_)].reshape(model.n_atoms, model.n_dims)
    for i, direction_samples in enumerate(directional_samples):
        print(f"Direction {i+1}:")
        for j, conf in enumerate(direction_samples):
            print(f"  RMSD for sample {j+1}: {rmsd(conf, mean_structure):.2f} Å")

    print("Saving directional samples as PDB file")
    all_samples = np.concatenate(directional_samples)
    save_conformations_to_pdb(all_samples, 
                              template_pdb=template_pdb,
                              output_prefix=f"bgmm_{protein_code}_multi_directional_sample",
                              output_folder=output_folder)
    
    # print("Analyzing sampling directions")
    # analyze_sampling_directions(directional_samples, mean_structure)    
    
if __name__ == "__main__":
    main()
