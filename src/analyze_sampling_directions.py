import numpy as np
import matplotlib.pyplot as plt

def analyze_sampling_directions(directional_samples, mean_structure):
    n_directions = len(directional_samples)
    n_atoms = mean_structure.shape[0]

    plt.figure(figsize=(15, 5 * n_directions))

    for i, samples in enumerate(directional_samples):
        # Calculate displacements from mean structure
        displacements = samples - mean_structure

        # Calculate RMSD per atom
        rmsd_per_atom = np.sqrt(np.mean(np.sum(displacements**2, axis=2), axis=0))

        # Plot RMSD per atom
        plt.subplot(n_directions, 1, i+1)
        plt.plot(range(n_atoms), rmsd_per_atom)
        plt.title(f"Direction {i+1}: RMSD per atom")
        plt.xlabel("Atom index")
        plt.ylabel("RMSD (Å)")

    plt.tight_layout()
    plt.savefig("sampling_direction_analysis.png")
    plt.close()

    print("Sampling direction analysis plot saved as 'sampling_direction_analysis.png'")
