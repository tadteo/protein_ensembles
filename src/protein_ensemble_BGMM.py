import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from scipy.stats import multivariate_normal
from scipy.interpolate import splprep, splev
from sklearn.decomposition import PCA

class ProteinEnsembleBGMM:
    def __init__(self, n_components=5, reg_coef=1e-6):
        self.bgmm = BayesianGaussianMixture(n_components=n_components, 
                                            covariance_type='full',
                                            reg_covar=reg_coef)
        self.reg_coef = reg_coef
        self.n_atoms = None
        self.n_dims = None
        self.pca = None

    def fit(self, conformations):
        self.n_samples, self.n_atoms, self.n_dims = conformations.shape
        flat_conformations = conformations.reshape(self.n_samples, -1)
        self.bgmm.fit(flat_conformations)
        
         # Perform PCA on the data
        self.pca = PCA()
        self.pca.fit(flat_conformations)
        
        # Ensure positive semidefiniteness of covariance matrices
        for i in range(self.bgmm.n_components):
            self.bgmm.covariances_[i] = self.ensure_positive_semidefinite(self.bgmm.covariances_[i])

    def ensure_positive_semidefinite(self, cov):
        eigvals = np.linalg.eigvalsh(cov)
        if np.any(eigvals < 0):
            print(f"Adjusting eigenvalues for component...")
            min_eigval = np.min(eigvals)
            cov += (np.abs(min_eigval) + self.reg_coef) * np.eye(cov.shape[0])
        return cov

    def posterior(self, x):
        flat_x = x.flatten()
        weights = self.bgmm.weights_
        means = self.bgmm.means_
        covariances = self.bgmm.covariances_
        
        likelihoods = np.array([multivariate_normal.pdf(flat_x, mean=mean, cov=cov) 
                                for mean, cov in zip(means, covariances)])
        
        posterior_weights = weights * likelihoods
        posterior_weights /= np.sum(posterior_weights)
        
        return posterior_weights, means, covariances

    def sample_conformations(self, n_samples=1):
        flat_samples, _ = self.bgmm.sample(n_samples=n_samples)
        samples = flat_samples.reshape(n_samples, self.n_atoms, self.n_dims)
        
        # Apply constraints and smoothing
        constrained_samples = []
        for sample in samples:
            constrained = self.apply_constraints(sample)
            smoothed = self.smooth_conformation(constrained)
            constrained_samples.append(smoothed)
        
        return np.array(constrained_samples)

    def sample_along_directions(self, directions, n_samples=10, scale=2.0):
        if self.pca is None:
            raise ValueError("Model must be fitted before sampling along directions.")

        all_samples = []
        
        # Choose the component with the highest weight
        component = np.argmax(self.bgmm.weights_)
        mean = self.bgmm.means_[component]

        # Project mean to PCA space
        pca_mean = self.pca.transform(mean.reshape(1, -1))[0]

        for direction in directions:
            # Ensure the direction is a unit vector
            direction = np.array(direction)
            direction = direction / np.linalg.norm(direction)

            # Project the direction onto the PCA space
            pca_direction = self.pca.transform(direction.reshape(1, -1))[0]
            pca_direction = pca_direction / np.linalg.norm(pca_direction)

            samples = []
            for i in range(n_samples):
                # Generate a sample along the direction in PCA space
                factor = (i - (n_samples - 1) / 2) * scale
                pca_sample = pca_mean + factor * pca_direction

                # Transform back to original space
                flat_sample = self.pca.inverse_transform(pca_sample.reshape(1, -1))[0]
                sample = flat_sample.reshape(self.n_atoms, self.n_dims)

                # Apply constraints and smoothing
                constrained = self.apply_constraints(sample)
                smoothed = self.smooth_conformation(constrained)
                samples.append(smoothed)

            all_samples.append(np.array(samples))

        return all_samples
    
    def apply_constraints(self, conformation):
        # Enforce CA-CA distance constraint (typical range: 3.7-3.9 Å)
        target_distance = 3.7  # Angstroms
        for i in range(1, len(conformation)):
            vector = conformation[i] - conformation[i-1]
            current_distance = np.linalg.norm(vector)
            if current_distance != 0:
                conformation[i] = conformation[i-1] + vector * (target_distance / current_distance)
        return conformation

    def smooth_conformation(self, conformation, smoothing=0.5):
        # Use spline interpolation to smooth the conformation
        tck, u = splprep(conformation.T, s=smoothing)
        smooth_conformation = np.array(splev(u, tck)).T
        return smooth_conformation

    def log_likelihood(self, conformation):
        flat_conformation = conformation.flatten()
        return self.bgmm.score_samples([flat_conformation])[0]

def rmsd(conf1, conf2):
    """Calculate RMSD between two conformations"""
    return np.sqrt(np.mean(np.sum((conf1 - conf2)**2, axis=1)))
