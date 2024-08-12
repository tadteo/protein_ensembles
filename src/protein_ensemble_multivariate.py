import numpy as np
from scipy.stats import multivariate_normal
from scipy.interpolate import splprep, splev

class ProteinEnsembleMultivariate:
    def __init__(self, reg_coef=1e-6):
        self.mean = None
        self.cov = None
        self.reg_coef = reg_coef

    def fit(self, conformations):
        n_samples, n_atoms, n_dims = conformations.shape
        self.mean = np.mean(conformations, axis=0)
        centered_conformations = conformations - self.mean

        # Flatten the centered conformations
        flat_centered_conformations = centered_conformations.reshape(n_samples, -1)

        # Compute covariance
        self.cov = np.cov(flat_centered_conformations, rowvar=False)

        # Ensure positive semidefiniteness
        self.ensure_positive_semidefinite()
       
    def ensure_positive_semidefinite(self):
        # Step 1: Check symmetry
        if not np.allclose(self.cov, self.cov.T):
            print("Warning: Covariance matrix is not symmetric")
            # Enforce symmetry
            self.cov = (self.cov + self.cov.T) / 2
        
        # Step 2: Compute eigenvalues
        eigvals = np.linalg.eigvalsh(self.cov)
        
        print(f"Eigenvalue range: [{eigvals.min()}, {eigvals.max()}]")
        print(f"Number of negative eigenvalues: {np.sum(eigvals < 0)}")
        
        # Step 3: Check condition number
        cond_num = np.abs(eigvals.max() / eigvals.min())
        print(f"Condition number: {cond_num}")
        
        # Step 4: Adjust eigenvalues if necessary
        if np.any(eigvals < 0) or cond_num > 1e15:  # Check for negative eigenvalues or poor conditioning
            print("Adjusting eigenvalues...")
            eigvals_adjusted = np.maximum(eigvals, self.reg_coef)
            adjustment_factor = np.abs(eigvals.min()) + self.reg_coef
            self.cov += np.eye(self.cov.shape[0]) * adjustment_factor
            
            # Recompute eigenvalues after adjustment
            eigvals_new = np.linalg.eigvalsh(self.cov)
            print(f"New eigenvalue range: [{eigvals_new.min()}, {eigvals_new.max()}]")
            print(f"New condition number: {np.abs(eigvals_new.max() / eigvals_new.min())}")
        
        # Final check
        final_eigvals = np.linalg.eigvalsh(self.cov)
        if np.any(final_eigvals < 0):
            print(f"Warning: Covariance matrix is still not positive-definite. Min eigenvalue: {final_eigvals.min()}")
        else:
            print("Covariance matrix is now positive-definite.")
    
    def log_likelihood(self, conformation):
        flat_conformation = conformation.flatten()
        flat_mean = self.mean.flatten()
        
        return multivariate_normal.logpdf(flat_conformation, mean=flat_mean, cov=self.cov)

    def sample(self, n_samples=1):
        flat_mean = self.mean.flatten()
        
        # Sample from the distribution
        flat_samples = np.random.multivariate_normal(flat_mean, self.cov, size=n_samples)
        samples = flat_samples.reshape(n_samples, *self.mean.shape)
        
        # Apply constraints and smoothing
        constrained_samples = []
        for sample in samples:
            constrained = self.apply_constraints(sample)
            smoothed = self.smooth_conformation(constrained)
            constrained_samples.append(smoothed)
        
        return np.array(constrained_samples)

    def apply_constraints(self, conformation):
        # Enforce CA-CA distance constraint (typical range: 3.7-3.9 Å)
        target_distance = 3.8  # Angstroms
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

def rmsd(conf1, conf2):
    """Calculate RMSD between two conformations"""
    return np.sqrt(np.mean(np.sum((conf1 - conf2)**2, axis=1)))
