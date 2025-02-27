import numpy as np
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import spsolve
from joblib import Parallel, delayed
from tqdm import tqdm


class AsymmetricPlsEstimator:
    def __init__(self, lam=1e6, p=0.001, d=2, max_iter=50, tol=1e-6, n_jobs=-1):
        """
        Params
          lam     : smoothing factor (normally 1e5 to 1e8)
          p       : asymmetric factor (default 0.001)
          d       : difference matrix rank (default 2)
          max_iter: max iteration
          tol     : tolerance of convergence
          n_jobs  : maximum cores for parallel calculartion
        """
        self.lam = lam
        self.p = p
        self.d = d
        self.max_iter = max_iter
        self.tol = tol
        self.n_jobs = n_jobs

    def _difference_matrix(self, m):
        """
        construction difference matrix
        """
        D = np.diff(np.eye(m), n=self.d, axis=0)
        return csc_matrix(D)

    def _process_spectrum(self, col):
        """
        handle single spectrum and return baseline estimation

        Params:
          col: 1D numpy array
        """
        m = len(col)
        # difference matrix and L
        D = self._difference_matrix(m)
        L = self.lam * (D.T @ D)
        w = np.ones(m)
        for _ in range(self.max_iter):
            A = diags(w, 0, shape=(m, m)) + L
            b = w * col
            z_col = spsolve(A, b)
            # update weight
            w_new = self.p * (col > z_col) + (1 - self.p) * (col <= z_col)
            if np.sum(np.abs(w - w_new)) < self.tol:
                break
            w = w_new
        return z_col

    def asysm_parallel(self, y):
        """
        parallel version of asymmetric pls baseline estimation
        """
        y = np.asarray(y)
        # 1D array
        if y.ndim == 1:
            return self._process_spectrum(y)
        # 2D array
        elif y.ndim == 2:
            m, n = y.shape
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._process_spectrum)(y[:, j])
                for j in tqdm(range(n), desc="Processing spectra", unit="spectra")
            )
            z = np.column_stack(results)
            return z
        else:
            raise ValueError("data should be 1d or 2d numpy array")

