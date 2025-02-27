import numpy as np
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve
from joblib import Parallel, delayed
from tqdm import tqdm


class AirPLSEstimator:
    def __init__(self, data, lam=100, porder=1, max_iter=50, n_jobs=-1):
        """
        Asymmetric iterative reweighted PLS by Zhiming Zhang.
        https://doi.org/10.1039/b922045c

        Params
          lam     : smoothing factor (normally 1e5 to 1e8)
          p       : asymmetric factor (default 0.001)
          d       : difference matrix rank (default 2)
          max_iter: max iteration
          tol     : tolerance of convergence
          n_jobs  : maximum cores for parallel calculartion
        """
        self.X = data
        self.lam = lam
        self.p = porder
        self.max_iter = max_iter
        self.n_jobs = n_jobs

    @staticmethod
    def WhittakerSmooth(x, w, lambda_, differences=1):
        '''
        Penalized least squares algorithm for background fitting

        input
            x: input data (i.e. chromatogram or spectrum)
            w: binary masks (value is 0 if a point belongs to peaks and 1 otherwise)
            lambda_: smoothing parameter; larger lambda produces smoother background
            differences: order of the difference (penalty order)

        output
            background: the fitted background vector
        '''

        X = np.matrix(x)
        m = X.size
        E = eye(m, format='csc')
        for i in range(differences):
            E = E[1:]-E[:-1] # numpy.diff() does not work with sparse matrix. This is a workaround.
        W = diags(w, 0, shape=(m, m))
        A = csc_matrix(W+(lambda_*E.T*E))
        B = csc_matrix(W*X.T)
        background = spsolve(A, B)
        return np.array(background)

    def airPLS(self, x, lambda_, porder, max_iter):
        '''
        Adaptive iteratively reweighted penalized least squares for baseline fitting

        input
            x: input data (1D numpy array representing a spectrum)
            lambda_: smoothing parameter; larger lambda produces smoother baseline
            porder: order of the difference (penalty order)
            itermax: maximum number of iterations

        output
            baseline: the estimated background vector
        '''

        m = x.shape[0]
        w = np.ones(m)
        for i in range(1, max_iter + 1):
            z = self.WhittakerSmooth(x, w, lambda_, porder)
            d = x - z
            dssn = np.abs(d[d < 0].sum())
            # break when the sum of the negative deviations is small enough, or the maximum number of iterations is reached
            if dssn < 0.001 * np.abs(x).sum() or i == max_iter:
                if i == max_iter:
                    print('WARNING: max iteration reached!')
                break
            #  For positively deviated points considered as peaks, their weights are set to 0
            w[d >= 0] = 0
            # For negatively deviated points, the update is weighted according to the current iteration steps
            w[d < 0] = np.exp(i * np.abs(d[d < 0]) / dssn)
            # Maximum values are used for boundary points
            w[0] = np.exp(i * (d[d < 0]).max() / dssn) if np.any(d < 0) else 1
            w[-1] = w[0]
        return z

    def airPLS_parallel(self, X=None, lambda_=None, porder=None, max_iter=None, n_jobs=None):
        '''
        Parallel version of airPLS

        input
            X: signal/data shape: (n_points, n_spectra)
            lambda_: smoothing parameter
            porder: penalty order
            itermax: maximum of iteration
            n_jobs: number of CPU cores for implementation.

        output
            baseline: airPLS fitted baseline shape: (n_points, n_spectra)
        '''
        if not X:
            X = self.X
        if not lambda_:
            lambda_ = self.lam
        if not porder:
            porder = self.p
        if not max_iter:
            max_iter = self.max_iter
        if not n_jobs:
            n_jobs = self.n_jobs

        X = np.asarray(X)
        # empoly airPLS for single spectrum
        if X.ndim == 1:
            return self.airPLS(X, lambda_, porder, max_iter)
        elif X.ndim == 2:
            n_points, n_spectra = X.shape
            # handle multiple spectrum
            results = Parallel(n_jobs=n_jobs)(
                delayed(self.airPLS)(X[:, i], lambda_, porder, max_iter)
                for i in tqdm(range(n_spectra), desc="Processing spectra", unit="spectra")
            )
            # stack all results
            baseline = np.column_stack(results)
            return baseline
        else:
            raise ValueError("data should be 1d or 2d numpy array")

# example
# if __name__ == '__main__':
#     np.random.seed(0)
#     X = np.random.rand(1506, 10000)
#     baseline = airPLS_parallel(X, lambda_=100, porder=1, itermax=15, n_jobs=-1)
#     print(f"Baseline fitting complete, shape:{baseline.shape}")
