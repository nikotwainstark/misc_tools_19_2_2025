import numpy as np
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import spsolve
from joblib import Parallel, delayed
from tqdm import tqdm


class AsymmetricPlsEstimator:
    def __init__(self, lam=1e6, p=0.001, d=2, max_iter=50, tol=1e-6, n_jobs=-1):
        """
        初始化基线处理器参数

        参数:
          lam     : 平滑参数（一般 1e5 到 1e8）
          p       : 不对称参数（一般为 0.001）
          d       : 差分阶数（一般为 2）
          max_iter: 最大迭代次数
          tol     : 收敛容差
          n_jobs  : 并行处理时使用的 CPU 核数，-1 表示使用所有可用核
        """
        self.lam = lam
        self.p = p
        self.d = d
        self.max_iter = max_iter
        self.tol = tol
        self.n_jobs = n_jobs

    def _difference_matrix(self, m):
        """
        构造 m×m 的 d 阶差分矩阵
        """
        D = np.diff(np.eye(m), n=self.d, axis=0)
        return csc_matrix(D)

    def _process_spectrum(self, col):
        """
        处理单个光谱（1D 数组）的基线校正，返回基线估计

        参数:
          col: 1D numpy 数组，单条光谱
        """
        m = len(col)
        # 预先计算差分矩阵和 L（L = lam * (D.T @ D)）
        D = self._difference_matrix(m)
        L = self.lam * (D.T @ D)
        w = np.ones(m)
        for _ in range(self.max_iter):
            A = diags(w, 0, shape=(m, m)) + L
            b = w * col
            z_col = spsolve(A, b)
            # 根据 z_col 更新权重：若 col > z_col 则取 p，否则取 (1-p)
            w_new = self.p * (col > z_col) + (1 - self.p) * (col <= z_col)
            if np.sum(np.abs(w - w_new)) < self.tol:
                break
            w = w_new
        return z_col

    def asysm_parallel(self, y):
        """
        并行计算版的 AirPLS 基线校正

        参数:
          y: numpy 数组，若为二维数组，其形状为 (m, n)，每列为一条光谱；
             若为一维数组，则直接处理该单条光谱。

        返回:
          z: 基线估计结果，维度与 y 相同
        """
        y = np.asarray(y)
        # 1D 情况
        if y.ndim == 1:
            return self._process_spectrum(y)
        # 2D 情况：对每条光谱并行处理，并用 tqdm 显示进度条
        elif y.ndim == 2:
            m, n = y.shape
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._process_spectrum)(y[:, j])
                for j in tqdm(range(n), desc="Processing spectra", unit="spectra")
            )
            z = np.column_stack(results)
            return z
        else:
            raise ValueError("输入 y 必须为 1D 或 2D numpy 数组。")

