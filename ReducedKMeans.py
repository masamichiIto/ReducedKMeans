import numpy as np
from numpy.linalg import svd
import matplotlib.pyplot as plt

class ReducedKMeans:
    def __init__(self, n_clusters=3, n_components=2, max_iters=100, tol=1e-5, random_state=None):
        self.n_clusters = n_clusters
        self.n_components = n_components
        self.max_iters = max_iters
        self.tol = tol
        self.objective_history = []
        self.random_state = random_state

        self.P = None  # d × q
        self.Z = None  # n × K
        self.C = None  # K × q
        self.objective_history = []

    def _kmeans_plusplus_init(self, X_proj, K):
        n_samples = X_proj.shape[0]
        centers = []
        idx = np.random.randint(n_samples)
        centers.append(X_proj[idx])

        for _ in range(1, K):
            dist_sq = np.min([np.sum((X_proj - c) ** 2, axis=1) for c in centers], axis=0)
            probs = dist_sq / np.sum(dist_sq)
            cumulative_probs = np.cumsum(probs)
            r = np.random.rand()
            next_idx = np.searchsorted(cumulative_probs, r)
            centers.append(X_proj[next_idx])

        return np.array(centers)

    def _update_centroids(self, X_proj, Z):
        Z_sum = Z.sum(axis=0)
        return (Z.T @ X_proj) / Z_sum[:, np.newaxis]

    def _update_Z(self, X_proj, C):
        distances = np.linalg.norm(X_proj[:, np.newaxis, :] - C[np.newaxis, :, :], axis=2)
        labels = np.argmin(distances, axis=1)
        Z = np.zeros((X_proj.shape[0], self.n_clusters))
        Z[np.arange(X_proj.shape[0]), labels] = 1
        return Z

    def _update_projection(self, X, Z, C):
        W = Z @ C
        M = X.T @ W
        U, _, Vt = svd(M, full_matrices=False)
        return U @ Vt

    def _compute_objective(self, X):
        X_hat = self.Z @ self.C @ self.P.T
        return np.linalg.norm(X - X_hat, 'fro') ** 2/(np.linalg.norm(X, 'fro')**2)

    def fit(self, X, random_state=None):
        if random_state is not None: # 明示的にシードを指定した場合（ex. 多重スタートの場合）ここに値が入る。そうでない場合(=elseブロック)は、インスタンスの初期化時に指定したseedが指定される。
            np.random.seed(random_state) # 多重スタート用
        else:
            np.random.seed(self.random_state)


        n_samples, d = X.shape
        q = self.n_components
        K = self.n_clusters

        self.P = np.linalg.qr(np.random.randn(d, q))[0]
        X_proj = X @ self.P
        self.C = self._kmeans_plusplus_init(X_proj, K)
        self.Z = self._update_Z(X_proj, self.C)

        prev_obj = None
        self.objective_history = []

        for iteration in range(self.max_iters):
            X_proj = X @ self.P
            self.C = self._update_centroids(X_proj, self.Z)
            Z_new = self._update_Z(X_proj, self.C)
            P_new = self._update_projection(X, Z_new, self.C)

            self.Z = Z_new
            self.P = P_new

            obj = self._compute_objective(X)
            self.objective_history.append(obj)

            if prev_obj is not None and abs(prev_obj - obj) < self.tol:
                print(f"収束しました (iter={iteration})")
                break

            prev_obj = obj

        return self

    def fit_multiple_starts(self, X, n_init=10):
        best_obj = np.inf
        best_result = None
        all_objectives = []

        for i in range(n_init):
            random_state = None if self.random_state is None else self.random_state + i
            self.fit(X, random_state=random_state)
            final_obj = self.objective_history[-1]
            all_objectives.append(final_obj)
            if final_obj < best_obj:
                best_obj = final_obj
                best_result = {
                    "membership_matrix": self.Z.copy(),
                    "centroids": self.C.copy(),
                    "projection": self.P.copy(),
                    "objective_history": self.objective_history.copy(),
                }

        # 最良解をセット
        self.Z = best_result["membership_matrix"]
        self.C = best_result["centroids"]
        self.P = best_result["projection"]
        self.objective_history = best_result["objective_history"]

        return self

    def predict(self, X):
        X_proj = X @ self.P
        distances = np.linalg.norm(X_proj[:, np.newaxis, :] - self.C[np.newaxis, :, :], axis=2)
        return np.argmin(distances, axis=1)

    def transform(self, X):
        return X @ self.P

    def plot_objective(self):
        if not self.objective_history:
            print("fit() を先に実行してください。")
            return
        plt.plot(self.objective_history, marker='o')
        plt.xlabel("Iteration")
        plt.ylabel("Objective (Frobenius norm squared)")
        plt.title("Objective Function Over Iterations")
        plt.grid(True)
        plt.show()
