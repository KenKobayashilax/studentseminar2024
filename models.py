import numpy as np

# ベイズ線形回帰クラスの定義
class BayesianLinearRegression:
    #推定パラメータの初期化
    def __init__(self, mu_prior, cov_prior, alpha, beta):
        self.mu_prior = mu_prior   # パラメータの事前平均
        self.cov_prior = cov_prior  # パラメータの事前共分散行列
        self.alpha = alpha
        self.beta = beta            # ノイズの精度パラメータ

    def update(self, phi, y):
        # データの数と特徴量の次元数を取得
        N, D = phi.shape

        # パラメータの事後分布の計算
        # p152 式(3.51) 
        self.cov_posterior = np.linalg.inv(np.linalg.inv(self.cov_prior) + self.beta * phi.T @ phi)
        # p152 式(3.50)
        self.mu_posterior = self.cov_posterior @ (np.linalg.inv(self.cov_prior) @ self.mu_prior + self.beta * phi.T @ y)
        
    # 予測分布の計算
    def predict(self, phi_pred):
        
        # p155 式(3.58) 
        pred_mean = phi_pred @ self.mu_posterior
        # p155 式(3.59)
        pred_cov = 1 / self.beta + np.sum(phi_pred @ self.cov_posterior * phi_pred, axis=1)
        return pred_mean, pred_cov
    
    # priorを一回前に予測された事後確率に更新
    def update_prior(self):
        self.cov_prior = self.cov_posterior
        self.mu_prior = self.mu_posterior
        
    # beta_priorを更新する
    def update_beta(self, phi, y):
        # numpy.linalg.eigを使って固有値と固有ベクトルを計算する(式3.87)
        eigenvalues, eigenvectors = np.linalg.eig(self.beta * phi.T @ phi)

        # p168 式(3.98)
        gamma = np.sum(eigenvalues / eigenvalues + self.alpha)
        # βの更新 p168 式(3.95)
        beta_new = 1 / (1 / (len(y) - gamma) * np.sum((y - phi @ self.mu_posterior)**2))
        self.beta = beta_new