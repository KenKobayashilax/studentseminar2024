import matplotlib.pyplot as plt

def plot_figure(x, y, x_pred, phi_pred, true_params, pred_mean, pred_std, title='Bayesian Linear Regression'):
    # 可視化
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, c='crimson', marker='o', label='Observation')
    plt.plot(x_pred, phi_pred @ true_params, label='True Function', color='blue')
    plt.plot(x_pred, pred_mean, label='Predicted Mean', color='green')
    plt.fill_between(x_pred, pred_mean - 2 * pred_std, pred_mean + 2 * pred_std, color='lightgreen', alpha=0.3, label='Uncertainty')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()