import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from RFF import get_rbf_feature_transformer

def test_rff_application():
    # 1. Generate synthetic data
    #np.random.seed(0)
    X = np.sort(5 * np.random.rand(40, 1), axis=0)
    y = np.sin(X).ravel()
    y += 0.2 * np.random.randn(len(y))

    # 2. Train a Kernel Ridge Regression model
    kr = KernelRidge(kernel='rbf', alpha=0.1, gamma=1.0)
    kr.fit(X, y)

    # 3. Create a RFF transformer
    m = 1000
    n, d = X.shape
    rbf_transformer = get_rbf_feature_transformer(m=m, d=d, sigma=1.0)

    # 4. Transform the data
    Z = rbf_transformer(X)

    # 5. Train a Ridge Regression model with Random Fourier Features
    ridge = Ridge(alpha=0.1)
    ridge.fit(Z, y)

    # 6. Generate test data
    X_plot = np.linspace(0, 5, 100)[:, None]
    y_kr = kr.predict(X_plot)
    Z_plot = rbf_transformer(X_plot)
    y_rff = ridge.predict(Z_plot)

    # 7. Visualize the results
    plt.figure(figsize=(10, 5))
    plt.plot(X_plot, np.sin(X_plot), color='navy', lw=2, label='True function')
    plt.scatter(X, y, color='turquoise', label='Data')
    plt.plot(X_plot, y_kr, color='red', lw=2, label='Kernel Ridge')
    plt.plot(X_plot, y_rff, color='orange', lw=2, label='RFF Ridge')
    plt.legend()
    plt.title('Kernel Ridge vs. RFF Ridge (Fixed Transformer)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('./figures/rff_application_test.png')
    plt.close()

    # 8. Compare the performance
    mse_kr = np.mean((kr.predict(X) - y) ** 2)
    mse_rff = np.mean((ridge.predict(Z) - y) ** 2)
    
    print(f"MSE (Kernel Ridge): {mse_kr}")
    print(f"MSE (RFF Ridge): {mse_rff}")
    
    assert np.isclose(mse_kr, mse_rff, atol=0.1)

if __name__ == "__main__":
    test_rff_application()
