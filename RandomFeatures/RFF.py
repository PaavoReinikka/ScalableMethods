import numpy as np

def random_rbf_features(X, m, sigma):
    """
    X: Data matrix (n x d)
    m: Number of features
    sigma: Kernel bandwidth
    """
    n, d = X.shape
    
    # 1. Sample frequencies from Fourier transform of RBF
    W = np.random.normal(
        loc=0.0,
        scale=1.0/sigma,
        size=(m, d)
    )
    
    # 2. Sample random phases
    b = np.random.uniform(0, 2*np.pi, size=m)
    
    # 3. Compute feature matrix Z (n x m)
    Z = np.sqrt(2.0/m) * np.cos(X @ W.T + b)
    
    return Z

def random_laplacian_features(X, m, gamma):
    """
    X: Data matrix (n x d)
    m: Number of features
    gamma: Kernel parameter
    """
    n, d = X.shape
    
    # Frequencies from Cauchy distribution
    W = np.random.standard_cauchy(size=(m, d)) * gamma
    
    b = np.random.uniform(0, 2 * np.pi, size=m)
    
    Z = np.sqrt(2.0/m) * np.cos(X @ W.T + b)
    
    return Z

def random_matern_features(X, m, nu, length_scale):
    """
    X: Data matrix (n x d)
    m: Number of features
    nu: Smoothness parameter
    length_scale: Length scale
    """
    n, d = X.shape
    
    # Student's t-distribution for Matern kernel
    df = 2 * nu
    t_dist = np.random.standard_t(df=df, size=(m, d))
    
    W = t_dist / (length_scale * np.sqrt(df))
    
    b = np.random.uniform(0, 2*np.pi, size=m)
    
    Z = np.sqrt(2.0/m) * np.cos(X @ W.T + b)
    
    return Z

def random_polynomial_features(X, m, degree, c):
    """
    X: Data matrix (n x d)
    m: Number of features
    degree: Degree of the polynomial
    c: Constant term
    """
    n, d = X.shape
    
    # Simplified version using random projections
    W = np.random.randn(m, d)
    
    # Apply polynomial transformation
    Z = (X @ W.T + c) ** degree
    
    return Z

def get_rbf_feature_transformer(m, d, sigma):
    """
    m: Number of features
    d: Data dimension
    sigma: Kernel bandwidth
    """
    W = np.random.normal(loc=0.0, scale=1.0/sigma, size=(m, d))
    b = np.random.uniform(0, 2*np.pi, size=m)

    def transformer(X):
        return np.sqrt(2.0/m) * np.cos(X @ W.T + b)

    return transformer

def get_laplacian_feature_transformer(m, d, gamma):
    """
    m: Number of features
    d: Data dimension
    gamma: Kernel parameter
    """
    W = np.random.standard_cauchy(size=(m, d)) * gamma
    b = np.random.uniform(0, 2 * np.pi, size=m)

    def transformer(X):
        return np.sqrt(2.0/m) * np.cos(X @ W.T + b)

    return transformer

def get_matern_feature_transformer(m, d, nu, length_scale):
    """
    m: Number of features
    d: Data dimension
    nu: Smoothness parameter
    length_scale: Length scale
    """
    df = 2 * nu
    t_dist = np.random.standard_t(df=df, size=(m, d))
    W = t_dist / (length_scale * np.sqrt(df))
    b = np.random.uniform(0, 2*np.pi, size=m)

    def transformer(X):
        return np.sqrt(2.0/m) * np.cos(X @ W.T + b)

    return transformer

def get_polynomial_feature_transformer(m, d, degree, c):
    """
    m: Number of features
    d: Data dimension
    degree: Degree of the polynomial
    c: Constant term
    """
    W = np.random.randn(m, d)

    def transformer(X):
        return (X @ W.T + c) ** degree

    return transformer
