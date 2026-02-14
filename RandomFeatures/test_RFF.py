import numpy as np
import pytest
from RFF import (
    random_rbf_features,
    random_laplacian_features,
    random_matern_features,
    random_polynomial_features,
    get_rbf_feature_transformer,
    get_laplacian_feature_transformer,
    get_matern_feature_transformer,
    get_polynomial_feature_transformer,
)

@pytest.fixture
def sample_data():
    return np.random.rand(10, 5)

def test_rbf_features_shape(sample_data):
    X = sample_data
    m = 100
    sigma = 1.0
    Z = random_rbf_features(X, m, sigma)
    assert Z.shape == (X.shape[0], m)

def test_laplacian_features_shape(sample_data):
    X = sample_data
    m = 100
    gamma = 1.0
    Z = random_laplacian_features(X, m, gamma)
    assert Z.shape == (X.shape[0], m)

def test_matern_features_shape(sample_data):
    X = sample_data
    m = 100
    nu = 1.5
    length_scale = 1.0
    Z = random_matern_features(X, m, nu, length_scale)
    assert Z.shape == (X.shape[0], m)

def test_polynomial_features_shape(sample_data):
    X = sample_data
    m = 100
    degree = 2
    c = 1
    Z = random_polynomial_features(X, m, degree, c)
    assert Z.shape == (X.shape[0], m)

def test_rbf_features_values(sample_data):
    X = sample_data
    m = 100
    sigma = 1.0
    Z = random_rbf_features(X, m, sigma)
    assert np.all(Z >= -np.sqrt(2.0 / m))
    assert np.all(Z <= np.sqrt(2.0 / m))

def test_laplacian_features_values(sample_data):
    X = sample_data
    m = 100
    gamma = 1.0
    Z = random_laplacian_features(X, m, gamma)
    assert np.all(Z >= -np.sqrt(2.0 / m))
    assert np.all(Z <= np.sqrt(2.0 / m))

def test_matern_features_values(sample_data):
    X = sample_data
    m = 100
    nu = 1.5
    length_scale = 1.0
    Z = random_matern_features(X, m, nu, length_scale)
    assert np.all(Z >= -np.sqrt(2.0 / m))
    assert np.all(Z <= np.sqrt(2.0 / m))

def test_rbf_transformer_shape(sample_data):
    X = sample_data
    n, d = X.shape
    m = 100
    sigma = 1.0
    transformer = get_rbf_feature_transformer(m, d, sigma)
    Z = transformer(X)
    assert Z.shape == (n, m)

def test_laplacian_transformer_shape(sample_data):
    X = sample_data
    n, d = X.shape
    m = 100
    gamma = 1.0
    transformer = get_laplacian_feature_transformer(m, d, gamma)
    Z = transformer(X)
    assert Z.shape == (n, m)

def test_matern_transformer_shape(sample_data):
    X = sample_data
    n, d = X.shape
    m = 100
    nu = 1.5
    length_scale = 1.0
    transformer = get_matern_feature_transformer(m, d, nu, length_scale)
    Z = transformer(X)
    assert Z.shape == (n, m)

def test_polynomial_transformer_shape(sample_data):
    X = sample_data
    n, d = X.shape
    m = 100
    degree = 2
    c = 1
    transformer = get_polynomial_feature_transformer(m, d, degree, c)
    Z = transformer(X)
    assert Z.shape == (n, m)
