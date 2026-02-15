# Random Fourier Features for Kernel Approximation and scalable Spectral methods.

**NOTE:** This project and repository is still very much evolving. The main focus will eventually be on scalable spectral methods (once I regain my mathematical footing...).

## 1. Introduction to Kernel Methods

Kernel methods are a powerful class of algorithms in machine learning that allow us to work with data in a higher-dimensional feature space without explicitly computing the coordinates of the data in that space. This is achieved by using a **kernel function**, which computes the inner product of the data points in the feature space.

A common example is the **Radial Basis Function (RBF) kernel**, which is defined as:

$$k(x, y) = \exp\left(-\frac{\|x-y\|^2}{2\sigma^2}\right)$$

where $\sigma$ is a bandwidth parameter.

While powerful, kernel methods suffer from a major drawback: they require the computation and storage of the **kernel matrix** (or Gram matrix) $K \in \mathbb{R}^{n \times n}$, where $n$ is the number of data points. This leads to a time complexity of at least $O(n^2)$ and a space complexity of $O(n^2)$. Furthermore, many methods require solving either eigendecompositions or inverses of the relevant matrices, which is prohibitive for large datasets.

## 2. Random Fourier Features

**Random Fourier Features (RFF)** provide a way to approximate the kernel matrix, allowing us to scale kernel methods to large datasets. The key idea behind RFF is to approximate the kernel function with an inner product of finite-dimensional feature vectors:

$$k(x, y) \approx z(x)^T z(y)$$

where $z(x) \in \mathbb{R}^m$ and $m \ll n$. This way, we can work with the feature matrix $Z \in \mathbb{R}^{n \times m}$ instead of the full kernel matrix $K$.

### 2.1. The Theory: Bochner's Theorem

The theoretical foundation for RFF is **Bochner's Theorem**, which states that a continuous and real-valued kernel function $k(x, y)$ is positive definite if and only if it is the Fourier transform of a non-negative measure. For a shift-invariant kernel $k(x, y) = k(x-y)$, we can write:

$$k(\delta) = \int p(\omega) e^{i\omega^T \delta} d\omega$$

where $p(\omega)$ is a probability distribution. The RFF method approximates this integral with a Monte Carlo simulation.

### 2.2. The Feature Map

For a given kernel, we can construct the feature map $z(x)$ by sampling frequencies $\omega$ from the Fourier transform of the kernel, $p(\omega)$. The feature map is then defined as:

$$z(x) = \sqrt{\frac{2}{m}} \left[ \cos(\omega_1^T x + b_1), \dots, \cos(\omega_m^T x + b_m) \right]$$

where $b_j$ are sampled from a uniform distribution over $[0, 2\pi]$.

## 3. Implemented Kernels and their Fourier Transforms

This project provides implementations for several common kernels and their corresponding Random Fourier Features.

### 3.1. RBF (Gaussian) Kernel

-   **Kernel:** $k(x, y) = \exp\left(-\frac{\|x-y\|^2}{2\sigma^2}\right)$
-   **Fourier Transform:** A Gaussian distribution $N(0, \sigma^{-2}I)$.

### 3.2. Laplacian Kernel

-   **Kernel:** $k(x, y) = \exp(-\gamma \|x-y\|_1)$
-   **Fourier Transform:** A Cauchy distribution.

### 3.3. Matern Kernel

-   **Kernel:** A family of kernels that are solutions to a stochastic differential equation. The smoothness is controlled by a parameter $\nu$.
-   **Fourier Transform:** A Student's t-distribution.

### 3.4. Polynomial Kernel

-   **Kernel:** $k(x, y) = (x^T y + c)^d$
-   **Approximation:** While not a shift-invariant kernel, it can be approximated using methods like "Random Maclaurin" features or "Tensor Sketch". This project includes a simplified version using random projections.

## 4. The Importance of a Consistent Feature Transformer

An important aspect of using Random Fourier Features in experiments is to ensure that the *same* feature transformation is applied to both the training and test data. This means that the random frequencies $\omega$ and phases $b$ must be generated only once and then used to transform both datasets.

In this project, we use "transformer" closures (e.g., `get_rbf_feature_transformer`) that capture the freq. and phase resolution/span. The returned function can then be used to apply the same transformation to any data.

## 5. Further Considerations: Spectral Clustering with Random Fourier Features

Spectral clustering is a powerful clustering technique that uses the eigenvalues and eigenvectors of a similarity matrix to partition data into clusters. The similarity matrix is often constructed using a kernel function, such as the RBF kernel.

### 5.1. The Computational Bottleneck in Spectral Clustering

The standard spectral clustering algorithm involves the following steps:

1.  **Construct a similarity matrix** $K \in \mathbb{R}^{n \times n}$ from the data.
2.  **Construct the graph Laplacian** $L$ from the similarity matrix.
3.  **Compute the eigenvectors** of the graph Laplacian.
4.  **Cluster the data** using the eigenvectors.

The main bottleneck in this process is the construction and storage of the similarity matrix and the computation of its eigenvectors, which have a time complexity of $O(n^3)$ and a space complexity of $O(n^2)$. This makes spectral clustering infeasible for large datasets.

### 5.2. Scaling Spectral Clustering with RFFs

Random Fourier Features can be used to scale spectral clustering to large datasets by approximating the kernel matrix. By using the feature matrix $Z \in \mathbb{R}^{n \times m}$ instead of the full kernel matrix $K$, we can reformulate the spectral clustering problem in terms of $Z$.

The graph Laplacian can be approximated as $L \approx D - ZZ^T$, where $D$ is the degree matrix. The eigenvectors of this approximate Laplacian can be computed much more efficiently, often by solving a smaller $m \times m$ eigenproblem. This reduces the time complexity to be in the order of $O(nm^2)$ and the space complexity to $O(nm)$, which is a significant improvement when $m \ll n$.

### 5.3. Low-Rank Approximation and the Eigendecomposition Trick

The power of using RFFs in spectral clustering comes from a computational trick involving low-rank matrices. Since the rank of the feature matrix $Z$ is at most $m$, the kernel approximation $K \approx ZZ^T$ is also a low-rank approximation.

For the normalized Laplacian $L_{sym} = I - D^{-1/2} K D^{-1/2}$, we can substitute the RFF approximation of $K$:

$$L_{sym} \approx I - D^{-1/2} ZZ^T D^{-1/2} = I - \hat{Z}\hat{Z}^T$$

where $\hat{Z} = D^{-1/2} Z$ is the normalized feature matrix.

The eigenvectors of $L_{sym}$ that are of interest correspond to the smallest eigenvalues; these match the dominant eigenvectors of $\hat{Z}\hat{Z}^T$.

The key insight is that instead of computing the eigenvectors of the $n \times n$ matrix $\hat{Z}\hat{Z}^T$, we can solve the much smaller $m \times m$ eigenproblem:

$$\hat{Z}^T \hat{Z} v = \sigma v$$

The eigenvectors $u$ of $\hat{Z}\hat{Z}^T$ can then be recovered from the eigenvectors $v$ of $\hat{Z}^T \hat{Z}$ by the relation $u = \hat{Z}v$ (think of SVD). This reduces the complexity of the eigendecomposition from $O(n^3)$ to $O(m^3)$, which is a dramatic improvement when $m \ll n$.


## 6 Continue...

Here be more on:
* kernel Ridge regression.
* Iterative methods for eigen systems.
* Nyström method.
* Deflation method.
* Various Laplacians.


## Reference

1.  Rahimi, A., & Recht, B. (2007). Random features for large-scale kernel machines. In *Advances in neural information processing systems*.
2.  Sutherland, D. J., & Schneider, J. (2015). On the error of random Fourier features. In *Uncertainty in Artificial Intelligence*.


For a comprehensive introduction to spectral clustering, see:

-   von Luxburg, U. (2007). A tutorial on spectral clustering. *Statistics and computing*, *17*(4), 395-416.
