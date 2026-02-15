
# Random Features, Nyström, and Large-Scale Spectral Methods

## 1. Motivation

Spectral clustering, diffusion maps, kernel PCA, and related methods all rely on eigen-analysis of a kernel matrix or a normalized graph Laplacian. For $n$ data points, constructing the full kernel matrix $W \in \mathbb{R}^{n \times n}$ costs:

* Memory: $O(n^2)$
* Eigendecomposition: up to $O(n^3)$

This becomes infeasible for large $n$. The core challenge is:

> How do we approximate the spectral structure of a kernel operator without ever forming the full $n \times n$ matrix?

Two major families of solutions are:

* **Random Fourier Features (RFF)**
* **Nyström methods**

They approximate different objects, and understanding that distinction clarifies how to combine them effectively.

---

# 2. Operator View of Spectral Methods

Given a positive definite kernel $k(x,y)$, for example the Gaussian kernel,

$$
k(x,y) = \exp\left(-\frac{|x-y|^2}{2\sigma^2}\right),
$$

there is an associated integral operator:

$$
(Tf)(x) = \int k(x,y) f(y) , d\mu(y).
$$

All of the following methods compute eigenfunctions of this operator (or a normalized variant):

* Spectral clustering
* Diffusion maps
* Kernel PCA
* Normalized Laplacian embeddings

The graph matrix $W$ is simply a discretization of $T$.

Thus large-scale spectral learning reduces to:

> Approximate leading eigenfunctions of a kernel integral operator efficiently.

---

# 3. Random Fourier Features (RFF)

## 3.1 What RFF Approximates

For shift-invariant kernels, Bochner’s theorem implies:

$$
k(x,y) = \mathbb{E}_{\omega} \left[ e^{i \omega^\top (x-y)} \right].
$$

RFF approximates this expectation via Monte Carlo sampling:

$$
k(x,y) \approx z(x)^\top z(y),
$$

where $z(x) \in \mathbb{R}^D$.

Thus:

$$
W \approx Z Z^\top,
$$

with $Z \in \mathbb{R}^{n \times D}$.

### Key Property

RFF approximates the **kernel function itself**, producing a global rank-$D$ surrogate of the kernel operator.

The associated operator becomes:

$$
\tilde T f(x)
= \int z(x)^\top z(y) f(y) d\mu(y),
$$

which has rank at most $D$.

---

## 3.2 Symmetric Normalized Laplacian with RFF

The symmetric normalized Laplacian is:

$$
L_{\text{sym}} = I - D^{-1/2} W D^{-1/2},
$$

where:

$$
d_i = \sum_j W_{ij}.
$$

With ( W \approx Z Z^\top ):

$$
d_i = z_i^\top \left(\sum_j z_j \right).
$$

Define:

$$
s = \sum_{j=1}^n z_j.
$$

Then:

$$
d_i = z_i^\top s.
$$

Define normalized features:

$$
\tilde z_i = \frac{z_i}{\sqrt{d_i}}.
$$

Then:

$$
D^{-1/2} W D^{-1/2}
\approx \tilde Z \tilde Z^\top.
$$

So spectral clustering reduces to eigen-decomposition of:

$$
\tilde Z^\top \tilde Z \in \mathbb{R}^{D \times D}.
$$

### Computational Cost

* Feature construction: $O(nD)$
* Covariance build: $O(nD^2)$
* Eigendecomposition: $O(D^3)$

No $n \times n$ matrix is ever formed.

---

# 4. Nyström Method

## 4.1 What Nyström Approximates

Nyström selects $m \ll n$ landmark points and constructs:

$$
K \approx C W_m^{-1} C^\top,
$$

where:

* $W_m \in \mathbb{R}^{m \times m}$ is kernel among landmarks
* $C \in \mathbb{R}^{n \times m}$ is cross-kernel matrix

Nyström approximates the **empirical kernel matrix**, not the kernel function itself.

Thus it approximates the empirical operator induced by the data.

---

## 4.2 Behavior in Spectral Clustering

Nyström approximates:

$$
W \approx \hat W_m,
$$

then computes eigenvectors from this low-rank surrogate.

Its quality depends on:

* Landmark selection
* Spectral decay of $W$
* Data geometry

Nyström adapts to the dataset.

---

# 5. RFF vs Nyström: Structural Differences

| Aspect          | RFF                       | Nyström                        |
| --------------- | ------------------------- | ------------------------------ |
| Approximates    | Kernel function           | Empirical kernel matrix        |
| Data dependence | No                        | Yes                            |
| Rank            | $D$                     | $m$                          |
| Error behavior  | $O(1/\sqrt{D})$ uniform | Depends on spectrum + sampling |
| Parallelizable  | Very                      | Moderate                       |

### Conceptual Distinction

* RFF compresses the harmonic structure of the kernel globally.
* Nyström compresses the empirical operator via landmark sampling.

---

# 6. Hybrid RFF + Nyström Methods

Hybrid approaches combine:

* **Kernel approximation via RFF**
* **Data-adaptive subspace selection via Nyström**

There are multiple strategies.

---

## 6.1 RFF Preconditioning + Nyström

Pipeline:

1. Compute RFF features $Z \in \mathbb{R}^{n \times D}$
2. Apply Nyström in feature space
3. Select landmark rows of $Z$
4. Build low-rank approximation of $Z Z^\top$

This reduces kernel computation cost while retaining data-adaptive compression.

---

## 6.2 Nyström on Random Feature Covariance

Instead of applying Nyström to $W$, apply it to:

$$
M = Z^\top Z.
$$

Since $M \in \mathbb{R}^{D \times D}$, one can:

* Select feature subsets
* Approximate covariance spectrum
* Further reduce rank

This creates a two-stage compression:

1. Function-level approximation (RFF)
2. Data-adaptive subspace refinement (Nyström)

---

## 6.3 Landmark-Guided Random Features

A more refined hybrid:

1. Use Nyström to identify informative regions of the data manifold.
2. Bias RFF sampling toward frequencies that matter for that region.

This moves RFF from uniform Monte Carlo sampling to geometry-aware sampling.

---

# 7. Error Geometry in Hybrid Methods

Let:

$$
T = \text{true operator},
\quad
\tilde T_{\text{RFF}} = \text{RFF approximation},
\quad
\tilde T_{\text{Hybrid}} = \text{Hybrid approximation}.
$$

We want:

$$
|T - \tilde T_{\text{Hybrid}}| < \text{spectral gap}.
$$

Hybrid methods can reduce operator norm error faster than pure RFF when:

* Spectrum decays quickly
* Data manifold is low-dimensional
* Landmark selection captures cluster structure

Thus hybrids can achieve:

* Smaller rank than pure RFF
* Better stability than pure Nyström

---

# 8. Interpretation for Normalized Laplacians

Recall:

$$
L_{\text{sym}} = I - D^{-1/2} W D^{-1/2}.
$$

Hybrid approach:

1. Approximate $W$ via RFF
2. Compute approximate degrees
3. Form normalized feature matrix
4. Apply Nyström-style compression to normalized features

This yields a low-rank approximation of the **diffusion operator**, not just the raw kernel.

Effectively:

> First approximate the kernel operator harmonically.
> Then compress its dominant empirical modes geometrically.

---

# 9. Practical Guidance

### When to Use RFF Alone

* Extremely large $n$
* Need streaming or distributed computation
* Kernel is shift-invariant
* Moderate spectral decay

### When to Use Nyström Alone

* Kernel evaluations cheap
* Good landmark selection possible
* Fast spectral decay

### When to Use Hybrid

* Very large $n$
* Strong manifold structure
* Need aggressive rank reduction
* Want better approximation per dimension

---

# 10. Big Picture

Spectral clustering, diffusion maps, and kernel PCA are all eigenproblems of kernel operators.

Large-scale approximations fall into two philosophies:

* **Function-level approximation**: Random Fourier Features
* **Operator-level approximation**: Nyström

Hybrid methods combine both:

> First approximate the law of interaction.
> Then approximate the empirical geometry.

This layered compression provides scalable access to diffusion geometry and normalized Laplacian embeddings without constructing dense $n \times n$ graphs.

---

## Summary

* RFF replaces the kernel with an explicit finite-dimensional feature map.
* Nyström approximates the empirical kernel matrix via landmark sampling.
* For symmetric normalized Laplacians, RFF allows degree computation and normalization without ever forming $W$.
* Hybrid methods combine harmonic approximation with data-adaptive compression.
* Stability depends on maintaining operator perturbation below the spectral gap.

These tools transform spectral methods from quadratic-memory algorithms into scalable operator approximations suitable for modern large-scale learning.
