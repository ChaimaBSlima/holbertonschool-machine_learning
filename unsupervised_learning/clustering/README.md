<h1><p align="center"> Clustering </h1></p></font>

<p align="center">
  <img src="" alt="Image"/>
</p>

# 📚 Resources

Read or watch:

- [Understanding K-means Clustering in Machine Learning](https://medium.com/data-science/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1)
- [K-means clustering: how it works](https://www.youtube.com/watch?v=_aWzGGNrcic)
- [How many clusters?](https://www.youtube.com/watch?v=xNfOheh-res)
- [Bimodal distribution](hhttps://www.youtube.com/watch?v=BWItfiVnDfU)
- [Gaussian Mixture Model](https://intranet.hbtn.io/rltoken/wdP_gXMOYcwdrZf2tUAbcA)
- [EM algorithm: how it works](https://www.youtube.com/watch?v=REypj2sy_5U)
- [Expectation Maximization: how it works](https://www.youtube.com/watch?v=iQoXFmbXRJA)
- [Mixture Models 4: multivariate Gaussians](https://www.youtube.com/watch?v=zL_MHtT56S0)
- [Mixture Models 5: how many Gaussians?](https://www.youtube.com/watch?v=BWXd5dOkuTo)
- [Gaussian Mixture Model (GMM) using Expectation Maximization (EM) Technique](https://www.cse.iitm.ac.in/~vplab/courses/DVP/PDF/gmm.pdf)
- [What is Hierarchical Clustering?](https://www.displayr.com/what-is-hierarchical-clustering/)
- [Hierarchical Clustering](https://www.youtube.com/watch?v=rg2cjfMsCk4)

Definitions to skim:

- [Cluster analysis](https://en.wikipedia.org/wiki/Cluster_analysis)
- [K-means clustering](https://en.wikipedia.org/wiki/Cluster_analysis)
- [Multimodal distribution](https://en.wikipedia.org/wiki/Multimodal_distribution)
- [Mixture_model](https://en.wikipedia.org/wiki/Mixture_model)
- [Expectation–maximization algorithm](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm)
- [Hierarchical clustering](https://en.wikipedia.org/wiki/Hierarchical_clustering)
- [Ward’s method](https://en.wikipedia.org/wiki/Ward%27s_method)
- [Cophenetic](https://en.wikipedia.org/wiki/Cophenetic)

References:

- [scikit-learn](https://scikit-learn.org/stable/index.html)
- [Clustering](https://scikit-learn.org/stable/modules/clustering.html#clustering)
- [sklearn.cluster.KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)
- [Gaussian mixture models](https://scikit-learn.org/stable/modules/mixture.html#mixture)
- [sklearn.mixture.GaussianMixture](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture)
- [scipy](https://scipy.org/)
- [scipy.cluster.hierarchy](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html)
- [scipy.cluster.hierarchy.linkage](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage)
- [scipy.cluster.hierarchy.fcluster](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html#scipy.cluster.hierarchy.fcluster)
- [scipy.cluster.hierarchy.dendrogram](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html#scipy.cluster.hierarchy.dendrogram)

---

# 🎯 Learning Objectives

- What is a multimodal distribution?  
- What is a cluster?  
- What is cluster analysis?  
- What is “soft” vs “hard” clustering?  
- What is K-means clustering?  
- What are mixture models?  
- What is a Gaussian Mixture Model (GMM)?  
- What is the Expectation-Maximization (EM) algorithm?  
- How to implement the EM algorithm for GMMs  
- What is cluster variance?  
- What is the mountain/elbow method?  
- What is the Bayesian Information Criterion?  
- How to determine the correct number of clusters  
- What is Hierarchical clustering?  
- What is Agglomerative clustering?  
- What is Ward’s method?  
- What is Cophenetic distance?  
- What is `scikit-learn`?  
- What is `scipy`?

---

# 🧾 Requirements

### General

- Allowed editors: `vi`, `vim`, `emacs`  
- All your files will be interpreted/compiled on **Ubuntu 20.04 LTS** using **python3 (version 3.9)**  
- Your files will be executed with **numpy (version 1.25.2) sklearn (version 1.5.0), and scipy (version 1.11.4)**  
- All your files should end with a **new line**  
- The first line of all your files should be exactly: `#!/usr/bin/env python3`
- A `README.md` file, at the root of the folder of the project, is mandatory
- Your code should use the `pycodestyle` style (version 2.11.1)
- All your modules should have documentation (`python3 -c 'print(__import__("my_module").__doc__)'`)
- All your classes should have documentation (`python3 -c 'print(__import__("my_module").MyClass.__doc__)'`)
- All your functions (inside and outside a class) should have documentation (`python3 -c 'print(__import__("my_module").my_function.__doc__)'` and python3 -c `'print(__import__("my_module").MyClass.my_function.__doc__)'`)
- Unless otherwise noted, you are not allowed to import any module except `import numpy as np`
- All your files must be executable
- Your code should use the minimum number of operations

## Installing Scikit-Learn 1.5.0
```
pip install --user scikit-learn==1.5.0
```

## Installing Scipy 1.11.4
`scipy` should have already been installed with `matplotlib` and `numpy`, but just in case:
```
pip install --user scipy==1.11.4
```

---
# 📝 Tasks

### 0. Initialize K-means

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Write a function `def initialize(X, k):` that initializes cluster centroids for **K-means**:

- `X` is a **numpy.ndarray** of shape (n, d) containing the dataset that will be used for K-means clustering.
  - `n` is the number of data points.
  - `d` is the number of dimensions for each data point.
- `k` is a positive integer containing the number of clusters.

- The cluster centroids should be initialized with a **multivariate uniform distribution** along each dimension in `d`.
  - The **minimum values** for the distribution should be the minimum values of `X` along each dimension in `d`.
  - The **maximum values** for the distribution should be the maximum values of `X` along each dimension in `d`.
- You should use `numpy.random.uniform` exactly **once**.
- **No loops** allowed.

#### Returns:
- A **numpy.ndarray** of shape (k, d) containing the initialized centroids for each cluster.
- Return `None` on failure.
```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/unsupervised_learning/hmm#./test_files/0-main.py
```
<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 1. K-means

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Write a function `def kmeans(X, k, iterations=1000):` that performs **K-means** on a dataset:

- `X` is a **numpy.ndarray** of shape (n, d) containing the dataset.
  - `n` is the number of data points.
  - `d` is the number of dimensions for each data point.
- `k` is a positive integer containing the number of clusters.
- `iterations` is a positive integer containing the maximum number of iterations that should be performed.

- If no change in the cluster centroids occurs between iterations, your function should return.
- Initialize the cluster centroids using a **multivariate uniform distribution** (based on `0-initialize.py`).
- If a cluster contains no data points during the update step, **reinitialize its centroid**.
- You should use `numpy.random.uniform` exactly **twice**.
- You may use at most **2 loops**.

#### Returns:
- **C**: A **numpy.ndarray** of shape (k, d) containing the centroid means for each cluster.
- **clss**: A **numpy.ndarray** of shape (n,) containing the index of the cluster in **C** that each data point belongs to.

- On failure, return `None, None`.
```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/unsupervised_learning/hmm#./test_files/1-main.py
[[ 9.92511389 25.73098987]
 [30.06722465 40.41123947]
 [39.62770705 19.89843487]
 [59.22766628 29.19796006]
 [20.0835633  69.81592298]]
```

<p align="center">⭐⭐⭐⭐⭐⭐</p>


### 2. Variance

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Write a function `def variance(X, C):` that calculates the **total intra-cluster variance** for a data set:

- `X` is a **numpy.ndarray** of shape (n, d) containing the dataset.
  - `n` is the number of data points.
  - `d` is the number of dimensions for each data point.
- `C` is a **numpy.ndarray** of shape (k, d) containing the centroid means for each cluster.
  - `k` is the number of clusters.

- You are not allowed to use any loops in this function.

#### Returns:
- **var**: The total variance.
- Return `None` on failure.
```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/unsupervised_learning/hmm#./test_files/2-main.py
Variance with 1 clusters: 157927.7052
Variance with 2 clusters: 82095.68297
Variance with 3 clusters: 34784.23723
Variance with 4 clusters: 23158.40095
Variance with 5 clusters: 7868.52123
Variance with 6 clusters: 7406.93077
Variance with 7 clusters: 6930.66361
Variance with 8 clusters: 6162.15884
Variance with 9 clusters: 5843.92455
Variance with 10 clusters: 5727.41124
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/unsupervised_learning/hmm#
```

<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 3. Optimize k

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)   


Write a function `def optimum_k(X, kmin=1, kmax=None, iterations=1000):` that tests for the **optimum number of clusters by variance**:

- `X` is a **numpy.ndarray** of shape (n, d) containing the dataset.
  - `n` is the number of data points.
  - `d` is the number of dimensions for each data point.
- `kmin` is a positive integer containing the **minimum number of clusters** to check for (inclusive).
- `kmax` is a positive integer containing the **maximum number of clusters** to check for (inclusive).
- `iterations` is a positive integer containing the **maximum number of iterations** for K-means.

- This function must analyze **at least 2 different cluster sizes**.
- You must use:
  - `kmeans = __import__('1-kmeans').kmeans`
  - `variance = __import__('2-variance').variance`
- You may use **at most 2 loops**.

#### Returns:
- **results**: A list containing the outputs of K-means for each cluster size.
- **d_vars**: A list containing the difference in variance from the smallest cluster size for each cluster size.
- Return `None, None` on failure.

```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/unsupervised_learning/hmm#./test_files/3-main.py
[(array([[31.78625503, 37.01090945]]), array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0])), (array([[34.76990289, 28.71421162],
       [20.14417812, 69.38429903]]), array([0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0,
       1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
       0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0,
       0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0,
       0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0,
       1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1,
       0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
       1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
       0, 1, 0, 0, 0, 0, 0, 0])), (array([[49.55185774, 24.76080087],
       [20.0835633 , 69.81592298],
       [19.8719982 , 32.85851127]]), array([2, 2, 1, 2, 0, 2, 2, 0, 1, 0, 0, 2, 1, 0, 1, 2, 2, 0, 0, 1, 2, 2,
       1, 0, 2, 2, 0, 0, 2, 0, 0, 2, 1, 2, 0, 2, 0, 2, 0, 1, 1, 0, 2, 2,
       0, 2, 2, 2, 1, 0, 0, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 0, 0,
       2, 0, 1, 0, 1, 0, 2, 2, 2, 2, 1, 2, 2, 1, 0, 2, 0, 1, 2, 0, 2, 2,
       0, 1, 2, 2, 0, 2, 2, 1, 2, 1, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 0, 0,
       2, 2, 0, 1, 0, 2, 2, 1, 2, 2, 0, 0, 2, 1, 0, 1, 1, 0, 1, 2, 2, 2,
       0, 1, 0, 2, 2, 0, 2, 0, 2, 0, 1, 0, 1, 1, 0, 2, 2, 1, 0, 2, 0, 0,
       1, 1, 1, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 2, 1, 1, 1, 1, 2, 0, 0, 1,
       2, 0, 0, 2, 0, 0, 2, 1, 2, 1, 2, 0, 1, 1, 0, 0, 0, 2, 0, 0, 0, 0,
       2, 2, 0, 0, 0, 0, 0, 1, 0, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2,
       1, 0, 0, 2, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0,
       2, 1, 0, 2, 2, 0, 2, 0])), (array([[39.57566544, 20.48452248],
       [20.0835633 , 69.81592298],
       [19.62313956, 33.02895961],
       [59.22766628, 29.19796006]]), array([2, 2, 1, 2, 3, 2, 2, 0, 1, 0, 0, 2, 1, 0, 1, 2, 0, 3, 0, 1, 2, 2,
       1, 3, 2, 2, 0, 0, 2, 3, 3, 2, 1, 2, 0, 2, 3, 2, 3, 1, 1, 0, 2, 2,
       0, 2, 2, 2, 1, 3, 3, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 3, 3,
       2, 0, 1, 3, 1, 0, 2, 2, 2, 2, 1, 2, 2, 1, 3, 2, 3, 1, 2, 0, 2, 2,
       3, 1, 2, 2, 3, 2, 2, 1, 2, 1, 0, 3, 2, 2, 3, 0, 0, 2, 2, 0, 3, 0,
       2, 2, 0, 1, 0, 2, 2, 1, 2, 2, 0, 0, 2, 1, 0, 1, 1, 3, 1, 2, 2, 2,
       3, 1, 3, 2, 2, 0, 2, 3, 2, 3, 1, 3, 1, 1, 3, 2, 2, 1, 3, 2, 3, 0,
       1, 1, 1, 3, 3, 0, 2, 2, 2, 0, 0, 3, 0, 2, 1, 1, 1, 1, 2, 0, 3, 1,
       2, 3, 3, 2, 0, 0, 2, 1, 2, 1, 0, 0, 1, 1, 3, 0, 3, 2, 0, 0, 3, 3,
       2, 2, 3, 0, 0, 0, 3, 1, 3, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2,
       1, 0, 0, 2, 1, 0, 3, 0, 3, 1, 1, 0, 0, 3, 0, 3, 0, 2, 2, 3, 3, 3,
       2, 1, 0, 2, 2, 0, 2, 0])), (array([[30.06722465, 40.41123947],
       [59.22766628, 29.19796006],
       [ 9.92511389, 25.73098987],
       [20.0835633 , 69.81592298],
       [39.62770705, 19.89843487]]), array([0, 0, 3, 0, 1, 2, 0, 4, 3, 4, 4, 0, 3, 4, 3, 2, 4, 1, 4, 3, 2, 0,
       3, 1, 0, 2, 4, 4, 2, 1, 1, 0, 3, 2, 4, 0, 1, 0, 1, 3, 3, 4, 0, 2,
       4, 0, 0, 0, 3, 1, 1, 3, 0, 3, 0, 2, 0, 2, 0, 0, 2, 2, 1, 2, 1, 1,
       0, 4, 3, 1, 3, 4, 0, 0, 2, 0, 3, 2, 0, 3, 1, 0, 1, 3, 2, 4, 2, 0,
       1, 3, 0, 2, 1, 0, 0, 3, 2, 3, 4, 1, 0, 2, 1, 4, 4, 0, 2, 4, 1, 4,
       2, 2, 4, 3, 4, 2, 2, 3, 2, 2, 4, 4, 0, 3, 0, 3, 3, 1, 3, 2, 0, 2,
       1, 3, 1, 2, 0, 4, 0, 1, 0, 1, 3, 1, 3, 3, 1, 2, 2, 3, 1, 2, 1, 4,
       3, 3, 3, 1, 1, 4, 2, 2, 0, 4, 4, 1, 4, 2, 3, 3, 3, 3, 0, 4, 1, 3,
       2, 1, 1, 2, 4, 4, 2, 3, 2, 3, 4, 4, 3, 3, 1, 4, 1, 0, 4, 4, 1, 1,
       2, 0, 1, 4, 4, 0, 1, 3, 1, 2, 0, 1, 0, 2, 0, 2, 2, 0, 0, 2, 3, 0,
       3, 4, 4, 2, 3, 4, 1, 4, 1, 3, 3, 4, 4, 1, 4, 1, 4, 0, 2, 1, 1, 1,
       2, 3, 4, 0, 2, 4, 2, 4])), (array([[44.18492017, 16.98881789],
       [30.06722465, 40.41123947],
       [38.18858711, 20.81726128],
       [59.22766628, 29.19796006],
       [20.0835633 , 69.81592298],
       [ 9.92511389, 25.73098987]]), array([1, 1, 4, 1, 3, 5, 1, 2, 4, 0, 2, 1, 4, 2, 4, 5, 2, 3, 2, 4, 5, 1,
       4, 3, 1, 5, 2, 2, 5, 3, 3, 1, 4, 5, 2, 1, 3, 1, 3, 4, 4, 2, 1, 5,
       0, 1, 1, 1, 4, 3, 3, 4, 1, 4, 1, 5, 1, 5, 1, 1, 5, 5, 3, 5, 3, 3,
       1, 2, 4, 3, 4, 2, 1, 1, 5, 1, 4, 5, 1, 4, 3, 1, 3, 4, 5, 2, 5, 1,
       3, 4, 1, 5, 3, 1, 1, 4, 5, 4, 2, 3, 1, 5, 3, 2, 2, 1, 5, 2, 3, 2,
       5, 5, 2, 4, 2, 5, 5, 4, 5, 5, 0, 2, 1, 4, 1, 4, 4, 3, 4, 5, 1, 5,
       3, 4, 3, 5, 1, 0, 1, 3, 1, 3, 4, 3, 4, 4, 3, 5, 5, 4, 3, 5, 3, 0,
       4, 4, 4, 3, 3, 0, 5, 5, 1, 0, 2, 3, 2, 5, 4, 4, 4, 4, 1, 0, 3, 4,
       5, 3, 3, 5, 2, 2, 5, 4, 5, 4, 2, 2, 4, 4, 3, 0, 3, 1, 0, 2, 3, 3,
       5, 1, 3, 2, 2, 1, 3, 4, 3, 5, 1, 3, 1, 5, 1, 5, 5, 1, 1, 5, 4, 1,
       4, 2, 2, 5, 4, 2, 3, 2, 3, 4, 4, 0, 2, 3, 0, 3, 2, 1, 5, 3, 3, 3,
       5, 4, 2, 1, 5, 2, 5, 2])), (array([[26.23935735, 39.56231098],
       [33.86397001, 36.21416257],
       [32.10246392, 43.52452575],
       [39.78587939, 19.72783208],
       [59.22766628, 29.19796006],
       [20.0835633 , 69.81592298],
       [ 9.92511389, 25.73098987]]), array([0, 1, 5, 2, 4, 6, 1, 3, 5, 3, 3, 2, 5, 3, 5, 6, 3, 4, 3, 5, 6, 0,
       5, 4, 1, 6, 3, 3, 6, 4, 4, 0, 5, 6, 3, 2, 4, 1, 4, 5, 5, 3, 2, 6,
       3, 0, 0, 1, 5, 4, 4, 5, 2, 5, 0, 6, 2, 6, 2, 1, 6, 6, 4, 6, 4, 4,
       0, 3, 5, 4, 5, 3, 0, 2, 6, 1, 5, 6, 2, 5, 4, 2, 4, 5, 6, 3, 6, 0,
       4, 5, 0, 6, 4, 2, 0, 5, 6, 5, 3, 4, 2, 6, 4, 3, 3, 0, 6, 3, 4, 3,
       6, 6, 3, 5, 3, 6, 6, 5, 6, 6, 3, 3, 0, 5, 1, 5, 5, 4, 5, 6, 0, 6,
       4, 5, 4, 6, 0, 3, 0, 4, 2, 4, 5, 4, 5, 5, 4, 6, 6, 5, 4, 6, 4, 3,
       5, 5, 5, 4, 4, 3, 6, 6, 2, 3, 3, 4, 3, 6, 5, 5, 5, 5, 0, 3, 4, 5,
       6, 4, 4, 6, 3, 3, 6, 5, 6, 5, 1, 3, 5, 5, 4, 3, 4, 2, 3, 3, 4, 4,
       6, 2, 4, 3, 3, 1, 4, 5, 4, 6, 1, 4, 0, 6, 0, 6, 6, 2, 2, 6, 5, 0,
       5, 3, 3, 6, 5, 3, 4, 3, 4, 5, 5, 3, 3, 4, 3, 4, 3, 0, 6, 4, 4, 4,
       6, 5, 3, 1, 6, 3, 6, 3])), (array([[46.45917139, 23.16813158],
       [17.46350686, 68.10438494],
       [59.60504708, 29.19922401],
       [32.37545229, 35.93762723],
       [38.66481538, 19.34921236],
       [28.96634723, 42.29059653],
       [ 9.92511389, 25.73098987],
       [24.01364794, 72.38323004]]), array([5, 3, 1, 5, 0, 6, 3, 4, 7, 4, 4, 5, 1, 4, 1, 6, 4, 2, 4, 1, 6, 5,
       7, 2, 3, 6, 4, 4, 6, 2, 2, 3, 7, 6, 4, 3, 2, 3, 2, 1, 7, 4, 5, 6,
       0, 5, 5, 3, 7, 2, 2, 1, 5, 7, 5, 6, 5, 6, 5, 3, 6, 6, 2, 6, 2, 2,
       5, 4, 1, 2, 1, 4, 5, 5, 6, 3, 7, 6, 5, 1, 2, 5, 2, 7, 6, 0, 6, 5,
       2, 1, 3, 6, 2, 5, 5, 7, 6, 7, 4, 2, 5, 6, 2, 4, 0, 5, 6, 4, 2, 4,
       6, 6, 4, 1, 4, 6, 6, 7, 6, 6, 0, 4, 5, 7, 3, 7, 1, 2, 1, 6, 3, 6,
       2, 1, 2, 6, 5, 4, 5, 2, 5, 2, 7, 2, 1, 7, 2, 6, 6, 1, 2, 6, 2, 4,
       7, 1, 1, 2, 2, 0, 6, 6, 5, 0, 4, 2, 0, 6, 7, 1, 1, 7, 5, 4, 2, 7,
       6, 2, 0, 6, 4, 4, 6, 1, 6, 1, 3, 4, 7, 1, 2, 4, 2, 5, 4, 4, 2, 2,
       6, 5, 2, 4, 4, 3, 2, 1, 2, 6, 3, 2, 3, 6, 5, 6, 6, 5, 5, 6, 1, 5,
       1, 4, 4, 6, 1, 4, 2, 4, 2, 1, 1, 0, 4, 2, 4, 2, 4, 5, 6, 2, 2, 2,
       6, 1, 4, 3, 6, 4, 6, 4])), (array([[41.41465903, 21.62570842],
       [37.61417321, 17.19733029],
       [26.6124459 , 43.91604964],
       [16.74238916, 69.20086704],
       [ 9.92511389, 25.73098987],
       [24.00581119, 70.53794517],
       [31.42335817, 38.75662838],
       [55.10816891, 27.75253127],
       [61.16625328, 29.87816185]]), array([6, 6, 3, 6, 7, 4, 6, 1, 5, 0, 1, 2, 3, 1, 3, 4, 1, 7, 0, 3, 4, 2,
       3, 7, 6, 4, 1, 0, 4, 8, 8, 6, 5, 4, 1, 6, 8, 6, 7, 3, 5, 0, 2, 4,
       0, 2, 6, 6, 5, 8, 7, 3, 2, 5, 6, 4, 6, 4, 6, 6, 4, 4, 8, 4, 8, 8,
       2, 1, 3, 8, 3, 0, 6, 6, 4, 6, 5, 4, 6, 3, 7, 6, 8, 5, 4, 0, 4, 2,
       7, 3, 6, 4, 7, 6, 6, 5, 4, 5, 0, 8, 6, 4, 8, 0, 0, 6, 4, 0, 8, 1,
       4, 4, 0, 3, 0, 4, 4, 5, 4, 4, 0, 0, 2, 5, 6, 5, 3, 8, 3, 4, 6, 4,
       8, 3, 7, 4, 2, 0, 2, 7, 2, 8, 5, 8, 3, 5, 8, 4, 4, 3, 7, 4, 7, 0,
       5, 3, 5, 8, 7, 0, 4, 4, 6, 0, 1, 8, 0, 4, 5, 3, 3, 5, 6, 1, 8, 5,
       4, 8, 7, 4, 1, 0, 4, 5, 4, 3, 6, 0, 5, 3, 8, 1, 8, 6, 1, 1, 8, 8,
       4, 6, 8, 1, 0, 6, 8, 3, 8, 4, 6, 8, 6, 4, 6, 4, 4, 6, 2, 4, 5, 2,
       3, 1, 0, 4, 3, 0, 7, 1, 8, 5, 3, 0, 1, 8, 1, 8, 0, 2, 4, 7, 8, 8,
       4, 3, 0, 6, 4, 1, 4, 1])), (array([[22.16577861, 70.20228452],
       [26.6124459 , 43.91604964],
       [19.74208872, 64.1000882 ],
       [39.78587939, 19.72783208],
       [15.57193046, 70.58516939],
       [60.58250295, 27.97262624],
       [25.38793378, 73.72176884],
       [ 9.92511389, 25.73098987],
       [55.37159269, 32.68544864],
       [31.42335817, 38.75662838]]), array([9, 9, 4, 9, 8, 7, 9, 3, 0, 3, 3, 1, 4, 3, 0, 7, 3, 8, 3, 0, 7, 1,
       4, 5, 9, 7, 3, 3, 7, 5, 5, 9, 0, 7, 3, 9, 5, 9, 5, 2, 0, 3, 1, 7,
       3, 1, 9, 9, 6, 5, 5, 4, 1, 6, 9, 7, 9, 7, 9, 9, 7, 7, 8, 7, 5, 5,
       1, 3, 2, 5, 0, 3, 9, 9, 7, 9, 0, 7, 9, 2, 8, 9, 5, 6, 7, 3, 7, 1,
       5, 4, 9, 7, 8, 9, 9, 0, 7, 6, 3, 5, 9, 7, 5, 3, 3, 9, 7, 3, 5, 3,
       7, 7, 3, 4, 3, 7, 7, 6, 7, 7, 3, 3, 1, 0, 9, 0, 4, 8, 4, 7, 9, 7,
       5, 4, 5, 7, 1, 3, 1, 8, 1, 8, 6, 5, 4, 0, 8, 7, 7, 4, 5, 7, 8, 3,
       6, 2, 2, 5, 5, 3, 7, 7, 9, 3, 3, 5, 3, 7, 6, 4, 2, 6, 9, 3, 5, 0,
       7, 5, 8, 7, 3, 3, 7, 0, 7, 4, 9, 3, 6, 2, 5, 3, 8, 9, 3, 3, 5, 8,
       7, 9, 5, 3, 3, 9, 5, 4, 5, 7, 9, 5, 9, 7, 9, 7, 7, 9, 1, 7, 2, 1,
       4, 3, 3, 7, 4, 3, 5, 3, 5, 2, 4, 3, 3, 5, 3, 5, 3, 1, 7, 5, 5, 5,
       7, 2, 3, 9, 7, 3, 7, 3]))]
[     0.       75832.02223 123143.46797 134769.30424 150059.18397
 150520.77443 150997.04159 151765.54636 152083.78065 152200.29395]
```
<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 4. The Viterbi Algorithm
![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)  

Write the function `def viterbi(Observation, Emission, Transition, Initial):` that calculates the most likely sequence of hidden states for a hidden Markov model:

- **Observation** is a `numpy.ndarray` of shape `(T,)` that contains the index of the observation.  
  **T** is the number of observations.
- **Emission** is a `numpy.ndarray` of shape `(N, M)` containing the emission probability of a specific observation given a hidden state.  
  **Emission[i, j]** is the probability of observing `j` given the hidden state `i`.  
  **N** is the number of hidden states, and **M** is the number of all possible observations.
- **Transition** is a 2D `numpy.ndarray` of shape `(N, N)` containing the transition probabilities.  
  **Transition[i, j]** is the probability of transitioning from the hidden state `i` to `j`.
- **Initial** is a `numpy.ndarray` of shape `(N, 1)` containing the probability of starting in a particular hidden state.

#### Returns:
- **path** is a list of length `T` containing the most likely sequence of hidden states.
- **P** is the probability of obtaining the `path` sequence.

```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/unsupervised_learning/hmm#./test_files/4-main.py
4.701733355108224e-252
[2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 1, 1, 1, 1, 0, 0, 1, 2, 2, 2, 3, 3, 3, 2, 1, 2, 1, 1, 2, 2, 2, 3, 3, 2, 2, 3, 4, 4, 3, 3, 2, 2, 3, 3, 3, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 2, 3, 3, 2, 1, 2, 1, 1, 1, 2, 2, 3, 4, 4, 4, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 3, 2, 2, 3, 2, 2, 3, 4, 4, 4, 3, 2, 1, 0, 0, 0, 1, 2, 2, 1, 1, 2, 3, 3, 2, 1, 1, 1, 2, 3, 3, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 2, 1, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0, 1, 2, 2, 1, 2, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 3, 3, 4, 4, 4, 4, 3, 3, 3, 2, 1, 1, 1, 1, 2, 1, 0, 0, 0, 0, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 3, 4, 4, 4, 3, 3, 3, 3, 2, 2, 3, 3, 3, 3, 4, 4, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 1, 2, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 1, 1, 2, 1, 1, 2, 2, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 2, 1, 1, 2, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 3, 3, 3, 3, 3]
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/unsupervised_learning/hmm#
```
<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 5. The Backward Algorithm

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Write the function `def backward(Observation, Emission, Transition, Initial):` that performs the backward algorithm for a hidden Markov model:

- **Observation** is a `numpy.ndarray` of shape `(T,)` that contains the index of the observation.  
  **T** is the number of observations.
- **Emission** is a `numpy.ndarray` of shape `(N, M)` containing the emission probability of a specific observation given a hidden state.  
  **Emission[i, j]** is the probability of observing `j` given the hidden state `i`.  
  **N** is the number of hidden states, and **M** is the number of all possible observations.
- **Transition** is a 2D `numpy.ndarray` of shape `(N, N)` containing the transition probabilities.  
  **Transition[i, j]** is the probability of transitioning from the hidden state `i` to `j`.
- **Initial** is a `numpy.ndarray` of shape `(N, 1)` containing the probability of starting in a particular hidden state.

#### Returns:
- **P** is the likelihood of the observations given the model.
- **B** is a `numpy.ndarray` of shape `(N, T)` containing the backward path probabilities.  
  **B[i, j]** is the probability of generating the future observations from hidden state `i` at time `j`.
```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/unsupervised_learning/hmm#./test_files/5-main.py
1.7080966131859631e-214
[[1.28912952e-215 6.12087935e-212 1.00555701e-211 ... 6.75000000e-005
  0.00000000e+000 1.00000000e+000]
 [3.86738856e-214 2.69573528e-212 4.42866330e-212 ... 2.02500000e-003
  0.00000000e+000 1.00000000e+000]
 [6.44564760e-214 5.15651808e-213 8.47145100e-213 ... 2.31330000e-002
  2.70000000e-002 1.00000000e+000]
 [1.93369428e-214 0.00000000e+000 0.00000000e+000 ... 6.39325000e-002
  1.15000000e-001 1.00000000e+000]
 [1.28912952e-215 0.00000000e+000 0.00000000e+000 ... 5.77425000e-002
  2.19000000e-001 1.00000000e+000]]
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/unsupervised_learning/hmm#
```
<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 6. The Baum-Welch Algorithm
![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)  

Write the function `def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):` that performs the Baum-Welch algorithm for a hidden Markov model:

- **Observations** is a `numpy.ndarray` of shape `(T,)` that contains the index of the observation.  
  **T** is the number of observations.
- **Transition** is a `numpy.ndarray` of shape `(M, M)` that contains the initialized transition probabilities.  
  **M** is the number of hidden states.
- **Emission** is a `numpy.ndarray` of shape `(M, N)` that contains the initialized emission probabilities.  
  **N** is the number of output states.
- **Initial** is a `numpy.ndarray` of shape `(M, 1)` that contains the initialized starting probabilities.
- **iterations** is the number of times expectation-maximization should be performed.

#### Returns:
- The converged **Transition**, **Emission**, or **None**, **None** on failure.
```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/unsupervised_learning/hmm#./test_files/6-main.py
[[0.81 0.19]
 [0.28 0.72]]
[[0.82 0.18 0.  ]
 [0.26 0.58 0.16]]
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/unsupervised_learning/hmm#
```
# Random Information ℹ️

- **Repository Name**: holbertonschool-machine_learning 
- **Description**:  
  This repository is a collection of machine learning algorithms and models, implemented as part of my work at Holberton School. It includes various tasks and solutions for different algorithms such as Markov Chains, Viterbi Algorithm, and Baum-Welch Algorithm. The purpose of this repository is to showcase my understanding and implementation of these algorithms in Python, using numpy and other relevant libraries.
- **Repository Link**: [https://github.com/ChaimaBSlima/holbertonschool-machine_learning/](https://github.com/ChaimaBSlima/holbertonschool-machine_learning/)  
- **Clone Command**:  
  To clone this repository to your local machine, use the following command in your terminal:
  ```bash
  git clone https://github.com/ChaimaBSlima/holbertonschool-machine_learning.git
  ```
- **Test Files**:  
  All test files for this project are located in the `test_files` folder within the repository.

- **Additional Information**:  
  - All code is written in Python, and it uses numpy for numerical operations.
  - The repository follows best practices for coding style and includes documentation for each function, class, and module.
  - The repository is intended for educational purposes and as a reference for learning and practicing machine learning algorithms.

# 📊 Project Summary

This project focuses on implementing and solving key algorithms in **Hidden Markov Models (HMMs)**, a fundamental concept in probability theory and machine learning. The objective of this project is to apply different techniques used in HMMs to process sequences of data, including tasks such as sequence prediction, observation likelihood estimation, and model training.

The project is divided into several critical tasks, such as:
- **Markov Chains**: Understanding the probabilities of states transitioning over time.
- **Regular Chains**: Determining steady-state probabilities for regular Markov chains.
- **Absorbing Chains**: Identifying if a Markov chain is absorbing or not.
- **The Forward Algorithm**: Calculating the likelihood of a sequence of observations.
- **The Viterbi Algorithm**: Identifying the most likely sequence of hidden states.
- **The Baum-Welch Algorithm**: Training an HMM using the Expectation-Maximization approach to optimize model parameters.

# 👩‍💻 Authors
Tasks by [Holberton School](https://www.holbertonschool.com/)

**Chaima Ben Slima** - Holberton School Student, ML Developer

[GitHub](https://github.com/ChaimaBSlima)
[Linkedin](https://www.linkedin.com/in/chaima-ben-slima-35477120a/)
