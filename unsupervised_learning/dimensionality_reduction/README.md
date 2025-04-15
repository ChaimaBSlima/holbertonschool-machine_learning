<h1><p align="center"> Dimensionality Reduction </h1></p></font>

<p align="center">
  <img src="" alt="Image"/>
</p>

# 📚 Resources
Read or watch:
- [Dimensionality Reduction For Dummies — Part 1: Intuition](https://towardsdatascience.com/https-medium-com-abdullatif-h-dimensionality-reduction-for-dummies-part-1-a8c9ec7b7e79/?gi=35f0782d9542)
- [Singular Value Decomposition](https://www.youtube.com/watch?v=P5mlg91as1c)
- [Understanding SVD (Singular Value Decomposition)](https://towardsdatascience.com/svd-8c2f72e264f/?gi=a482351224ab)
- [Intuitively, what is the difference between Eigendecomposition and Singular Value Decomposition?](https://math.stackexchange.com/questions/320220/intuitively-what-is-the-difference-between-eigendecomposition-and-singular-valu)
- [Dimensionality Reduction: Principal Components Analysis, Part 1](https://www.youtube.com/watch?v=ZqXnPcyIAL8)
- [Dimensionality Reduction: Principal Components Analysis, Part 2](https://www.youtube.com/watch?v=NUn6WeFM5cM)
- [StatQuest: t-SNE, Clearly Explained](https://www.youtube.com/watch?v=NEaUSP4YerM)
- [t-SNE tutorial Part1](https://www.youtube.com/watch?v=ohQXphVSEQM)
- [t-SNE tutorial Part2](https://www.youtube.com/watch?v=W-9L6v_rFIE)
- [How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/)

Definitions to skim:
- [Dimensionality Reduction](https://en.wikipedia.org/wiki/Dimensionality_reduction)
- [Principal component analysis](https://en.wikipedia.org/wiki/Principal_component_analysis)
- [Eigendecomposition of a matrix](https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix)
- [Singular value decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition)
- [Manifold](https://en.wikipedia.org/wiki/Manifold)
- [Kullback–Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#:~:text=In%20the%20simple%20case%2C%20a,mechanics%2C%20neuroscience%20and%20machine%20learning.)
- [T-distributed stochastic neighbor embedding](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding)

As references:
- [numpy.cumsum](https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html)
- [Visualizing Data using t-SNE](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)
- [Visualizing Data Using t-SNE](https://www.youtube.com/watch?v=RJVL80Gg3lA)

Advanced:
- [Kernel principal component analysis](https://en.wikipedia.org/wiki/Kernel_principal_component_analysis)
- [Nonlinear Dimensionality Reduction: KPCA](https://www.youtube.com/watch?v=HbDHohXPLnU)
---


# 🎯 Learning Objectives

- What is eigendecomposition?  
- What is singular value decomposition?  
- What is the difference between eig and svd?  
- What is dimensionality reduction and what are its purposes?  
- What is principal components analysis (PCA)?  
- What is t-distributed stochastic neighbor embedding (t-SNE)?  
- What is a manifold?  
- What is the difference between linear and non-linear dimensionality reduction?  
- Which techniques are linear/non-linear?  

---

# 🧾 Requirements

### General

- Allowed editors: `vi`, `vim`, `emacs`  
- All your files will be interpreted/compiled on **Ubuntu 20.04 LTS** using **python3 (version 3.9)**  
- Your files will be executed with **numpy (version 1.25.2)**  
- All your files should end with a **new line**  
- The first line of all your files should be exactly: `#!/usr/bin/env python3`  
- A `README.md` file, at the root of the folder of the project, is **mandatory**  
- Your code should use the `pycodestyle` style (version 2.11.1)  
- All your modules should have documentation (`python3 -c 'print(__import__("my_module").__doc__)'`)  
- All your classes should have documentation (`python3 -c 'print(__import__("my_module").MyClass.__doc__)'`)  
- All your functions (inside and outside a class) should have documentation (`python3 -c 'print(__import__("my_module").my_function.__doc__)'` and `python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)'`)  
- Unless otherwise noted, you are not allowed to import any module except `import numpy as np`  
- All your files must be executable  
- Your code should use the minimum number of operations to avoid floating point errors 
---

# 📊 Data

- [mnist2500_X.txt](https://intranet-projects-files.s3.amazonaws.com/holbertonschool-ml/mnist2500_X.txt)
- [mnist2500_labels.txt](https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/text/2019/10/72a86270e2a1c2cbc14b.txt?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20250415%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20250415T172725Z&X-Amz-Expires=345600&X-Amz-SignedHeaders=host&X-Amz-Signature=a2cb5674b3d71f207b046ce5da3f429d24522e02ced3f6b4872bec26eaaa2485)
---

# ⚠️ Watch Out!

Just like lists, `np.ndarrays` are mutable objects:
```python
>>> vector = np.ones((100, 1))
>>> m1 = vector[55]
>>> m2 = vector[55, 0]
>>> vector[55] = 2
>>> m1
array([2.])
>>> m2
1.0
```
---
# ⚙️ Performance between SVD and EIG
Here a graph of execution time (Y-axis) for the number of iteration (X-axis) - red line is EIG and blue line is SVG

<p align="center">
  <img src="" alt="Image"/>
</p>
---
# 📝 Tasks

### 0. PCA

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Write a function `def pca(X, var=0.95):` that performs PCA on a dataset:

- `X` is a `numpy.ndarray` of shape `(n, d)` where:  
  - `n` is the number of data points  
  - `d` is the number of dimensions in each point  
  - All dimensions have a mean of 0 across all data points  

- `var` is the fraction of the variance that the PCA transformation should maintain  

#### Returns:
- The weights matrix `W` that maintains `var` fraction of `X`'s original variance  
- `W` is a `numpy.ndarray` of shape `(d, nd)` where `nd` is the new dimensionality of the transformed `X`
```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/unsupervised_learning/dimensionality_reduction#./test_files/0-main.py
[[-16.71379391   3.25277063  -3.21956297]
 [ 16.22654311  -0.7283969   -0.88325252]
 [ 15.05945199   3.81948929  -1.97153621]
 [ -7.69814111   5.49561088  -4.34581561]
 [ 14.25075197   1.37060228  -4.04817187]
 [-16.66888233  -3.77067823   2.6264981 ]
 [  6.71765183   0.18115089  -1.91719288]
 [ 10.20004065  -0.84380128   0.44754302]
 [-16.93427229   1.72241573   0.9006236 ]
 [-12.4100987    0.75431367  -0.36518129]
 [-16.40464248   1.98431953   0.34907508]
 [ -6.69439671   1.30624703  -2.77438892]
 [ 10.84363895   4.99826372  -1.36502623]
 [-17.2656016    7.29822621   0.63226953]
 [  5.32413372  -0.54822516  -0.79075935]
 [ -5.63240657   1.50278876  -0.27590797]
 [ -7.63440366   7.72788006  -2.58344477]
 [  4.3348786   -2.14969035   0.61262033]
 [ -3.95417052   4.22254889  -0.14601319]
 [ -6.59947069  -1.00867621   2.29551761]
 [ -0.78942283  -4.15454151   5.87117533]
 [ 13.62292856   0.40038586  -1.36043631]
 [  0.03536684  -5.85950737  -1.86196569]
 [-11.1841298    5.20313078   2.37753549]
 [  9.62095425  -1.17179699  -4.97535412]
 [  3.85296648   3.55808      3.65166717]
 [  6.57934417   4.87503426   0.30243418]
 [-16.17025935   1.49358788   1.0663259 ]
 [ -4.33639793   1.26186205  -2.99149191]
 [ -1.52947063  -0.39342225  -2.96475006]
 [  9.80619496   6.65483286   0.07714817]
 [ -2.45893463  -4.89091813  -0.6918453 ]
 [  9.56282904  -1.8002211    2.06720323]
 [  1.70293073   7.68378254   5.03581954]
 [  9.58030378  -6.97453776   0.64558546]
 [ -3.41279182 -10.07660784  -0.39277019]
 [ -2.74983634  -6.25461193  -2.65038235]
 [  4.54987003   1.28692201  -2.40001675]
 [ -1.81149682   5.16735962   1.4245976 ]
 [ 13.97823555  -4.39187437   0.57600155]
 [ 17.39107161   3.26808567   2.50429006]
 [ -1.25835112  -6.60720376   3.24220508]
 [  1.06405562  -1.25980089   4.06401644]
 [ -3.44578711  -5.21002054  -4.20836152]
 [-21.1181523   -3.72353504   1.6564066 ]
 [ -6.56723647  -4.31268383   1.22783639]
 [ 11.77670231   0.67338386   2.94885044]
 [ -7.89417224  -9.82300322  -1.69743681]
 [ 15.87543091   0.3804009    3.67627751]
 [  7.38044431  -1.58972122   0.60154138]]
1.7353180054998176e-29
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/unsupervised_learning/dimensionality_reduction#
```
<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 1. PCA v2

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Write a function `def pca(X, ndim):` that performs PCA on a dataset:

- `X` is a `numpy.ndarray` of shape `(n, d)` where:  
  - `n` is the number of data points  
  - `d` is the number of dimensions in each point  

- `ndim` is the new dimensionality of the transformed `X`  

#### Returns:
- `T`, a `numpy.ndarray` of shape `(n, ndim)` containing the transformed version of `X`
```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/unsupervised_learning/dimensionality_reduction#./test_files/1-main.py
X: (2500, 784)
[[1. 1. 1. ... 1. 1. 1.]
 [1. 1. 1. ... 1. 1. 1.]
 [1. 1. 1. ... 1. 1. 1.]
 ...
 [1. 1. 1. ... 1. 1. 1.]
 [1. 1. 1. ... 1. 1. 1.]
 [1. 1. 1. ... 1. 1. 1.]]
T: (2500, 50)
[[-0.61344587  1.37452188 -1.41781926 ... -0.42685217  0.02276617
   0.1076424 ]
 [-5.00379081  1.94540396  1.49147124 ...  0.26249077 -0.4134049
  -1.15489853]
 [-0.31463237 -2.11658407  0.36608266 ... -0.71665401 -0.18946283
   0.32878802]
 ...
 [ 3.52302175  4.1962009  -0.52129062 ... -0.24412645  0.02189273
   0.19223197]
 [-0.81387035 -2.43970416  0.33244717 ... -0.55367626 -0.64632309
   0.42547833]
 [-2.25717018  3.67177791  2.83905021 ... -0.35014766 -0.01807652
   0.31548087]]
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/unsupervised_learning/dimensionality_reduction#
```

<p align="center">⭐⭐⭐⭐⭐⭐</p>


### 2. Absorbing Chains2. Initialize t-SNE

![Advanced](https://img.shields.io/badge/advanced-🚀-blueviolet)

Write a function `def P_init(X, perplexity):` that initializes all variables required to calculate the P affinities in t-SNE:

- `X` is a `numpy.ndarray` of shape `(n, d)` containing the dataset to be transformed by t-SNE  
  - `n` is the number of data points  
  - `d` is the number of dimensions in each point  
- `perplexity` is the perplexity that all Gaussian distributions should have

#### Returns: `(D, P, betas, H)`  
- `D`: a `numpy.ndarray` of shape `(n, n)` that calculates the squared pairwise distance between two data points  
  - The diagonal of `D` should be 0s  
- `P`: a `numpy.ndarray` of shape `(n, n)` initialized to all 0‘s that will contain the P affinities  
- `betas`: a `numpy.ndarray` of shape `(n, 1)` initialized to all 1’s that will contain all of the beta values  
  - \(\beta_{i} = \frac{1}{2{\sigma_{i}}^{2}}\)  
- `H`: the Shannon entropy for perplexity `perplexity` with a base of 2
```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/unsupervised_learning/dimensionality_reduction#./test_files/2-main.py
X: (2500, 50)
[[-0.61344587  1.37452188 -1.41781926 ... -0.42685217  0.02276617
   0.1076424 ]
 [-5.00379081  1.94540396  1.49147124 ...  0.26249077 -0.4134049
  -1.15489853]
 [-0.31463237 -2.11658407  0.36608266 ... -0.71665401 -0.18946283
   0.32878802]
 ...
 [ 3.52302175  4.1962009  -0.52129062 ... -0.24412645  0.02189273
   0.19223197]
 [-0.81387035 -2.43970416  0.33244717 ... -0.55367626 -0.64632309
   0.42547833]
 [-2.25717018  3.67177791  2.83905021 ... -0.35014766 -0.01807652
   0.31548087]]
D: (2500, 2500)
[[  0.   107.88 160.08 ... 129.62 127.61 121.67]
 [107.88   0.   170.97 ... 142.58 147.69 116.64]
 [160.08 170.97   0.   ... 138.66 110.26 115.89]
 ...
 [129.62 142.58 138.66 ...   0.   156.12 109.99]
 [127.61 147.69 110.26 ... 156.12   0.   114.08]
 [121.67 116.64 115.89 ... 109.99 114.08   0.  ]]
P: (2500, 2500)
[[0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 ...
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]]
betas: (2500, 1)
[[1.]
 [1.]
 [1.]
 ...
 [1.]
 [1.]
 [1.]]
H: 4.906890595608519
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/unsupervised_learning/dimensionality_reduction#
```

<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 3. Entropy

![Advanced](https://img.shields.io/badge/advanced-🚀-blueviolet)

and P affinities relative to a data point:

- `Di` is a `numpy.ndarray` of shape `(n - 1,)` containing the pairwise distances between a data point and all other points except itself  
  - `n` is the number of data points  
- `beta` is a `numpy.ndarray` of shape `(1,)` containing the beta value for the Gaussian distribution

#### Returns: `(Hi, Pi)`  
- `Hi`: the Shannon entropy of the points  
- `Pi`: a `numpy.ndarray` of shape `(n - 1,)` containing the P affinities of the points

💡 *Hint*: see page 4 of the [t-SNE](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)
```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/unsupervised_learning/dimensionality_reduction#./test_files/3-main.py
0.057436093636173254
[0.00000000e+00 3.74413188e-35 8.00385528e-58 ... 1.35664798e-44
 1.00374765e-43 3.81537517e-41]
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/unsupervised_learning/dimensionality_reduction#
```
<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 4. P affinities

![Advanced](https://img.shields.io/badge/advanced-🚀-blueviolet)

Write a function `def P_affinities(X, tol=1e-5, perplexity=30.0):` that calculates the symmetric P affinities of a data set:

- `X` is a `numpy.ndarray` of shape `(n, d)` containing the dataset to be transformed by t-SNE  
  - `n` is the number of data points  
  - `d` is the number of dimensions in each point  
- `perplexity` is the perplexity that all Gaussian distributions should have  
- `tol` is the maximum tolerance allowed (inclusive) for the difference in Shannon entropy from perplexity for all Gaussian distributions  

You should use `P_init = __import__('2-P_init').P_init` and `HP = __import__('3-entropy').HP`.

#### Returns:  
- `P`, a `numpy.ndarray` of shape `(n, n)` containing the symmetric P affinities

💡 *Hint 1*: See page 6 of [t-SNE](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) 
💡 *Hint 2*: For this task, you will need to perform a binary search on each point’s distribution to find the correct value of `beta` that will give a Shannon Entropy `H` within the tolerance (Think about why we analyze the Shannon entropy instead of perplexity). Since beta can be in the range `(0, inf)`, you will have to do a binary search with the `high` and `low` initially set to None. If in your search, you are supposed to increase/decrease `beta` to `high`/`low` but they are still set to `None`, you should double/half the value of `beta` instead.

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
# 📊 Project Summary

This project focuses on implementing and solving key algorithms in **Hidden Markov Models (HMMs)**, a fundamental concept in probability theory and machine learning. The objective of this project is to apply different techniques used in HMMs to process sequences of data, including tasks such as sequence prediction, observation likelihood estimation, and model training.


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

# 👩‍💻 Authors
Tasks by [Holberton School](https://www.holbertonschool.com/)

**Chaima Ben Slima** - Holberton School Student, ML Developer

[GitHub](https://github.com/ChaimaBSlima)
[Linkedin](https://www.linkedin.com/in/chaima-ben-slima-35477120a/)

