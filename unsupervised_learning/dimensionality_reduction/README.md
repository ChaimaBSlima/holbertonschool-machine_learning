<h1><p align="center"> Dimensionality Reduction </h1></p></font>

<p align="center">
  <img src="https://github.com/user-attachments/assets/216c5ded-0fec-4d60-94b4-e7df9af662dc" alt="Image"/>
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

# 🗃️ Data

- [mnist2500_X.txt](https://github.com/ChaimaBSlima/holbertonschool-machine_learning/blob/main/unsupervised_learning/dimensionality_reduction/data/mnist2500_X.txt)
- [mnist2500_labels.txt](https://github.com/ChaimaBSlima/holbertonschool-machine_learning/blob/main/unsupervised_learning/dimensionality_reduction/data/mnist2500_labels.txt)
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
  <img src="https://github.com/user-attachments/assets/268df614-3dc2-46c8-bfa7-06420e060af2" alt="Image"/>
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

  <p align="center">
  <img src="https://github.com/user-attachments/assets/201a6d01-91ef-456a-a043-935b569ec8c4" alt="Image"/>
</p>

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
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/unsupervised_learning/dimensionality_reduction#./test_files/4-main.py
P: (2500, 2500)
[[0.00000000e+00 7.40714907e-10 9.79862968e-13 ... 2.37913671e-11
  1.22844912e-10 1.75011944e-10]
 [7.40714907e-10 0.00000000e+00 1.68735728e-13 ... 2.11150140e-12
  1.05003596e-11 2.42913116e-10]
 [9.79862968e-13 1.68735728e-13 0.00000000e+00 ... 2.41827214e-11
  3.33128330e-09 1.25696380e-09]
 ...
 [2.37913671e-11 2.11150140e-12 2.41827214e-11 ... 0.00000000e+00
  3.62850172e-12 4.11671350e-10]
 [1.22844912e-10 1.05003596e-11 3.33128330e-09 ... 3.62850172e-12
  0.00000000e+00 6.70800054e-10]
 [1.75011944e-10 2.42913116e-10 1.25696380e-09 ... 4.11671350e-10
  6.70800054e-10 0.00000000e+00]]
1.0000000000000004
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/unsupervised_learning/dimensionality_reduction#
```
<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 5. Q affinities

![Advanced](https://img.shields.io/badge/advanced-🚀-blueviolet)

Write the function `def backward(Observation, Emission, Transition, Initial):` that performs the backward algorithm for a hidden Markov model:

Write a function `def Q_affinities(Y):` that calculates the Q affinities:

- `Y` is a `numpy.ndarray` of shape `(n, ndim)` containing the low dimensional transformation of `X`  
  - `n` is the number of points  
  - `ndim` is the new dimensional representation of `X`  

#### Returns:  
- `Q`, a `numpy.ndarray` of shape `(n, n)` containing the Q affinities  
- `num`, a `numpy.ndarray` of shape `(n, n)` containing the numerator of the Q affinities  

💡 *Hint*: See page 7 of [t-SNE](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) 
```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/unsupervised_learning/dimensionality_reduction#./test_files/5-main.py
num: (2500, 2500)
[[0.         0.1997991  0.34387413 ... 0.08229525 0.43197616 0.29803545]
 [0.1997991  0.         0.08232739 ... 0.0780192  0.36043254 0.20418429]
 [0.34387413 0.08232739 0.         ... 0.07484357 0.16975081 0.17792688]
 ...
 [0.08229525 0.0780192  0.07484357 ... 0.         0.13737822 0.22790422]
 [0.43197616 0.36043254 0.16975081 ... 0.13737822 0.         0.65251175]
 [0.29803545 0.20418429 0.17792688 ... 0.22790422 0.65251175 0.        ]]
2113140.980877581
Q: (2500, 2500)
[[0.00000000e+00 9.45507652e-08 1.62731275e-07 ... 3.89445137e-08
  2.04423728e-07 1.41039074e-07]
 [9.45507652e-08 0.00000000e+00 3.89597234e-08 ... 3.69209645e-08
  1.70567198e-07 9.66259681e-08]
 [1.62731275e-07 3.89597234e-08 0.00000000e+00 ... 3.54181605e-08
  8.03310395e-08 8.42001935e-08]
 ...
 [3.89445137e-08 3.69209645e-08 3.54181605e-08 ... 0.00000000e+00
  6.50113847e-08 1.07850932e-07]
 [2.04423728e-07 1.70567198e-07 8.03310395e-08 ... 6.50113847e-08
  0.00000000e+00 3.08787608e-07]
 [1.41039074e-07 9.66259681e-08 8.42001935e-08 ... 1.07850932e-07
  3.08787608e-07 0.00000000e+00]]
1.0000000000000004
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/unsupervised_learning/dimensionality_reduction#
```
<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 6. Gradients
![Advanced](https://img.shields.io/badge/advanced-🚀-blueviolet)

Write a function `def grads(Y, P):` that calculates the gradients of `Y`:

- `Y` is a `numpy.ndarray` of shape `(n, ndim)` containing the low dimensional transformation of `X`  
- `P` is a `numpy.ndarray` of shape `(n, n)` containing the P affinities of `X`

#### Returns:  
- `dY`, a `numpy.ndarray` of shape `(n, ndim)` containing the gradients of `Y`  
- `Q`, a `numpy.ndarray` of shape `(n, n)` containing the Q affinities of `Y`

💡 *Hint*: See page 8 of [t-SNE](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)
```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/unsupervised_learning/dimensionality_reduction#./test_files/6-main.py
dY: (2500, 2)
[[ 1.28824814e-05  1.55400363e-05]
 [ 3.21435525e-05  4.35358938e-05]
 [-1.02947106e-05  3.53998421e-07]
 ...
 [-2.27447049e-05 -3.05191863e-06]
 [ 9.69379032e-06  1.00659610e-06]
 [ 5.75113416e-05  7.65517123e-09]]
Q: (2500, 2500)
[[0.00000000e+00 9.45507652e-08 1.62731275e-07 ... 3.89445137e-08
  2.04423728e-07 1.41039074e-07]
 [9.45507652e-08 0.00000000e+00 3.89597234e-08 ... 3.69209645e-08
  1.70567198e-07 9.66259681e-08]
 [1.62731275e-07 3.89597234e-08 0.00000000e+00 ... 3.54181605e-08
  8.03310395e-08 8.42001935e-08]
 ...
 [3.89445137e-08 3.69209645e-08 3.54181605e-08 ... 0.00000000e+00
  6.50113847e-08 1.07850932e-07]
 [2.04423728e-07 1.70567198e-07 8.03310395e-08 ... 6.50113847e-08
  0.00000000e+00 3.08787608e-07]
 [1.41039074e-07 9.66259681e-08 8.42001935e-08 ... 1.07850932e-07
  3.08787608e-07 0.00000000e+00]]
1.0000000000000004
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/unsupervised_learning/dimensionality_reduction#
```
<p align="center">⭐⭐⭐⭐⭐⭐</p>


### 7. Cost

![Advanced](https://img.shields.io/badge/advanced-🚀-blueviolet)

Write a function `def cost(P, Q):` that calculates the cost of the t-SNE transformation:

- `P` is a `numpy.ndarray` of shape `(n, n)` containing the P affinities  
- `Q` is a `numpy.ndarray` of shape `(n, n)` containing the Q affinities

#### Returns:  
- `C`, the cost of the transformation

💡 *Hint 1*: See page 5 of [t-SNE](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)  
💡 *Hint 2*: Watch out for division by 0 errors! Take the minimum of all values in `p` and `q` with almost 0 (ex. `1e-12`)
```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/unsupervised_learning/dimensionality_reduction#./test_files/7-main.py
4.531113944164376
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/unsupervised_learning/dimensionality_reduction#
```

<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 8. t-SNE

![Advanced](https://img.shields.io/badge/advanced-🚀-blueviolet)

Write a function `def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):` that performs a t-SNE transformation:

- `X` is a `numpy.ndarray` of shape `(n, d)` containing the dataset to be transformed by t-SNE  
  `n` is the number of data points  
  `d` is the number of dimensions in each point  
- `ndims` is the new dimensional representation of X  
- `idims` is the intermediate dimensional representation of X after PCA  
- `perplexity` is the perplexity  
- `iterations` is the number of iterations  
- `lr` is the learning rate

#### Process:
- Every 100 iterations, not including 0, print `Cost at iteration {iteration}: {cost}`  
  `{iteration}` is the number of times Y has been updated and `{cost}` is the corresponding cost  
- After every iteration, `Y` should be re-centered by subtracting its mean  
- For the first 100 iterations, perform early exaggeration with an exaggeration of 4  
- A(t) = 0.5 for the first 20 iterations and 0.8 thereafter  

#### Returns:  
- `Y`, a `numpy.ndarray` of shape `(n, ndim)` containing the optimized low dimensional transformation of X  

You should use:
- `pca = __import__('1-pca').pca`  
- `P_affinities = __import__('4-P_affinities').P_affinities`  
- `grads = __import__('6-grads').grads`  
- `cost = __import__('7-cost').cost`

💡 *Hint 1*: See Algorithm 1 on page 9 of [t-SNE](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) . But WATCH OUT! There is a mistake in the gradient descent step  
💡 *Hint 2*: See Section 3.4 starting on page 9 of [t-SNE](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)  for early exaggeration
```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/unsupervised_learning/dimensionality_reduction#./test_files/8-main.py
Cost at iteration 100: 15.132745380504451
Cost at iteration 200: 1.4499349051185884
Cost at iteration 300: 1.2991961074009266
Cost at iteration 400: 1.2255530221811528
Cost at iteration 500: 1.179753264451479
Cost at iteration 600: 1.1476306791330366
Cost at iteration 700: 1.123501502573646
Cost at iteration 800: 1.1044968276172735
Cost at iteration 900: 1.0890468673949145
Cost at iteration 1000: 1.0762018736143146
Cost at iteration 1100: 1.0652921250043619
Cost at iteration 1200: 1.0558751316523136
Cost at iteration 1300: 1.047653338870073
Cost at iteration 1400: 1.0403981880716473
Cost at iteration 1500: 1.033935359326665
Cost at iteration 1600: 1.0281287524465708
Cost at iteration 1700: 1.0228885344794134
Cost at iteration 1800: 1.0181265576736775
Cost at iteration 1900: 1.0137760713813615
Cost at iteration 2000: 1.009782545181553
Cost at iteration 2100: 1.0061007125574222
Cost at iteration 2200: 1.0026950513450206
Cost at iteration 2300: 0.9995335333268901
Cost at iteration 2400: 0.9965894332394158
Cost at iteration 2500: 0.9938399255561283
Cost at iteration 2600: 0.9912653473151115
Cost at iteration 2700: 0.9888485527807156
Cost at iteration 2800: 0.9865746432480398
Cost at iteration 2900: 0.9844307720012038
Cost at iteration 3000: 0.9824051809484152
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/unsupervised_learning/dimensionality_reduction#
```

<p align="center">
  <img src="https://github.com/user-attachments/assets/8b363256-fced-4d5d-8803-ecfd32dd56b0" alt="Image"/>
</p>

**Awesome! We can see pretty good clusters! For comparison, here’s how PCA performs on the same dataset:**

```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/unsupervised_learning/dimensionality_reduction#./pca.py 
```

<p align="center">
  <img src="https://github.com/user-attachments/assets/5a087129-3960-40dd-b2b3-706c7edec4e0" alt="Image"/>
</p>

---

# 📄 Files

| Task Number | Task Title                   |File                 | Priority                                                             |
|-------------|------------------------------|---------------------|----------------------------------------------------------------------|
| 0           | 0. PCA                | `0-pca.py`   | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |
| 1           | 1. PCA v2              | `1-pca.py` | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |
| 2           | 2. Initialize t-SNE      | `2-P_init.py`     |  ![Advanced](https://img.shields.io/badge/advanced-🚀-blueviolet)  |
| 3           | 3. Entropy                | `3-entropy.py`    |  ![Advanced](https://img.shields.io/badge/advanced-🚀-blueviolet)  |
| 4           | 4. P affinities      | `4-P_affinities.py` | ![Advanced](https://img.shields.io/badge/advanced-🚀-blueviolet)    |
| 5           | 5. Q affinities      | `5-Q_affinities.py` | ![Advanced](https://img.shields.io/badge/advanced-🚀-blueviolet)    |
| 6           | 6. Gradients     | `6-grads.py` | ![Advanced](https://img.shields.io/badge/advanced-🚀-blueviolet)    |
| 7           | 7. Cost          | `7-cost.py` | ![Advanced](https://img.shields.io/badge/advanced-🚀-blueviolet)    |
| 8           | 8. t-SNE         | `8-tsne.py` | ![Advanced](https://img.shields.io/badge/advanced-🚀-blueviolet)    |

---

# 📊 Project Summary

This project focuses on implementing **t-SNE (t-Distributed Stochastic Neighbor Embedding)** for dimensionality reduction and data visualization. The goal is to transform high-dimensional datasets into a lower-dimensional space (typically 2D or 3D) while preserving the structure of the data.

---

# ℹ️ Random Information 

- **Repository Name**: holbertonschool-machine_learning 
- **Description**:  
  This repository is a comprehensive collection of my machine learning work completed during my time at Holberton School. It demonstrates my practical understanding of key concepts in machine learning, including supervised learning, unsupervised learning, and reinforcement learning.

  Machine learning is a field of study that enables systems to learn from data, identify patterns, and make decisions or predictions with minimal human intervention.

  - `Supervised learning` involves training a system using labeled data, allowing it to learn the relationship between inputs and known outputs.  
  - `Unsupervised learning` focuses on exploring data without predefined labels, aiming to discover hidden patterns or groupings within the data.  
  - `Reinforcement learning` centers around learning through interaction with an environment, where a system receives feedback in the form of rewards or penalties to improve its performance over time.

  This repository includes tasks and solutions implemented primarily in Python using libraries like NumPy, serving as a demonstration of my technical ability and understanding of foundational machine learning principles.

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

---

# 👩‍💻 Authors
Tasks by [Holberton School](https://www.holbertonschool.com/)

**Chaima Ben Slima** - Holberton School Student, ML Developer

[GitHub](https://github.com/ChaimaBSlima)
[Linkedin](https://www.linkedin.com/in/chaima-ben-slima-35477120a/)

