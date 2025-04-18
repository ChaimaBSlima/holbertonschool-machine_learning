<h1><p align="center"> Multivariate Probability </h1></p></font>

<p align="center">
  <img src="https://github.com/user-attachments/assets/987f3501-70ff-43f3-a327-6f6db35a05c5" alt="Image"/>
</p>


# 📚 Resources

Read or watch:
- [Joint Probability Distributions](https://dokumen.tips/documents/chapter-5-joint-probability-distributions-part-1-rdecookstat2020notesch5pt1pdf.html?page=1)
- [Multivariate Gaussian distributions](https://www.youtube.com/watch?v=eho8xH3E6mE)
- [The Multivariate Gaussian Distribution](https://cs229.stanford.edu/section/gaussians.pdf)
- [An Introduction to Variance, Covariance & Correlation](https://www.alchemer.com/resources/blog/variance-covariance-correlation/)
- [Variance-covariance matrix using matrix notation of factor analysis](https://www.youtube.com/watch?v=G16c2ZODcg8)

Definitions to skim:
- [Carl Friedrich Gauss](https://en.wikipedia.org/wiki/Carl_Friedrich_Gauss)
- [Joint probability distribution](https://en.wikipedia.org/wiki/Joint_probability_distribution)
- [Covariance](https://en.wikipedia.org/wiki/Covariance)
- [Covariance matrix](https://en.wikipedia.org/wiki/Covariance_matrix)

As references:
- [numpy.cov](https://numpy.org/doc/stable/reference/generated/numpy.cov.html)
- [numpy.corrcoef](https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html)
- [numpy.linalg.det](https://numpy.org/doc/stable/reference/generated/numpy.linalg.det.html)
- [numpy.linalg.inv](https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html)
- [numpy.random.multivariate_normal](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.multivariate_normal.html)

---

# 🎯 Learning Objectives
- Who is Carl Friedrich Gauss?  
- What is a joint/multivariate distribution?  
- What is a covariance?  
- What is a correlation coefficient?  
- What is a covariance matrix?  
- What is a multivariate Gaussian distribution?  

---
# 🧾 Requirements

### General

- Allowed editors: `vi`, `vim`, `emacs`  
- All your files will be interpreted/compiled on **Ubuntu 20.04 LTS** using **python3 (version 3.9)**  
- Your files will be executed with **numpy (version 1.25.2)**  
- All your files should end with a **new line**  
- The first line of all your files should be exactly: `#!/usr/bin/env python3`  
- A README.md file, at the root of the folder of the project, is mandatory  
- Your code should use the `pycodestyle` style (version 2.11.1)  
- All your modules should have documentation (`python3 -c 'print(__import__("my_module").__doc__)'`)  
- All your classes should have documentation (`python3 -c 'print(__import__("my_module").MyClass.__doc__)'`)  
- All your functions (inside and outside a class) should have documentation (`python3 -c 'print(__import__("my_module").my_function.__doc__)'` and python3 -c `'print(__import__("my_module").MyClass.my_function.__doc__)'`)  
- Unless otherwise noted, you are not allowed to import any module except `import numpy as np`  
- All your files must be executable  
- The length of your files will be tested using `wc`  

---
# ❓ Quiz:

### Question #0  
px, y(x, y) =

- ~~P(X = x)P(Y = y)~~  
- ~~P(X = x | Y = y)~~ 
- P(X = x | Y = y)P(Y = y) ✔️  
- ~~P(Y = y | X = x)~~  
- P(Y = y | X = x)P(X = x) ✔️  
- P(X = x ∩ Y = y) ✔️
- ~~P(X = x ∪ Y = y)~~  

### Question #1  
The i,jth entry in the covariance matrix is

- ~~the variance of variable i plus the variance of variable j~~  
- the variance of i if i == j ✔️  
- the same as the j,ith entry  ✔️
- the variance of variable i and variable j ✔️

### Question #2  
The correlation coefficient of the variables X and Y is defined as:

- ~~ρ = σXY2~~  
- ~~ρ = σXY~~  
- ρ = σXY / ( σX σY ) ✔️  
- ~~ρ = σXY / ( σXX σYY )~~  

---

# 📝 Tasks

### 0. Mean and Covariance

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Write the function `def mean_cov(X):` that calculates the mean and covariance of a data set:

- `X` is a numpy.ndarray of shape `(n, d)` containing the data set:
  - `n` is the number of data points
  - `d` is the number of dimensions in each data point  
- If `X` is not a 2D numpy.ndarray, raise a `TypeError` with the message `X must be a 2D numpy.ndarray`  
- If `n` is less than 2, raise a `ValueError` with the message `X must contain multiple data points`  

#### Returns:
- `mean`: a numpy.ndarray of shape `(1, d)` containing the mean of the data set  
- `cov`: a numpy.ndarray of shape `(d, d)` containing the covariance matrix of the data set  

You are not allowed to use the function `numpy.cov`
```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/multivariate_prob#./test_files/0-main.py
[[12.04341828 29.92870885 10.00515808]]
[[ 36.2007391  -29.79405239  15.37992641]
 [-29.79405239  97.77730626 -20.67970134]
 [ 15.37992641 -20.67970134  24.93956823]]
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/multivariate_prob#
```
<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 1. Correlation

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Write the function `def correlation(C):` that calculates a correlation matrix:

- `C` is a numpy.ndarray of shape `(d, d)` containing a covariance matrix  
  - `d` is the number of dimensions  
- If `C` is not a numpy.ndarray, raise a `TypeError` with the message `C must be a numpy.ndarray`  
- If `C` does not have shape `(d, d)`, raise a `ValueError` with the message `C must be a 2D square matrix`  

#### Returns:
- A numpy.ndarray of shape `(d, d)` containing the correlation matrix
```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/multivariate_prob#./test_files/1-main.py
[[ 36 -30  15]
 [-30 100 -20]
 [ 15 -20  25]]
[[ 1.  -0.5  0.5]
 [-0.5  1.  -0.4]
 [ 0.5 -0.4  1. ]]
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/multivariate_prob#
```

<p align="center">⭐⭐⭐⭐⭐⭐</p>


### 2. Initialize

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Create the class `MultiNormal` that represents a Multivariate Normal distribution:

Class constructor `def __init__(self, data):`

- `data` is a numpy.ndarray of shape `(d, n)` containing the data set:
  - `n` is the number of data points
  - `d` is the number of dimensions in each data point  
- If `data` is not a 2D numpy.ndarray, raise a `TypeError` with the message `data must be a 2D numpy.ndarray`  
- If `n` is less than 2, raise a `ValueError` with the message `data must contain multiple data points`  

Set the public instance variables:
- `mean`: a numpy.ndarray of shape `(d, 1)` containing the mean of data  
- `cov`: a numpy.ndarray of shape `(d, d)` containing the covariance matrix of data  

You are not allowed to use the function `numpy.cov`
```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/multivariate_prob#./test_files/2-main.py
[[12.04341828]
 [29.92870885]
 [10.00515808]]
[[ 36.2007391  -29.79405239  15.37992641]
 [-29.79405239  97.77730626 -20.67970134]
 [ 15.37992641 -20.67970134  24.93956823]]
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/multivariate_prob#
```

<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 3. PDF

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Update the class `MultiNormal`:

Add the public instance method `def pdf(self, x):` that calculates the PDF at a data point:

- `x` is a numpy.ndarray of shape `(d, 1)` containing the data point whose PDF should be calculated  
  - `d` is the number of dimensions of the `Multinomial` instance  
- If `x` is not a numpy.ndarray, raise a `TypeError` with the message `x must be a numpy.ndarray`  
- If `x` is not of shape `(d, 1)`, raise a `ValueError` with the message `x must have the shape ({d}, 1)`  

#### Returns:
- The value of the PDF  

You are not allowed to use the function `numpy.cov`
```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/multivariate_prob#./test_files/3-main.py
[[ 8.20311936]
 [32.84231319]
 [ 9.67254478]]
0.00022930236202143827
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/multivariate_prob#
```
---
# 📄 Files

| Task Number | Task Title                   |File                 | Priority                                                             |
|-------------|------------------------------|---------------------|----------------------------------------------------------------------|
| 0           | 0. Mean and Covariance        | `0-mean_cov.py`   | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |
| 1           | 1. Correlation              | `1-correlation.py` | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |
| 2           | 2. Initialize     | `multinormal.py`     | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |
| 3           | 3. PDF               | `multinormal.py`    | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |
 
---

# 📊 Project Summary

This project involves the implementation of a Multivariate Normal distribution class with several key methods to calculate important statistical measures.

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
  - The repository is intended for educational purposes and as a reference for learning and practicing machine learning algorithms

---

# 👩‍💻 Authors
Tasks by [Holberton School](https://www.holbertonschool.com/)

**Chaima Ben Slima** - Holberton School Student, ML Developer

[GitHub](https://github.com/ChaimaBSlima)
[Linkedin](https://www.linkedin.com/in/chaima-ben-slima-35477120a/)
