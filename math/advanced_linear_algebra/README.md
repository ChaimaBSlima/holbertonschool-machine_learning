<h1><p align="center"> Advanced Linear Algebra </h1></p></font>

<p align="center">
  <img src="https://github.com/user-attachments/assets/34916010-05b2-4c8d-8b6d-9751ff97fe31" alt="Image"/>
</p>

# 📚 Resources

Read or watch:
- [The determinant | Essence of linear algebra](https://www.youtube.com/watch?v=Ip3X9LOh2dk&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=8)
- [Determinant of a Matrix](https://www.mathsisfun.com/algebra/matrix-determinant.html)
- [Determinant](https://mathworld.wolfram.com/Determinant.html)
- [Determinant of an empty matrix](https://www.quora.com/What-is-the-determinant-of-an-empty-matrix-such-as-a-0x0-matrix)
- [Inverse matrices, column space and null space](https://www.youtube.com/watch?v=uQhTuRlWMxw&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=8&t=0s)
- [Inverse of a Matrix using Minors, Cofactors and Adjugate](https://www.mathsisfun.com/algebra/matrix-inverse-minors-cofactors-adjugate.html)
- [Minor](https://mathworld.wolfram.com/Minor.html)
- [Cofactor](https://mathworld.wolfram.com/Cofactor.html)
- [Adjugate matrix](https://en.wikipedia.org/wiki/Adjugate_matrix)
- [Singular Matrix](https://mathworld.wolfram.com/SingularMatrix.html)
- [Elementary Matrix Operations](https://stattrek.com/matrix-algebra/elementary-operations)
- [Gaussian Elimination](https://mathworld.wolfram.com/GaussianElimination.html)
- [Gauss-Jordan Elimination](https://mathworld.wolfram.com/Gauss-JordanElimination.html)
- [Matrix Inverse](https://mathworld.wolfram.com/MatrixInverse.html)
- [Eigenvectors and eigenvalues | Essence of linear algebra](https://www.youtube.com/watch?v=PFDu9oVAE-g)
- [Eigenvalues and eigenvectors](https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors)
- [Eigenvalues and Eigenvectors](https://math.mit.edu/~gs/linearalgebra/ila6/ila6_6_1.pdf)
- [Definiteness of a matrix](https://en.wikipedia.org/wiki/Definite_matrix) Up to Eigenvalues 
- [Definite, Semi-Definite and Indefinite Matrices](http://mathonline.wikidot.com/definite-semi-definite-and-indefinite-matrices) Ignore Hessian Matrices
- [Tests for Positive Definiteness of a Matrix](https://www.gaussianwaves.com/2013/04/tests-for-positive-definiteness-of-a-matrix/)
- [Positive Definite Matrices and Minima](https://www.youtube.com/watch?v=tccVVUnLdbc)
- [Positive Definite Matrices](https://www.math.utah.edu/~zwick/Classes/Fall2012_2270/Lectures/Lecture33_with_Examples.pdf)

As references:
- [numpy.linalg.eig](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html)


---

# 🎯 Learning Objectives

At the end of this project, you are expected to be able to [explain to anyone](https://fs.blog/feynman-learning-technique/), without the help of Google:

### General

- What is a determinant? How would you calculate it?  
- What is a minor, cofactor, adjugate? How would calculate them?  
- What is an inverse? How would you calculate it?  
- What are eigenvalues and eigenvectors? How would you calculate them?  
- What is definiteness of a matrix? How would you determine a matrix’s definiteness?  

---

# 🧾 Requirements

### General

- Allowed editors: `vi`, `vim`, `emacs`  
- All your files will be interpreted/compiled on **Ubuntu 20.04 LTS** using **python3 (version 3.9)**  
- Your files will be executed with **numpy (version 1.25.2)**  
- All your files should end with a **new line**  
- The first line of all your files should be exactly: `#!/usr/bin/env python3`  
- A `README.md` file, at the root of the folder of the project, is mandatory  
- Your code should use the `pycodestyle` style (version 2.11.1)  
- All your modules should have documentation (`python3 -c 'print(__import__("my_module").__doc__)'`)  
- All your classes should have documentation (`python3 -c 'print(__import__("my_module").MyClass.__doc__)'`)  
- All your functions (inside and outside a class) should have documentation (`python3 -c 'print(__import__("my_module").my_function.__doc__)'` and `python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)'`)  
- Unless otherwise noted, you are not allowed to import any module  
- All your files must be executable  
- The length of your files will be tested using `wc`  

---
# ❓ Quiz:

### Question #0  
What is the determinant of the following matrix?

[[ -7, 0, 6 ]  
  [ 5, -2, -10 ]  
  [ 4, 3, 2 ]]

- -44 ✔️  
- ~~44~~  
- ~~14~~  
- ~~-14~~  

### Question #1  
What is the minor of the following matrix?

[[ -7, 0, 6 ]  
  [ 5, -2, -10 ]  
  [ 4, 3, 2 ]]

- ~~[[ 26, 50, 23 ], [ -18, -38, -21 ], [ 12, 40, 15 ]]~~ 
- [[ 26, 50, 23 ], [ -18, -38, -21 ], [ 12, 40, 14 ]]  ✔️
- ~~[[ 26, 50, 23 ], [ -18, -39, -21 ], [ 12, 40, 14 ]]~~  
- ~~[[ 26, 50, 23 ], [ -18, -39, -21 ], [ 12, 40, 15 ]]~~  

### Question #2  
What is the cofactor of the following matrix?

[[ 6, -9, 9 ],  
  [ 7, 5, 0 ],  
  [ 4, 3, -8 ]]

- ~~[[ -40, 56, 1 ], [ -45, -84, -54 ], [ -45, 64, 93 ]]~~  
- ~~[[ -40, 56, 1 ], [ -44, -84, -54 ], [ -45, 64, 93 ]]~~  
- ~~[[ -40, 56, 1 ], [ -44, -84, -54 ], [ -45, 63, 93 ]]~~  
- [[ -40, 56, 1 ], [ -45, -84, -54 ], [ -45, 63, 93 ]] ✔️ 

### Question #3  
What is the adjugate of the following matrix?

[[ -4, 1, 9 ],  
  [ -9, -8, -5 ],  
  [ -3, 8, 10 ]]

- ~~[[ -40, 62, 67 ], [ 105, -13, -101 ], [ -97, 29, 41 ]]~~  
- ~~[[ -40, 62, 67 ], [ 105, -14, -101 ], [ -97, 29, 41 ]]~~  
- [[ -40, 62, 67 ], [ 105, -13, -101 ], [ -96, 29, 41 ]] ✔️ 
- ~~[[ -40, 62, 67 ], [ 105, -14, -101 ], [ -96, 29, 41 ]]~~  

### Question #4  
Is the following matrix invertible? If so, what is its inverse?

[[ 1, 0, 1 ]  
  [ 2, 1, 2 ]  
  [ 1, 0, -1 ]]

- ~~[[ 0.5, 0, 0.5 ], [ 0, 1, 2 ], [ 0.5, 0, 0.5 ]]~~  
- [[ 0.5, 0, 0.5 ], [ -2, 1, 0 ], [ 0.5, 0, -0.5 ]] ✔️  
- ~~[[ 0.5, 0, 0.5 ], [ 2, 1, 0 ], [ 0.5, 0, 0.5 ]]~~  
- ~~It is singular~~  

### Question #5  
Is the following matrix invertible? If so, what is its inverse?

[[ 2, 1, 2 ]  
  [ 1, 0, 1 ]  
  [ 4, 1, 4 ]]

- ~~[[ 4, 1, 2 ], [ 1, 0, 1 ], [ 4, 1, 2 ]]~~  
- ~~[[ 2, 1, 4 ], [ 1, 0, 1 ], [ 2, 1, 4 ]]~~  
- ~~[[ 4, 1, 4 ], [ 1, 0, 1 ], [ 2, 1, 2 ]]~~  
- It is singular ✔️  

### Question #6  
Given  
A = [[-2, -4, 2],  
      [-2, 1, 2],  
      [4, 2, 5]]  
v = [[2], [-3], [-1]]  
Where v is an eigenvector of A, calculate A¹⁰v

- [[118098], [-177147], [-59049]] ✔️  
- ~~[[2097152], [-3145728], [-1048576]]~~  
- ~~[[2048], [-3072], [-1024]]~~  
- ~~None of the above~~  

### Question #7  
Which of the following are also eigenvalues (λ) and eigenvectors (v) of A where  
A = [[-2, -4, 2],  
      [-2, 1, 2],  
      [4, 2, 5]]

- ~~λ = 5; v = [[2], [1], [1]]~~  
- λ = -5; v = [[-2], [-1], [1]] ✔️  
- ~~λ = -3; v = [[4], [-2], [3]]~~  
- λ = 6; v = [[1], [6], [16]]✔️  

### Question #8  
What is the definiteness of the following matrix:

[[ -1, 2, 0 ]  
  [ 2, -5, 2 ]  
  [ 0, 2, -6 ]]

- ~~Positive definite~~  
- ~~Positive semi-definite~~  
- ~~Negative semi-definite~~  
- Negative definite ✔️  
- ~~Indefinite~~  

### Question #9  
What is the definiteness of the following matrix:

[[ 2, 2, 1 ]  
  [ 2, 1, 3 ]  
  [ 1, 3, 8 ]]

- ~~Positive definite~~  
- ~~Positive semi-definite~~  
- ~~Negative semi-definite~~  
- ~~Negative definite~~  
- Indefinite ✔️  

### Question #10  
What is the definiteness of the following matrix:

[[ 2, 1, 1 ]  
  [ 1, 2, -1 ]  
  [ 1, -1, 2 ]]

- ~~Positive definite~~   
- ~~Positive semi-definite~~  
- ~~Negative semi-definite~~  
- ~~Negative definite~~  
- Indefinite ✔️  


---
 # 📝 Tasks

### 0. Determinant

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Write a function `def determinant(matrix):` that calculates the **determinant** of a given matrix:

- **Parameters:**
  - `matrix` is a **list of lists** whose determinant should be calculated.

- **Validation:**
  - If `matrix` is not a list of lists, raise a `TypeError` with the message:
    ```
    matrix must be a list of lists
    ```
  - If `matrix` is not square (i.e., the number of rows and columns are not equal), raise a `ValueError` with the message:
    ```
    matrix must be a square matrix
    ```
  - The list `[[]]` represents a `0x0` matrix.

- **Returns:**
  - The function should return the determinant of the matrix.

- **Notes:**
  - The determinant is only defined for square matrices.
  - You may use recursive expansion or another method to compute the determinant.

- **Example:**
  - For a matrix `[[1, 2], [3, 4]]`, the determinant is calculated as `1 * 4 - 2 * 3 = -2`.


```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/advanced_linear_algebra#./test_files/0-main.py
1
5
-2
0
192
matrix must be a list of lists
matrix must be a square matrix
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/advanced_linear_algebra#
```
<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 1. Minor

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Write a function `def minor(matrix):` that calculates the **minor matrix** of a given matrix:

- **Parameters:**
  - `matrix` is a list of lists whose minor matrix should be calculated.

- **Validation:**
  - If `matrix` is not a list of lists, raise a `TypeError` with the message:
    ```
    matrix must be a list of lists
    ```
  - If `matrix` is not square or is empty, raise a `ValueError` with the message:
    ```
    matrix must be a non-empty square matrix
    ```

- **Returns:**
  - The minor matrix of `matrix` as a list of lists.


```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/advanced_linear_algebra#./test_files/1-main.py
[[1]]
[[4, 3], [2, 1]]
[[1, 1], [1, 1]]
[[-12, -36, 0], [10, -34, -32], [47, 13, -16]]
matrix must be a list of lists
matrix must be a non-empty square matrix
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/advanced_linear_algebra#
```

<p align="center">⭐⭐⭐⭐⭐⭐</p>


### 2. Cofactor

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Write a function `def cofactor(matrix):` that calculates the **cofactor matrix** of a given matrix:

- **Parameters:**
  - `matrix` is a list of lists whose cofactor matrix should be calculated.

- **Validation:**
  - If `matrix` is not a list of lists, raise a `TypeError` with the message:
    ```
    matrix must be a list of lists
    ```
  - If `matrix` is not square or is empty, raise a `ValueError` with the message:
    ```
    matrix must be a non-empty square matrix
    ```

- **Returns:**
  - The cofactor matrix of `matrix` as a list of lists.


```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/advanced_linear_algebra#./test_files/1-main.py
[[1]]
[[4, 3], [2, 1]]
[[1, 1], [1, 1]]
[[-12, -36, 0], [10, -34, -32], [47, 13, -16]]
matrix must be a list of lists
matrix must be a non-empty square matrix
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/advanced_linear_algebra#
```

<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 3. Adjugate

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Write a function `def adjugate(matrix):` that calculates the **adjugate matrix** of a given matrix:

- **Parameters:**
  - `matrix` is a list of lists whose adjugate matrix should be calculated.

- **Validation:**
  - If `matrix` is not a list of lists, raise a `TypeError` with the message:
    ```
    matrix must be a list of lists
    ```
  - If `matrix` is not square or is empty, raise a `ValueError` with the message:
    ```
    matrix must be a non-empty square matrix
    ```

- **Returns:**
  - The adjugate matrix of `matrix` as a list of lists.

```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/advanced_linear_algebra#./test_files/3-main.py
[[1]]
[[4, -2], [-3, 1]]
[[1, -1], [-1, 1]]
[[-12, -10, 47], [36, -34, -13], [0, 32, -16]]
matrix must be a list of lists
matrix must be a non-empty square matrix
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/advanced_linear_algebra#
```

<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 4. Inverse

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Write a function `def inverse(matrix):` that calculates the **inverse** of a given matrix:

- **Parameters:**
  - `matrix` is a list of lists whose inverse should be calculated.

- **Validation:**
  - If `matrix` is not a list of lists, raise a `TypeError` with the message:
    ```
    matrix must be a list of lists
    ```
  - If `matrix` is not square or is empty, raise a `ValueError` with the message:
    ```
    matrix must be a non-empty square matrix
    ```

- **Returns:**
  - The inverse of `matrix` as a list of lists, or `None` if the matrix is **singular** (i.e., not invertible).

- **Notes:**
  - You must calculate the determinant first.
  - If the determinant is 0, the matrix has no inverse.
  - You should use your previously created `adjugate(matrix)` function.
```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/advanced_linear_algebra#./test_files/4-main.py
[[0.2]]
[[-2.0, 1.0], [1.5, -0.5]]
None
[[-0.0625, -0.052083333333333336, 0.24479166666666666], [0.1875, -0.17708333333333334, -0.06770833333333333], [0.0, 0.16666666666666666, -0.08333333333333333]]
matrix must be a list of lists
matrix must be a non-empty square matrix
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/advanced_linear_algebra#
```
<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 5. Definiteness

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Write a function `def definiteness(matrix):` that calculates the **definiteness** of a given matrix:

- **Parameters:**
  - `matrix` is a `numpy.ndarray` of shape `(n, n)` whose definiteness should be calculated.

- **Validation:**
  - If `matrix` is not a `numpy.ndarray`, raise a `TypeError` with the message:
    ```
    matrix must be a numpy.ndarray
    ```
  - If `matrix` is not a valid matrix (non-square or other invalid form), return `None`.

- **Returns:**
  - The function should return one of the following strings based on the matrix's definiteness:
    - **Positive definite**: If all eigenvalues are positive.
    - **Positive semi-definite**: If all eigenvalues are non-negative.
    - **Negative semi-definite**: If all eigenvalues are non-positive.
    - **Negative definite**: If all eigenvalues are negative.
    - **Indefinite**: If eigenvalues are mixed (some positive, some negative).

- **Notes:**
  - You may use `numpy` to calculate the eigenvalues of the matrix.
  - The matrix is valid for definiteness calculation if it is square (i.e., the same number of rows and columns).
  - The classification of the definiteness depends on the eigenvalues of the matrix.

- **Example:**
  - For a matrix with all positive eigenvalues, return `"Positive definite"`.
  - For a matrix with mixed positive and negative eigenvalues, return `"Indefinite"`.

```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/advanced_linear_algebra#./test_files/5-main.py
Positive definite
Positive semi-definite
Negative semi-definite
Negative definite
Indefinite
None
None
matrix must be a numpy.ndarray
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/advanced_linear_algebra#
```

---
# 📄 Files

| Task Number | Task Title                   |File                 | Priority                                                             |
|-------------|------------------------------|---------------------|----------------------------------------------------------------------|
| 0           | 0. Determinant        | `0-determinant.py`   | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |
| 1           | 1. Minor              | `1-minor.py` | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |
| 2           | 2. Cofactor           | `2-cofactor.py`     | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |
| 3           | 3. Adjugate           | `3-adjugate.py`    | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |
| 4           | 4. Inverse            | `4-inverse.py`   | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)  |
| 5           |5. Definiteness       | `5-definiteness.py` | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)    |

---

# 📊 Project Summary

This project aims to implement and analyze key concepts in advanced linear algebra, including **minor**, **cofactor**, **adjugate**, **inverse**, and matrix definiteness. Through Python scripts, the project demonstrates how to calculate these matrix properties and how they relate to the determinant and other matrix characteristics.

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
