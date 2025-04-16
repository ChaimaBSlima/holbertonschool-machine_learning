<h1><p align="center"> Bayesian Probability </h1></p></font>

<p align="center">
  <img src="" alt="Image"/>
</p>


# 📚 Resources

Read or watch:
- [Bayesian probability](https://en.wikipedia.org/wiki/Bayesian_probability)
- [Bayesian statistics](https://en.wikipedia.org/wiki/Bayesian_statistics)
- [Bayes’ Theorem - The Simplest Case](https://www.youtube.com/watch?v=XQoLVl31ZfQ)
- [A visual guide to Bayesian thinking](https://www.youtube.com/watch?v=BrK7X_XlGB8)
- [Base Rates](hhttps://onlinestatbook.com/2/probability/base_rates.html)
- [Bayesian statistics: a comprehensive course](https://www.youtube.com/playlist?list=PLFDbGp5YzjqXQ4oE4w9GVWdiokWB9gEpm)
- [Bayes’ rule - an intuitive explanation](https://www.youtube.com/watch?v=EbyUsf_jUjk&list=PLFDbGp5YzjqXQ4oE4w9GVWdiokWB9gEpm&index=15)
-  - [Bayes’ rule in statistics](https://www.youtube.com/watch?v=i567qvWejJA&list=PLFDbGp5YzjqXQ4oE4w9GVWdiokWB9gEpm&index=16)
-  - [Bayes’ rule in inference - likelihood](https://www.youtube.com/watch?v=c69a_viMRQU&list=PLFDbGp5YzjqXQ4oE4w9GVWdiokWB9gEpm&index=17)
-  - [Bayes’ rule in inference - the prior and denominator](https://www.youtube.com/watch?v=a5QDDZLGSXY&list=PLFDbGp5YzjqXQ4oE4w9GVWdiokWB9gEpm&index=18)
-  - [Bayes’ rule denominator: discrete and continuous](https://www.youtube.com/watch?v=QEzeLh6L9Tg&list=PLFDbGp5YzjqXQ4oE4w9GVWdiokWB9gEpm&index=25)
-  - [Bayes’ rule: why likelihood is not a probability](https://www.youtube.com/watch?v=sm60vapz2jQ&list=PLFDbGp5YzjqXQ4oE4w9GVWdiokWB9gEpm&index=26)

---

# 🎯 Learning Objectives  
- What is Bayesian Probability?  
- What is Bayes’ rule and how do you use it?  
- What is a base rate?  
- What is a prior?  
- What is a posterior?  
- What is a likelihood?  

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
- All your **modules** should have documentation:  
  `python3 -c 'print(__import__("my_module").__doc__)'`  
- All your **classes** should have documentation:  
  `python3 -c 'print(__import__("my_module").MyClass.__doc__)'`  
- All your **functions** (inside and outside a class) should have documentation:  
  `python3 -c 'print(__import__("my_module").my_function.__doc__)'`  
  and  
  `python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)'`  
- Unless otherwise noted, **you are not allowed to import any module** except:  
  `import numpy as np`  
- All your files must be **executable**  
- The length of your files will be tested using:  
  `wc`  

---
# ❓ Quiz:

### Question #0  
Bayes’ rule states that `P(A | B) = P(B | A) * P(A) / P(B)`  
What is **P(A | B)**?

- ~~Likelihood~~  
- ~~Marginal probability~~  
- Posterior probability ✔️  
- ~~Prior probability~~  

---

### Question #1  
Bayes’ rule states that `P(A | B) = P(B | A) * P(A) / P(B)`  
What is **P(B | A)**?

- Likelihood ✔️  
- ~~Marginal probability~~  
- ~~Posterior probability~~  
- ~~Prior probability~~  

---

### Question #2  
Bayes’ rule states that `P(A | B) = P(B | A) * P(A) / P(B)`  
What is **P(A)**?

- ~~Likelihood~~  
- ~~Marginal probability~~  
- ~~Posterior probability~~  
- Prior probability ✔️  

---

### Question #3  
Bayes’ rule states that `P(A | B) = P(B | A) * P(A) / P(B)`  
What is **P(B)**?

- ~~Likelihood~~  
- Marginal probability ✔️  
- ~~Posterior probability~~  
- ~~Prior probability~~ 

---

# 📝 Tasks

### 0. Mean and Covariance

### 0. Likelihood  

**File:** `0-likelihood.py` ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Write the function `def likelihood(x, n, P):` that calculates the likelihood of obtaining this data given various hypothetical probabilities of developing severe side effects:

- `x` is the number of patients that develop severe side effects  
- `n` is the total number of patients observed  
- `P` is a 1D `numpy.ndarray` containing the various hypothetical probabilities of developing severe side effects  

#### Requirements:

- If `n` is not a positive integer, raise a `ValueError` with the message:  
  `n must be a positive integer`  
- If `x` is not an integer that is greater than or equal to 0, raise a `ValueError` with the message:  
  `x must be an integer that is greater than or equal to 0`  
- If `x` is greater than `n`, raise a `ValueError` with the message:  
  `x cannot be greater than n`  
- If `P` is not a 1D `numpy.ndarray`, raise a `TypeError` with the message:  
  `P must be a 1D numpy.ndarray`  
- If any value in `P` is not in the range [0, 1], raise a `ValueError` with the message:  
  `All values in P must be in the range [0, 1]`  

#### Returns:
- A 1D `numpy.ndarray` containing the likelihood of obtaining the data, `x` and `n`, for each probability in `P`, respectively.
```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/bayesian_prob#./test_files/0-main.py
[0.00000000e+00 2.71330957e-04 8.71800070e-02 3.07345706e-03
 5.93701546e-07 1.14387595e-12 1.09257177e-20 6.10151799e-32
 9.54415702e-49 1.00596671e-78 0.00000000e+00]
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/bayesian_prob#
```
<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 1. Intersection  

**File:** `1-intersection.py` ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)


Based on `0-likelihood.py`, write the function `def intersection(x, n, P, Pr):` that calculates the intersection of obtaining this data with the various hypothetical probabilities:

- `x` is the number of patients that develop severe side effects  
- `n` is the total number of patients observed  
- `P` is a 1D `numpy.ndarray` containing the various hypothetical probabilities of developing severe side effects  
- `Pr` is a 1D `numpy.ndarray` containing the prior beliefs of `P`  

#### Requirements:

- If `n` is not a positive integer, raise a `ValueError` with the message:  
  `n must be a positive integer`  
- If `x` is not an integer that is greater than or equal to 0, raise a `ValueError` with the message:  
  `x must be an integer that is greater than or equal to 0`  
- If `x` is greater than `n`, raise a `ValueError` with the message:  
  `x cannot be greater than n`  
- If `P` is not a 1D `numpy.ndarray`, raise a `TypeError` with the message:  
  `P must be a 1D numpy.ndarray`  
- If `Pr` is not a `numpy.ndarray` with the same shape as `P`, raise a `TypeError` with the message:  
  `Pr must be a numpy.ndarray with the same shape as P`  
- If any value in `P` or `Pr` is not in the range [0, 1], raise a `ValueError` with the message:  
  `All values in {P} must be in the range [0, 1]`  
  - Replace `{P}` with the actual variable name causing the error  
- If `Pr` does not sum to 1, raise a `ValueError` with the message:  
  `Pr must sum to 1`  
- 💡 Hint: use `numpy.isclose` to check the sum  

#### Returns:
- A 1D `numpy.ndarray` containing the intersection of obtaining `x` and `n` with each probability in `P`, respectively.
```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/bayesian_prob#./test_files/1-main.py
[0.00000000e+00 2.46664506e-05 7.92545518e-03 2.79405187e-04
 5.39728678e-08 1.03988723e-13 9.93247059e-22 5.54683454e-33
 8.67650639e-50 9.14515194e-80 0.00000000e+00]
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/bayesian_prob#
```

<p align="center">⭐⭐⭐⭐⭐⭐</p>


### 2. Marginal Probability  

**File:** `2-marginal.py` ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Based on `1-intersection.py`, write the function `def marginal(x, n, P, Pr):` that calculates the **marginal probability** of obtaining the data:

- `x` is the number of patients that develop severe side effects  
- `n` is the total number of patients observed  
- `P` is a 1D `numpy.ndarray` containing the various hypothetical probabilities of patients developing severe side effects  
- `Pr` is a 1D `numpy.ndarray` containing the prior beliefs about `P`  

#### Requirements:

- If `n` is not a positive integer, raise a `ValueError` with the message:  
  `n must be a positive integer`  
- If `x` is not an integer that is greater than or equal to 0, raise a `ValueError` with the message:  
  `x must be an integer that is greater than or equal to 0`  
- If `x` is greater than `n`, raise a `ValueError` with the message:  
  `x cannot be greater than n`  
- If `P` is not a 1D `numpy.ndarray`, raise a `TypeError` with the message:  
  `P must be a 1D numpy.ndarray`  
- If `Pr` is not a `numpy.ndarray` with the same shape as `P`, raise a `TypeError` with the message:  
  `Pr must be a numpy.ndarray with the same shape as P`  
- If any value in `P` or `Pr` is not in the range [0, 1], raise a `ValueError` with the message:  
  `All values in {P} must be in the range [0, 1]`  
  - Replace `{P}` with the actual variable name causing the error  
- If `Pr` does not sum to 1, raise a `ValueError` with the message:  
  `Pr must sum to 1`  
  - Hint: use `numpy.isclose` to validate the sum  

#### Returns:
- A single float: the **marginal probability** of obtaining `x` and `n`.
```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/bayesian_prob#./test_files/2-main.py
0.008229580791426582
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/bayesian_prob#
```

<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 3. Posterior  

**File:** `3-posterior.py` ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Based on `2-marginal.py`, write the function `def posterior(x, n, P, Pr):` that calculates the **posterior probability** for the various hypothetical probabilities of developing severe side effects given the data:

- `x` is the number of patients that develop severe side effects  
- `n` is the total number of patients observed  
- `P` is a 1D `numpy.ndarray` containing the various hypothetical probabilities of developing severe side effects  
- `Pr` is a 1D `numpy.ndarray` containing the prior beliefs of `P`  

#### Requirements:

- If `n` is not a positive integer, raise a `ValueError` with the message:  
  `n must be a positive integer`  
- If `x` is not an integer that is greater than or equal to 0, raise a `ValueError` with the message:  
  `x must be an integer that is greater than or equal to 0`  
- If `x` is greater than `n`, raise a `ValueError` with the message:  
  `x cannot be greater than n`  
- If `P` is not a 1D `numpy.ndarray`, raise a `TypeError` with the message:  
  `P must be a 1D numpy.ndarray`  
- If `Pr` is not a `numpy.ndarray` with the same shape as `P`, raise a `TypeError` with the message:  
  `Pr must be a numpy.ndarray with the same shape as P`  
- If any value in `P` or `Pr` is not in the range [0, 1], raise a `ValueError` with the message:  
  `All values in {P} must be in the range [0, 1]`  
  - Replace `{P}` with the actual variable name causing the error  
- If `Pr` does not sum to 1, raise a `ValueError` with the message:  
  `Pr must sum to 1`  
  - Use `numpy.isclose` to validate the sum  

#### Returns:
- A 1D `numpy.ndarray` containing the **posterior probability** of each probability in `P` given `x` and `n`, respectively.
```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/bayesian_prob#./test_files/3-main.py
[0.00000000e+00 2.99729127e-03 9.63044824e-01 3.39513268e-02
 6.55839819e-06 1.26359684e-11 1.20692303e-19 6.74011797e-31
 1.05430721e-47 1.11125368e-77 0.00000000e+00]
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/bayesian_prob#
```

<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 4. Continuous Posterior

**File:** `100-continuous.py` ![Advanced](https://img.shields.io/badge/advanced-🚀-blueviolet)


Based on `3-posterior.py`, write a function `def posterior(x, n, p1, p2):` that calculates the posterior probability that the probability of developing severe side effects falls within a specific range given the data:

- `x` is the number of patients that develop severe side effects  
- `n` is the total number of patients observed  
- `p1` is the lower bound on the range  
- `p2` is the upper bound on the range  
- You can assume the prior beliefs of `p` follow a uniform distribution

### Requirements:
- If `n` is not a positive integer, raise a `ValueError` with the message  
  `n must be a positive integer`
- If `x` is not an integer that is greater than or equal to 0, raise a `ValueError` with the message  
  `x must be an integer that is greater than or equal to 0`
- If `x` is greater than `n`, raise a `ValueError` with the message  
  `x cannot be greater than n`
- If `p1` or `p2` are not floats within the range [0, 1], raise a `ValueError` with the message  
  `{p} must be a float in the range [0, 1]` where `{p}` is the corresponding variable
- If `p2 <= p1`, raise a `ValueError` with the message  
  `p2 must be greater than p1`

- The only import you are allowed to use is:  
`from scipy import special`

### Returns:
- The posterior probability that `p` is within the range `[p1, p2]` given `x` and `n`

💡 Hint: See [Binomial Distribution](https://en.wikipedia.org/wiki/Binomial_distribution) and [Beta Distribution](https://en.wikipedia.org/wiki/Beta_distribution)
```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/bayesian_prob#./test_files/100-main.py 
0.6098093274896221
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/bayesian_prob#
```

---

# 📊 Project Summary

This project focuses on calculating various probabilities using **Bayesian Inference**, particularly in the context of patients developing severe side effects from a treatment. The core concepts involve understanding and applying the **binomial distribution**, **Bayes' Theorem**, and the **Beta distribution** to estimate and update beliefs about probabilities given observed data.

---

# ℹ️ Random Information 

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

---

# 👩‍💻 Authors
Tasks by [Holberton School](https://www.holbertonschool.com/)

**Chaima Ben Slima** - Holberton School Student, ML Developer

[GitHub](https://github.com/ChaimaBSlima)
[Linkedin](https://www.linkedin.com/in/chaima-ben-slima-35477120a/)
