<h1><p align="center"> Probability </h1></p></font>

<p align="center">
  <img src="https://github.com/user-attachments/assets/d71723c9-af7c-4c87-9c1d-cd6f5bd829ff" alt="Image"/>
</p>


# 📚 Resources

Read or watch:
- [Probability](https://en.wikipedia.org/wiki/Probability)
- [Basic Concepts](https://onlinestatbook.com/2/probability/basic.html)
- [Intro to probability 1: Basic notation](https://www.youtube.com/watch?v=TkU3BvDAOtQ)
- [Intro to probability 2: Independent and disjoint](https://www.youtube.com/watch?v=GnWHt9nqwBA)
- [Intro to Probability 3: General Addition Rule; Union; OR](https://www.youtube.com/watch?v=TyAaVGR4MrA)
- [Intro to Probability 4: General multiplication rule; Intersection; AND](https://www.youtube.com/watch?v=wB-ZG9bgPXY)
- [Permutations and Combinations](https://onlinestatbook.com/2/probability/permutations.html)
- [Probability distribution](https://en.wikipedia.org/wiki/Probability_distribution)
- [Probability Theory](https://medium.com/data-science/probability-fundamentals-of-machine-learning-part-1-a156b4703e69)
- [Cumulative Distribution Functions](https://www.oreilly.com/library/view/think-stats-2nd/9781491907344/ch04.html)
- [Common Probability Distributions: The Data Scientist’s Crib Sheet](https://medium.com/@srowen/common-probability-distributions-347e6b945ce4)
- [NORMAL MODEL PART 1 — EMPIRICAL RULE](https://www.youtube.com/watch?v=xgolpGrAZWo&list=PLFGZup_HuWTtIs0Xbzt7vDoFrnZxN4VXT&index=22)
- [Normal Distribution](https://www.mathsisfun.com/data/standard-normal-distribution.html)
- [Variance](https://en.wikipedia.org/wiki/Variance)
- [Variance (Concept)](https://www.youtube.com/watch?v=2eP14USYwtg)
- [Binomial Distribution](https://onlinestatbook.com/2/probability/binomial.html)
- [Poisson Distribution](https://onlinestatbook.com/2/probability/poisson.html)
- [Hypergeometric Distribution](https://onlinestatbook.com/2/probability/hypergeometric.html)

As references:
- [numpy.random.poisson](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.poisson.html)
- [numpy.random.exponential](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.exponential.html)
- [numpy.random.normal](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.binomial.html)
- [numpy.random.binomial](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.normal.html)
- [erf](https://mathworld.wolfram.com/Erf.html)


---
# 🎯 Learning Objectives

At the end of this project, you are expected to be able to [explain to anyone](https://fs.blog/feynman-learning-technique/), without the help of Google:

### General

- What is probability?  
- Basic probability notation  
- What is independence? What is disjoint?  
- What is a union? intersection?  
- What are the general addition and multiplication rules?  
- What is a probability distribution?  
- What is a probability distribution function? probability mass function?  
- What is a cumulative distribution function?  
- What is a percentile?  
- What is mean, standard deviation, and variance?  
- Common probability distributions  

---

# 🧾 Requirements

### General

- Allowed editors: `vi`, `vim`, `emacs`  
- All your files will be interpreted/compiled on **Ubuntu 20.04 LTS** using **python3 (version 3.9)**  
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

# ➗ Mathematical Approximations

For the following tasks, you will have to use various irrational numbers and functions. Since you are not able to import any libraries, please use the following approximations:

- π = 3.1415926536  
- e = 2.7182818285
- <p align="left">
  <img src="https://github.com/user-attachments/assets/8df11ced-6714-4f27-958e-7012e4a528ee" alt="Image"/>
</p>

---
# ❓ Quiz:

### Question #0  
What does the expression P(A | B) represent?

- ~~The probability of A and B~~  
- ~~The probability of A or B~~  
- ~~The probability of A and not B~~  
- The probability of A given B ✔️  

### Question #1  
What does the expression P(A ∩ B') represent?

- ~~The probability of A and B~~  
- ~~The probability of A or B~~  
- The probability of A and not B ✔️  
- ~~The probability of A given B~~  

### Question #2  
What does the expression P(A ∩ B) represent?

- The probability of A and B ✔️  
- ~~The probability of A or B~~  
- ~~The probability of A and not B~~  
- ~~The probability of A given B~~  

### Question #3  
What does the expression P(A ∪ B) represent?

- ~~The probability of A and B~~  
- The probability of A or B ✔️  
- ~~The probability of A and not B~~  
- ~~The probability of A given B~~  

### Question #4

<p align="center">
  <img src="https://github.com/user-attachments/assets/b0204f3e-ad40-4cd4-87bc-9597aae42912" alt="Image"/>
</p>

The above image displays the normal distribution of male heights. What is the mode height?

- ~~5'6"~~
- ~~5'8"~~   
- 5'10" ✔️  
- ~~6’~~  
- ~~6'2"~~  

### Question #5

<p align="center">
  <img src="https://github.com/user-attachments/assets/b0204f3e-ad40-4cd4-87bc-9597aae42912" alt="Image"/>
</p>

The above image displays the normal distribution of male heights. What is the standard deviation?

- ~~1"~~
- ~~2"~~  
- 4" ✔️  
- ~~8"~~  

### Question #6

<p align="center">
  <img src="https://github.com/user-attachments/assets/b0204f3e-ad40-4cd4-87bc-9597aae42912" alt="Image"/>
</p>

The above image displays the normal distribution of male heights. What is the variance?

- ~~4"~~  
- ~~8"~~  
- 16" ✔️
- ~~64"~~  

### Question #7

<p align="center">
  <img src="https://github.com/user-attachments/assets/b0204f3e-ad40-4cd4-87bc-9597aae42912" alt="Image"/>
</p>

The above image displays the normal distribution of male heights. If a man is 6'6", what percentile would he be in?

- ~~84th percentile~~  
- ~~95th percentile~~  
- 97.25th percentile ✔️
- ~~99.7th percentile~~   

### Question #8 

<p align="center">
  <img src="https://github.com/user-attachments/assets/8ced9886-0bfb-490c-aedd-029720cf427c" alt="Image"/>
</p>

What type of distribution is displayed above?

- ~~Gaussian~~   
- ~~Hypergeometric~~  
- ~~Chi-Squared~~  
- Poisson ✔️  

### Question #9

<p align="center">
  <img src="https://github.com/user-attachments/assets/07410be9-fd40-41bd-b94e-09713e27071d" alt="Image"/>
</p>

What type of distribution is displayed above?

- ~~Gaussian~~   
- Hypergeometric ✔️ 
- ~~Chi-Squared~~  
- ~~Poisson~~

### Question #10  
What is the difference between a PDF and a PMF?

- ~~PDF is for discrete variables while PMF is for continuous variables~~  
- PDF is for continuous variables while PMF is for discrete variables ✔️  
- ~~There is no difference~~  

### Question #11  
For a given distribution, the value at the 50th percentile is always:

- ~~mean~~  
- median ✔️  
- ~~mode~~  
- ~~all of the above~~  

### Question #12  
For a given distribution, the CDF(x) where x ∈ X:

- ~~The probability that X = x~~  
- The probability that X <= x ✔️  
- The percentile of x ✔️  
- ~~The probability that X >= x~~  

---
 # 📝 Tasks

### 0. Initialize Poisson

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Create a class `Poisson` that represents a Poisson distribution:

- **Class constructor:** `def __init__(self, data=None, lambtha=1.):`
  - `data` is a list of the data to be used to estimate the distribution  
  - `lambtha` is the expected number of occurrences in a given time frame  
  - Sets the instance attribute `lambtha`  
  - Saves `lambtha` as a float  

- If `data` is not given (i.e., `None`):  
  - Use the given `lambtha`  
  - If `lambtha` is not a positive value or equals 0, raise a `ValueError` with the message:  
    **"lambtha must be a positive value"**

- If `data` is given:
  - Calculate the `lambtha` from `data`
  - If `data` is not a list, raise a `TypeError` with the message:  
    **"data must be a list"**
  - If `data` does not contain at least two data points, raise a `ValueError` with the message:  
    **"data must contain multiple values"**

```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/probability#./test_files/0-main.py
Lambtha: 4.84
Lambtha: 5.0
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/probability#
```
<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 1. Poisson PMF

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Update the class `Poisson`:

- **Instance method:** `def pmf(self, k):`
  - Calculates the value of the PMF for a given number of “successes”  
  - `k` is the number of “successes”  
  - If `k` is not an integer, convert it to an integer  
  - If `k` is out of range, return `0`  

#### Returns:
- the PMF value for `k`

```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/probability#./test_files/1-main.py
P(9): 0.03175849616802446
P(9): 0.036265577412911795
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/probability#
```

<p align="center">⭐⭐⭐⭐⭐⭐</p>


### 2. Poisson CDF

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Update the class `Poisson`:

- **Instance method:** `def cdf(self, k):`
  - Calculates the value of the CDF for a given number of “successes”  
  - `k` is the number of “successes”  
  - If `k` is not an integer, convert it to an integer  
  - If `k` is out of range, return `0`  

#### Returns:
- the CDF value for `k`

```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/probability#./test_files/2-main.py
F(9): 0.9736102067423525
F(9): 0.9681719426208609
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/probability#
```

<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 3. Initialize Exponential

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Create a class `Exponential` that represents an exponential distribution:

- **Class constructor:** `def __init__(self, data=None, lambtha=1.):`
  - `data` is a list of the data to be used to estimate the distribution  
  - `lambtha` is the expected number of occurrences in a given time frame  
  - Sets the instance attribute `lambtha`  
  - Saves `lambtha` as a float  

- If `data` is not given (i.e., `None`):
  - Use the given `lambtha`  
  - If `lambtha` is not a positive value, raise a `ValueError` with the message:  
    **"lambtha must be a positive value"**

- If `data` is given:
  - Calculate the `lambtha` from `data`
  - If `data` is not a list, raise a `TypeError` with the message:  
    **"data must be a list"**
  - If `data` does not contain at least two data points, raise a `ValueError` with the message:  
    **"data must contain multiple values"**

```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/probability#./test_files/3-main.py
Lambtha: 2.1771114730906937
Lambtha: 2.0
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/probability#
```

<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 4. Exponential PDF

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Update the class `Exponential`:

- **Instance method:** `def pdf(self, x):`
  - Calculates the value of the PDF for a given time period  
  - `x` is the time period  
  - If `x` is out of range, return `0`  

#### Returns:
- the PDF value for `x`

```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/probability#./test_files/4-main.py
f(1): 0.24681591903431568
f(1): 0.2706705664650693
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/probability#
```
<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 5. Exponential CDF

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Update the class `Exponential`:

- **Instance method:** `def cdf(self, x):`
  - Calculates the value of the CDF for a given time period  
  - `x` is the time period  
  - If `x` is out of range, return `0`  

#### Returns:
- the CDF value for `x`

```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/probability#./test_files/5-main.py
F(1): 0.886631473819791
F(1): 0.8646647167674654
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/probability#
```
<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 6. Initialize Normal

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Create a class `Normal` that represents a normal distribution:

- **Class constructor:** `def __init__(self, data=None, mean=0., stddev=1.):`
  - `data` is a list of the data to be used to estimate the distribution  
  - `mean` is the mean of the distribution  
  - `stddev` is the standard deviation of the distribution  
  - Sets the instance attributes `mean` and `stddev`  
  - Saves `mean` and `stddev` as floats  

- If `data` is not given (i.e., `None`):
  - Use the given `mean` and `stddev`
  - If `stddev` is not a positive value or equals to `0`, raise a `ValueError` with the message:  
    **"stddev must be a positive value"**

- If `data` is given:
  - Calculate the `mean` and `standard deviation` from `data`
  - If `data` is not a list, raise a `TypeError` with the message:  
    **"data must be a list"**
  - If `data` does not contain at least two data points, raise a `ValueError` with the message:  
    **"data must contain multiple values"**


```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/probability#./test_files/6-main.py
Mean: 70.59808015534485 , Stddev: 10.078822447165797
Mean: 70.0 , Stddev: 10.0
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/probability#
```
<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 7. Normalize Normal

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Update the class `Normal`:

- **Instance method:** `def z_score(self, x):`
  - Calculates the z-score of a given `x`-value  
  - `x` is the `x`-value  

- **Instance method:** `def x_value(self, z):`
  - Calculates the `x`-value of a given `z`-score  
  - `z` is the `z`-score  

#### Returns:
- For `z_score`: the z-score of `x`
- For `x_value`: the x-value of `z`

```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/probability#./test_files/7-main.py
Z(90): 1.9250185174272068
X(2): 90.75572504967644

Z(90): 2.0
X(2): 90.0
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/probability#
```
<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 8. Normal PDF

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Update the class `Normal`:

- **Instance method:** `def pdf(self, x):`
  - Calculates the value of the PDF for a given `x`-value  
  - `x` is the `x`-value  

#### Returns:
- the PDF value for `x`

```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/probability#./test_files/8-main.py
PSI(90): 0.006206096804434349
PSI(90): 0.005399096651147344
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/probability#
```

<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 9. Normal CDF

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Update the class `Normal`:

- **Instance method:** `def cdf(self, x):`
  - Calculates the value of the CDF for a given `x`-value  
  - `x` is the `x`-value  

#### Returns:
- the CDF value for `x`


```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/probability#./test_files/9-main.py
PHI(90): 0.9829020110852376
PHI(90): 0.9922398930659416
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/probability#
```
<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 10. Initialize Binomial

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Create a class `Binomial` that represents a binomial distribution:

- **Class constructor:** `def __init__(self, data=None, n=1, p=0.5):`
  - `data` is a list of the data to be used to estimate the distribution  
  - `n` is the number of Bernoulli trials  
  - `p` is the probability of a “success”  

- Sets the instance attributes `n` and `p`  
  - Saves `n` as an integer and `p` as a float  

- If `data` is not given (i.e., `None`):
  - Use the given `n` and `p`
  - If `n` is not a positive value, raise a `ValueError` with the message:  
    **"n must be a positive value"**
  - If `p` is not a valid probability, raise a `ValueError` with the message:  
    **"p must be greater than 0 and less than 1"**

- If `data` is given:
  - Calculate `n` and `p` from `data`
  - Round `n` to the nearest integer (rounded, not casting!)
  - Calculate `p` first and then recalculate `n`
  - If `data` is not a list, raise a `TypeError` with the message:  
    **"data must be a list"**
  - If `data` does not contain at least two data points, raise a `ValueError` with the message:  
    **"data must contain multiple values"**


```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/probability#./test_files/10-main.py
n: 50 p: 0.606
n: 50 p: 0.6
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/probability#
```
<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 11. Binomial PMF

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Update the class `Binomial`:

- **Instance method:** `def pmf(self, k):`
  - Calculates the value of the PMF for a given number of “successes”  
  - `k` is the number of “successes”  

- If `k` is not an integer, convert it to an integer  
- If `k` is out of range, return 0  

#### Returns:
- the PMF value for `k`

```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/probability#./test_files/11-main.py
P(30): 0.11412829839570347
P(30): 0.114558552829524
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/probability#
```
<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 12. Binomial CDF

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Update the class `Binomial`:

- **Instance method:** `def cdf(self, k):`
  - Calculates the value of the CDF for a given number of “successes”  
  - `k` is the number of “successes”  

- If `k` is not an integer, convert it to an integer  
- If `k` is out of range, return 0  

#### Returns:
- the CDF value for `k`  
- **Hint:** Use the `pmf` method

```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/probability#./test_files/12-main.py
F(30): 0.5189392017296368
F(30): 0.5535236207894576
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/probability#
```
---
# 📄 Files

| Task Number | Task Title                   |File                 | Priority                                                             |
|-------------|------------------------------|---------------------|----------------------------------------------------------------------|
| 0           | 0. Initialize Poisson        | `poisson.py`   | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |
| 1           | 1. Poisson PMF              | `poisson.py` | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |
| 2           | 2. Poisson CDF                | `poisson.py`     | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |
| 3           | 3. Initialize Exponential     | `exponential.py`    | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |
| 4           | 4. Exponential PDF       | `exponential.py`   | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)  |
| 5           | 5. Exponential CDF       | `exponential.py` | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)    |
| 6           | 6. Initialize Normal     | `normal.py`     | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)     |
| 7           | 7. Normalize Normal         | `normal.py`    | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)   |
| 8           | 8. Normal PDF               | `normal.py`    | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)   |
| 9           | 9. Normal CDF               | `normal.py`    | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)   |
| 10          | 10. Initialize Binomial     | `binomial.py`    | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |
| 11          | 11. Binomial PMF            | `binomial.py`    | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |
| 12          | 12. Binomial CDF            | `binomial.py`    | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |

---

# 📊 Project Summary

The goal of this project is to implement and explore key probability distributions using Python. The focus is on four fundamental probability distributions: **Poisson**, **Exponential**, **Normal**, and **Binomial**.
Through practical Python code, the project demonstrates how these distributions can be generated, visualized, and applied to real-world problems.

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
