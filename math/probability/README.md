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
- Who is Carl Friedrich Gauss?  
- What is a joint/multivariate distribution?  
- What is a covariance?  
- What is a correlation coefficient?  
- What is a covariance matrix?  
- What is a multivariate Gaussian distribution?  

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

