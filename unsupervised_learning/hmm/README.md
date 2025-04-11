<h1><p align="center"> Hidden Markov Models </h1></p></font>


# 📚 Resources
Read or watch:
- [Markov property](https://en.wikipedia.org/wiki/Markov_property)
- [Markov Chain](https://en.wikipedia.org/wiki/Markov_chain)
- [Properties of Markov Chains](https://pdf4pro.com/download/10-1-properties-of-markov-chains-governors-state-university-4c8fed.html)
- [Markov Chains](https://chance.dartmouth.edu/teaching_aids/books_articles/probability_book/Chapter11.pdf)
- [Markov Matrices](https://people.math.harvard.edu/~knill/teaching/math19b_2011/handouts/lecture33.pdf)
- [1.3 Convergence of Regular Markov Chains](http://www.tcs.hut.fi/Studies/T-79.250/tekstit/lecnotes_02.pdf)
- [Markov Chains, Part 1](https://www.youtube.com/watch?v=uvYTGEZQTEs)
- [Markov Chains, Part 2](https://www.youtube.com/watch?v=jtHBfLtMq4U)
- [Markov Chains, Part 3](https://www.youtube.com/watch?v=P8DuuiINAo4)
- [Markov Chains, Part 4](https://www.youtube.com/watch?v=31X-M4okAI0)
- [Markov Chains, Part 5](https://www.youtube.com/watch?v=-kwnnNSGFMc)
- [Markov Chains, Part 7](https://www.youtube.com/watch?v=bTeKu7WdbT8)
- [Markov Chains, Part 8](https://www.youtube.com/watch?v=BsOkOaB8SFk)
- [Markov Chains, Part 9](https://www.youtube.com/watch?v=qhnFHnLkrfA)
- [Hidden Markov model](https://en.wikipedia.org/wiki/Hidden_Markov_model)
- [Hidden Markov Models](https://web.stanford.edu/~jurafsky/slp3/A.pdf)
- [(ML 14.1) Markov models - motivating examples](https://www.youtube.com/watch?v=7KGdE2AK_MQ&list=PLD0F06AA0D2E8FFBA&index=97)
- [(ML 14.2) Markov chains (discrete-time) (part 1)](https://www.youtube.com/watch?v=WUjt98HcHlk&list=PLD0F06AA0D2E8FFBA&index=98)
- [(ML 14.3) Markov chains (discrete-time) (part 2)](https://www.youtube.com/watch?v=j6OUj9tleVM&list=PLD0F06AA0D2E8FFBA&index=99)
- [(ML 14.4) Hidden Markov models (HMMs) (part 1)](https://www.youtube.com/watch?v=TPRoLreU9lA&list=PLD0F06AA0D2E8FFBA&index=100)
- [(ML 14.5) Hidden Markov models (HMMs) (part 2)](https://www.youtube.com/watch?v=M_IIW0VYMEA&list=PLD0F06AA0D2E8FFBA&index=101)
- [(ML 14.6) Forward-Backward algorithm for HMMs](https://www.youtube.com/watch?v=7zDARfKVm7s&list=PLD0F06AA0D2E8FFBA&index=102)
- [(ML 14.7) Forward algorithm (part 1)](https://www.youtube.com/watch?v=M7afek1nEKM&list=PLD0F06AA0D2E8FFBA&index=103)
- [(ML 14.8) Forward algorithm (part 2)](https://www.youtube.com/watch?v=MPmrFu4jFk4&list=PLD0F06AA0D2E8FFBA&index=104)
- [(ML 14.9) Backward algorithm](https://www.youtube.com/watch?v=jwYuk)
- [(ML 14.10) Underflow and the log-sum-exp trick](https://www.youtube.com/watch?v=-RVM21Voo7Q&list=PLD0F06AA0D2E8FFBA&index=106)
- [(ML 14.11) Viterbi algorithm (part 1)](https://www.youtube.com/watch?v=RwwfUICZLsA&list=PLD0F06AA0D2E8FFBA&index=107)
- [(ML 14.12) Viterbi algorithm (part 2)](https://www.youtube.com/watch?v=t3JIk3Jgifs&list=PLD0F06AA0D2E8FFBA&index=108)

---

# 🎯 Learning Objectives

- What is the Markov property?  
- What is a Markov chain?  
- What is a state?  
- What is a transition probability/matrix?  
- What is a stationary state?  
- What is a regular Markov chain?  
- How to determine if a transition matrix is regular  
- What is an absorbing state?  
- What is a transient state?  
- What is a recurrent state?  
- What is an absorbing Markov chain?  
- What is a Hidden Markov Model?  
- What is a hidden state?  
- What is an observation?  
- What is an emission probability/matrix?  
- What is a Trellis diagram?  
- What is the Forward algorithm and how do you implement it?  
- What is decoding?  
- What is the Viterbi algorithm and how do you implement it?  
- What is the Forward-Backward algorithm and how do you implement it?  
- What is the Baum-Welch algorithm and how do you implement it?  

---

# 🧾 Requirements

### General

- Allowed editors: `vi`, `vim`, `emacs`  
- All your files will be interpreted/compiled on **Ubuntu 20.04 LTS** using **python3 (version 3.9)**  
- Your files will be executed with **numpy (version 1.25.2)**  
- All your files should end with a **new line**  
- The first line of all your files should be exactly: `#!/usr/bin/env python3`
- Your code should use the `pycodestyle` style (version 2.11.1)
- All your modules should have documentation (`python3 -c 'print(__import__("my_module").__doc__)'`)
- All your classes should have documentation (`python3 -c 'print(__import__("my_module").MyClass.__doc__)'`)
- All your functions (inside and outside a class) should have documentation (`python3 -c 'print(__import__("my_module").my_function.__doc__)'` and python3 -c `'print(__import__("my_module").MyClass.my_function.__doc__)'`)
- Unless otherwise noted, you are not allowed to import any module except `import numpy as np`
- All your files must be executable

---
# 📝 Tasks

### 0. Markov Chain

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Write the function `def markov_chain(P, s, t=1):` that determines the probability of a Markov chain being in a particular state after a specified number of iterations:

- `P` is a square 2D numpy.ndarray of shape `(n, n)` representing the transition matrix  
  `P[i, j]` is the probability of transitioning from state `i` to state `j`  
  `n` is the number of states in the Markov chain  
- `s` is a numpy.ndarray of shape `(1, n)` representing the probability of starting in each state  
- `t` is the number of iterations that the Markov chain has been through  

#### Returns:
 a numpy.ndarray of shape `(1, n)` representing the probability of being in a specific state after `t` iterations, or `None` on failure.
```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/unsupervised_learning/hmm#./test_files/0-main.py
[[0.2494929  0.26335362 0.23394185 0.25321163]]
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/unsupervised_learning/hmm#
```
<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 1. Regular Chains

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Write the function `def regular(P):` that determines the steady state probabilities of a regular Markov chain:

- `P` is a square 2D `numpy.ndarray` of shape `(n, n)` representing the transition matrix  
  `P[i, j]` is the probability of transitioning from state `i` to state `j`  
  `n` is the number of states in the Markov chain  

#### Returns:
 a `numpy.ndarray` of shape `(1, n)` containing the steady state probabilities, or `None` on failure.
```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/unsupervised_learning/hmm#./test_files/1-main.py
None
[[0.42857143 0.57142857]]
[[0.2494929  0.26335362 0.23394185 0.25321163]]
None
None
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/unsupervised_learning/hmm#
```

<p align="center">⭐⭐⭐⭐⭐⭐</p>


### 2. Absorbing Chains

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Write the function `def absorbing(P):` that determines if a Markov chain is absorbing:

- `P` is a square 2D `numpy.ndarray` of shape `(n, n)` representing the standard transition matrix  
  `P[i, j]` is the probability of transitioning from state `i` to state `j`  
  `n` is the number of states in the Markov chain  

#### Returns:
 `True` if it is absorbing, or `False` on failure.
```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/unsupervised_learning/hmm#./test_files/2-main.py
True
False
False
False
True
True
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/unsupervised_learning/hmm#
```

<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 3. The Forward Algorithm

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)   


Write the function `def forward(Observation, Emission, Transition, Initial):` that performs the forward algorithm for a hidden markov model:

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
- **F** is a `numpy.ndarray` of shape `(N, T)` containing the forward path probabilities.  
  **F[i, j]** is the probability of being in hidden state `i` at time `j` given the previous observations.
```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/unsupervised_learning/hmm#./test_files/3-main.py
1.7080966131859584e-214
[[0.00000000e+000 0.00000000e+000 2.98125000e-004 ... 0.00000000e+000
  0.00000000e+000 0.00000000e+000]
 [2.00000000e-002 0.00000000e+000 3.18000000e-003 ... 0.00000000e+000
  0.00000000e+000 0.00000000e+000]
 [2.50000000e-001 3.31250000e-002 0.00000000e+000 ... 2.13885975e-214
  1.17844112e-214 0.00000000e+000]
 [1.00000000e-002 4.69000000e-002 0.00000000e+000 ... 2.41642482e-213
  1.27375484e-213 9.57568349e-215]
 [0.00000000e+000 8.00000000e-004 0.00000000e+000 ... 1.96973759e-214
  9.65573676e-215 7.50528264e-215]]
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/unsupervised_learning/hmm#
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
