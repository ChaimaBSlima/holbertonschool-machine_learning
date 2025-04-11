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