<h1><p align="center"> Natural Language Processing - Evaluation Metrics </h1></p></font>

<p align="center">
  <img src="https://github.com/user-attachments/assets/7074fbb8-2c03-4d51-a8bc-b212405204ce" alt="Image"/>
</p>


# 📚 Resources

Read or watch:
- [7 Applications of Deep Learning for Natural Language Processing](https://machinelearningmastery.com/applications-of-deep-learning-for-natural-language-processing/)
- [10 Applications of Artificial Neural Networks in Natural Language Processing](https://medium.com/product-ai/artificial-neural-networks-in-natural-language-processing-bcf62aa9151a)
- [A Gentle Introduction to Calculating the BLEU Score for Text in Python](https://machinelearningmastery.com/calculate-bleu-score-for-text-python/)
- [Bleu Score](https://www.youtube.com/watch?v=DejHQYAGb7Q)
- [Evaluating Text Output in NLP: BLEU at your own risk](https://towardsdatascience.com/evaluating-text-output-in-nlp-bleu-at-your-own-risk-e8609665a213?gi=ce945ac1dd05)
- [ROUGE metric](https://towardsdatascience.com/the-ultimate-performance-metric-in-nlp-111df6c64460?gi=e9a998395fd9)
- [Evaluation and Perplexity](https://www.youtube.com/watch?v=BAN3NB_SNHY)
- [Evaluation metrics](https://aman.ai/primers/ai/evaluation-metrics/)

Definitions to skim
- [BLEU](https://en.wikipedia.org/wiki/BLEU)
- [ROUGE](https://en.wikipedia.org/wiki/ROUGE_(metric))
- [Perplexity](https://en.wikipedia.org/wiki/Perplexity)

References:
- [BLEU: a Method for Automatic Evaluation of Machine Translation (2002)](https://aclanthology.org/P02-1040.pdf)
- [ROUGE: A Package for Automatic Evaluation of Summaries (2004)](https://aclanthology.org/W04-1013.pdf)


---

# 🎯 Learning Objectives

At the end of this project, you are expected to be able to [explain to anyone](https://fs.blog/feynman-learning-technique/), without the help of Google:

### General

- What are the applications of natural language processing?  
- What is a BLEU score?  
- What is a ROUGE score?  
- What is perplexity?  
- When should you use one evaluation metric over another?  

---

# 🧾 Requirements

### General

- Allowed editors: `vi`, `vim`, `emacs`  
- All your files will be interpreted/compiled on **Ubuntu 20.04 LTS** using **python3 (version 3.9)**  
- Your files will be executed with **numpy (version 1.25.2)**  
- All your files should end with a **new line**  
- The first line of all your files should be exactly: `#!/usr/bin/env python3`  
- All of your files must be executable  
- A `README.md` file, at the root of the folder of the project, is mandatory  
- Your code should follow the `pycodestyle` style (version 2.11.1)  
- All your modules should have documentation (`python3 -c 'print(__import__("my_module").__doc__)'`)  
- All your classes should have documentation (`python3 -c 'print(__import__("my_module").MyClass.__doc__)'`)  
- All your functions (inside and outside a class) should have documentation (`python3 -c 'print(__import__("my_module").my_function.__doc__)'` and `python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)'`)  
- You are not allowed to use the `nltk` module  

---

# ❓ Quiz:

### Question #0  
The BLEU score measures:

- ~~A model’s accuracy~~  
- A model’s precision  ✔️ 
- ~~A model’s recall~~  
- ~~A model’s perplexity~~ 

### Question #1  
The ROUGE score measures:

- ~~A model’s accuracy~~  
- A model’s precision ✔️ 
- A model’s recall ✔️  
- ~~A model’s perplexity~~  

### Question #2  
Perplexity measures:

- ~~The accuracy of a prediction~~  
- The branching factor of a prediction ✔️  
- ~~A prediction’s recall~~  
- ~~A prediction’s accuracy~~  

### Question #3  
The BLEU score was designed for:

- ~~Sentiment Analysis~~  
- Machine Translation ✔️  
- ~~Question-Answering~~  
- ~~Document Summarization~~  

### Question #4  
What are the shortcomings of the BLEU score?

- It cannot judge grammatical accuracy ✔️  
- It cannot judge meaning ✔️  
- It does not work with languages that lack word boundaries ✔️  
- A higher score is not necessarily indicative of a better translation ✔️  

---

 # 📝 Tasks

 ### 0. Unigram BLEU score

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Write the function `def uni_bleu(references, sentence):` that calculates the unigram BLEU score for a sentence:

 - `references` is a list of reference translations
    - each reference translation is a list of the words in the translation
 - `sentence` is a list containing the model proposed sentence
 - **Returns:** the unigram BLEU score

```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/supervised_learning/nlp_metrics#./test_files/0-main.py
0.6549846024623855
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/supervised_learning/nlp_metrics#
```
<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 1. N-gram BLEU Score  

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Write the function `def ngram_bleu(references, sentence, n):` that calculates the n-gram BLEU score for a sentence:  

- `references` is a list of reference translations  
  - Each reference translation is a list of the words in the translation  
- `sentence` is a list containing the model proposed sentence  
- `n` is the size of the n-gram to use for evaluation  
- **Returns:** the n-gram BLEU score  

```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/supervised_learning/nlp_metrics#./test_files/1-main.py
0.6140480648084865
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/supervised_learning/nlp_metrics#
```

<p align="center">⭐⭐⭐⭐⭐⭐</p>


### 2. Cumulative N-gram BLEU Score

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Write the function `def cumulative_bleu(references, sentence, n):` that calculates the cumulative n-gram BLEU score for a sentence:

- `references` is a list of reference translations
  - Each reference translation is a list of the words in the translation
- `sentence` is a list containing the model proposed sentence
- `n` is the size of the largest n-gram to use for evaluation
- All n-gram scores should be weighted evenly
- **Returns:** the cumulative n-gram BLEU score

```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/supervised_learning/nlp_metrics#./test_files/2-main.py
0.5475182535069453
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/supervised_learning/nlp_metrics#
```

---

# 📄 Files

| Task Number | Task Title                   |File                 | Priority                                                             |
|-------------|------------------------------|---------------------|----------------------------------------------------------------------|
| 0           | 0. Unigram BLEU score        | `0-uni_bleu.py`   | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |
| 1           |1. N-gram BLEU score        | `1-ngram_bleu.py` | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |
| 2           |2. Cumulative N-gram BLEU score          | `2-cumulative_bleu.py`     | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |

---

# 📊 Project Summary

This project focuses on key evaluation metrics used in `Natural Language Processing (NLP)` to assess model performance. It covers metrics like `Accuracy`, `Precision`, `Recall`, `F1-Score` for classification; `BLEU` and `ROUGE` for text generation; and Perplexity for language modeling. Through examples and experiments, the project highlights how choosing the right metric is crucial for evaluating different `NLP` tasks effectively.

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
