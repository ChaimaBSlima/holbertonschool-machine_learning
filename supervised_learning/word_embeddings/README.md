<h1><p align="center"> Natural Language Processing - Word Embeddings </h1></p></font>

<p align="center">
  <img src="" alt="Image"/>
</p>

# 📚 Resources

Read or watch:
- [An Introduction to Word Embeddings](https://www.springboard.com/blog/data-science/introduction-word-embeddings/)
- [Introduction to Word Embeddings](https://hunterheidenreich.com/posts/intro-to-word-embeddings/)
- [Natural Language Processing|Bag Of Words Intuition](https://www.youtube.com/watch?v=IKgBLTeQQL8&list=PLZoTAELRMXVMdJ5sqbCK2LiM0HhQVWNzm&index=6)
- [Natural Language Processing|TF-IDF Intuition| Text Prerocessing](https://www.youtube.com/watch?v=D2V1okCEsiE&list=PLZoTAELRMXVMdJ5sqbCK2LiM0HhQVWNzm&index=8)
- [Word Embedding - Natural Language Processing| Deep Learning](https://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)
- [Word2Vec Tutorial - The Skip-Gram Model](https://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/)
- [Word2Vec Tutorial Part 2 - Negative Sampling](https://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/)
- [GloVe Explained](https://medium.com/sciforce/word-vectors-in-natural-language-processing-global-vectors-glove-51339db89639)
- [FastText: Under the Hood](https://towardsdatascience.com/fasttext-under-the-hood-11efc57b2b3?gi=d3587249a749)
- [ELMo Explained](https://www.mihaileric.com/posts/deep-contextualized-word-representations-elmo/)

Definitions to skim
- [Natural Language Processing](https://en.wikipedia.org/wiki/Natural_language_processing)

References:
- [Efficient Estimation of Word Representations in Vector Space (Skip-gram, 2013)](https://arxiv.org/pdf/1301.3781)
- [Distributed Representations of Words and Phrases and their Compositionality (Word2Vec, 2013)](https://arxiv.org/pdf/1310.4546)
- [GloVe: Global Vectors for Word Representation (website)](https://nlp.stanford.edu/projects/glove/)
- [GloVe: Global Vectors for Word Representation (2014)](https://nlp.stanford.edu/pubs/glove.pdf)
- [fastText (website)](https://fasttext.cc/)
- [Bag of Tricks for Efficient Text Classification (fastText, 2016)](https://arxiv.org/pdf/1607.01759)
- [Enriching Word Vectors with Subword Information (fastText, 2017)](https://arxiv.org/pdf/1607.04606)
- [Probabilistic FastText for Multi-Sense Word Embeddings (2018)](https://arxiv.org/pdf/1806.02901)
- [ELMo (website)](https://github.com/allenai/allennlp-models)
- [Deep contextualized word representations (ELMo, 2018)](https://arxiv.org/pdf/1802.05365)
- [sklearn.feature_extraction.text.CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer)
- [sklearn.feature_extraction.text.TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer)
- [genism.models.word2vec](https://radimrehurek.com/gensim/models/word2vec.html)
- [genism.models.fasttext](https://radimrehurek.com/gensim/models/fasttext.html)
- [Using Gensim Embeddings with Keras and Tensorflow](https://github.com/piskvorky/gensim/wiki/Using-Gensim-Embeddings-with-Keras-and-Tensorflow)

---

# 🎯 Learning Objectives

At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

### General

- What is natural language processing?  
- What is a word embedding?  
- What is bag of words?  
- What is TF-IDF?  
- What is CBOW?  
- What is a skip-gram?  
- What is an n-gram?  
- What is negative sampling?  
- What is word2vec, GloVe, fastText, ELMo?  

---

# 🧾 Requirements

### General

- Allowed editors: `vi`, `vim`, `emacs`  
- All your files will be interpreted/compiled on **Ubuntu 20.04 LTS** using **python3 (version 3.9)**  
- Your files will be executed with **numpy (version 1.25.2)** and **tensorflow (version 2.15)**  
- All your files should end with a **new line**  
- The first line of all your files should be exactly: `#!/usr/bin/env python3`  
- All of your files must be executable  
- A `README.md` file, at the root of the folder of the project, is mandatory  
- Your code should follow the `pycodestyle` style (version 2.11.1)  
- All your modules should have documentation (`python3 -c 'print(__import__("my_module").__doc__)'`)  
- All your classes should have documentation (`python3 -c 'print(__import__("my_module").MyClass.__doc__)'`)  
- All your functions (inside and outside a class) should have documentation (`python3 -c 'print(__import__("my_module").my_function.__doc__)'` and `python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)'`)  

# 💾 Download Gensim
```
pip install --user gensim==4.3.3
```
# 💾 Check that Keras version is 2.15.0
```
>>> import keras; keras.__version__
'2.15.0'
```
---
# ❓ Quiz:

### Question #0  
Word2Vec uses:

- ~~Character n-grams~~  
- Skip-grams ✔️  
- CBOW ✔️  
- ~~Co-occurrence matrices~~  
- Negative sampling ✔️  

### Question #1  
GloVe uses:

- ~~Character n-grams~~  
- ~~Skip-grams~~  
- ~~CBOW~~  
- Co-occurrence matrices ✔️  
- ~~Negative sampling~~  

### Question #2
FastText uses:

- Character n-grams ✔️  
- Skip-grams ✔️  
- CBOW ✔️  
- ~~Co-occurrence matrices~~  
- Negative sampling ✔️ 

### Question #3  
ELMo uses:

- Character n-grams ✔️
- ~~Skip-grams~~  
- ~~CBOW~~  
- ~~Co-occurrence matrices~~  
- ~~Negative sampling~~  

### Question #4  
Which of the following can be used in conjunction with the others?

- ~~Word2Vec~~  
- ~~GloVe~~  
- ~~FastText~~  
- ELMo ✔️  

---
 # 📝 Tasks

### 0. Bag Of Words

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Write a function `def bag_of_words(sentences, vocab=None):` that creates a bag of words embedding matrix:

- `sentences` is a list of sentences to analyze
- `vocab` is a list of the vocabulary words to use for the analysis
  - If None, all words within sentences should be used
- **Returns:** embeddings, features
  - `embeddings` is a numpy.ndarray of shape (s, f) containing the embeddings
    - `s` is the number of sentences in sentences
    - `f` is the number of features analyzed
  - `features` is a list of the features used for embeddings
- You are not allowed to use `genism` library.


```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/supervised_learning/word_embeddings#./test_files/0-main.py
[[0 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0]
 [0 1 0 0 0 0 0 0 0 1 1 0 1 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 1 0 0]
 [1 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0]
 [1 0 0 0 2 0 0 1 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0]
 [0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 1 1]
 [0 0 0 1 0 0 1 0 0 0 0 0 0 0 1 1 1 0 1 0 1 1 1 1]
 [0 0 1 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0]]
['are' 'awesome' 'beautiful' 'cake' 'children' 'future' 'good'
 'grandchildren' 'holberton' 'is' 'learning' 'life' 'machine' 'nlp' 'no'
 'not' 'one' 'our' 'said' 'school' 'that' 'the' 'very' 'was']
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/supervised_learning/word_embeddings#
```
<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 1. TF-IDF

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Write a function `def tf_idf(sentences, vocab=None):` that creates a TF-IDF embedding:

- `sentences` is a list of sentences to analyze
- `vocab` is a list of the vocabulary words to use for the analysis
  - If None, all words within `sentences` should be used
- **Returns:** embeddings, features
  - `embeddings` is a numpy.ndarray of shape (s, f) containing the embeddings
    - `s` is the number of sentences in `sentences`
    - `f` is the number of features analyzed
  - `features` is a list of the features used for embeddings

```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/supervised_learning/word_embeddings#./test_files/1-main.py
[[1.         0.         0.         0.         0.         0.
  0.        ]
 [0.5098139  0.60831315 0.         0.         0.         0.
  0.60831315]
 [0.         0.         0.         0.         0.         0.
  0.        ]
 [0.         0.         1.         0.         0.         0.
  0.        ]
 [0.         0.         1.         0.         0.         0.
  0.        ]
 [0.         0.         0.         0.70710678 0.70710678 0.
  0.        ]
 [0.         0.         0.         0.70710678 0.70710678 0.
  0.        ]
 [0.         0.         0.         0.         0.         0.
  0.        ]]
['awesome' 'learning' 'children' 'cake' 'good' 'none' 'machine']

root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/supervised_learning/word_embeddings#
```

<p align="center">⭐⭐⭐⭐⭐⭐</p>


### 2. Train Word2Vec

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Write a function `def word2vec_model(sentences, vector_size=100, min_count=5, window=5, negative=5, cbow=True, epochs=5, seed=0, workers=1):` that creates, builds and trains a gensim word2vec model:

- `sentences` is a list of sentences to be trained on
- `vector_size` is the dimensionality of the embedding layer
- `min_count` is the minimum number of occurrences of a word for use in training
- `window` is the maximum distance between the current and predicted word within a sentence
- `negative` is the size of negative sampling
- `cbow` is a boolean to determine the training type:
  - `True` for CBOW
  - `False` for Skip-gram
- `epochs` is the number of iterations to train over
- `seed` is the seed for the random number generator
- `workers` is the number of worker threads to train the model
- **Returns:** the trained model


```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/supervised_learning/word_embeddings#./test_files/2-main.py
[['human', 'interface', 'computer'], ['survey', 'user', 'computer', 'system', 'response', 'time']]
[-5.4084123e-03 -4.0024161e-04 -3.4630739e-03 -5.3525423e-03
  7.8537250e-03  6.0376106e-03 -7.2068786e-03  8.4706023e-03
  9.4194375e-03 -4.6773944e-03 -1.4714753e-03  7.7868701e-04
  3.1418847e-03 -1.1449445e-03 -7.0248209e-03  8.6203460e-03
  3.8405668e-03 -9.1897873e-03  6.2861182e-03  4.6401238e-03
 -6.3345446e-03  2.2874642e-03  3.3452510e-05 -9.4326939e-03
  8.5479887e-03  4.3843947e-03 -3.7956119e-03 -9.6801659e-03
 -8.1744418e-03  5.1590190e-03 -7.0132040e-03  2.5517345e-04
  7.9740928e-03  8.5820844e-03 -4.6414314e-03 -8.6783506e-03
 -1.0252714e-04  6.8263449e-03  2.4930835e-03 -8.6662006e-03
  3.0034208e-03 -3.1138016e-03 -5.4757069e-03 -1.3940263e-03
  7.4658301e-03  9.3212416e-03 -7.1789003e-03  1.2446367e-03
  5.2299835e-03 -4.8227082e-03 -4.5468416e-03 -5.1664864e-03
 -5.8076275e-03  7.7623655e-03 -5.6275711e-03 -5.4826117e-03
 -7.4911392e-03 -7.5089061e-03  5.5693723e-03 -4.2333854e-03
  6.0395217e-03  1.7224610e-03  7.1680485e-03  1.0818100e-03
  5.2833045e-03  6.1942148e-03 -8.7793246e-03  1.2095189e-03
 -9.0695143e-04 -4.2315759e-03 -9.5113518e-04 -1.7420733e-03
 -1.6348124e-04  6.3624191e-03  6.5098871e-03  2.5301289e-03
  4.2057564e-03  9.1815516e-03  2.7381873e-03 -2.6119126e-03
 -8.3582308e-03  1.0522294e-03 -5.3706346e-03  1.8784833e-03
 -9.4858548e-03  6.9658230e-03  8.8912249e-03 -7.0905304e-03
  6.3830256e-03 -1.8697941e-03 -9.1663310e-03  8.1991795e-03
  8.8182641e-03 -9.1386624e-03  1.8672824e-03  6.4541246e-03
  5.7970393e-03 -1.6923201e-03  7.1983398e-03  6.5960791e-03]
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/supervised_learning/word_embeddings#
```

<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 3. Extract Word2Vec

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Write a function `def gensim_to_keras(model):` that converts a gensim word2vec model to a keras Embedding layer:

- `model` is a trained gensim word2vec model
- **Returns:** the trainable keras Embedding layer
**Note:** The weights can/will be further updated in Keras

```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/supervised_learning/word_embeddings#./test_files/3-main.py
[['human', 'interface', 'computer'], ['survey', 'user', 'computer', 'system', 'response', 'time']]
<keras.src.layers.core.embedding.Embedding object at 0x7f08126b8910>
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/supervised_learning/word_embeddings#
```

<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 4. FastText

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Write a function `def fasttext_model(sentences, vector_size=100, min_count=5, negative=5, window=5, cbow=True, epochs=5, seed=0, workers=1):` that creates, builds and trains a gensim fastText model:

- `sentences` is a list of sentences to be trained on
- `vector_size` is the dimensionality of the embedding layer
- `min_count` is the minimum number of occurrences of a word for use in training
- `window` is the maximum distance between the current and predicted word within a sentence
- `negative` is the size of negative sampling
- `cbow` is a boolean to determine the training type:
  - `True` for CBOW
  - `False` for Skip-gram
- `epochs` is the number of iterations to train over
- `seed` is the seed for the random number generator
- `workers` is the number of worker threads to train the model
- **Returns:** the trained model
```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/supervised_learning/word_embeddings#./test_files/3-main.py
[['human', 'interface', 'computer'], ['survey', 'user', 'computer', 'system', 'response', 'time']]
[-4.4518875e-04  1.9057443e-04  7.1344204e-04  1.5088863e-04
  7.3785416e-04  2.0828047e-03 -1.4264339e-03 -6.6978252e-04
 -3.9446630e-04  6.1643129e-04  3.7035978e-04 -1.7527672e-03
  2.0829479e-05  1.0929988e-03 -6.6954875e-04  7.9767447e-04
 -9.0742309e-04  1.9187949e-03 -6.9725298e-04  3.7622583e-04
 -5.0849823e-05  1.6160590e-04 -8.3575735e-04 -1.4309353e-03
  1.8365250e-04 -1.1365860e-03 -2.1796341e-03  3.3816829e-04
 -1.0266158e-03  1.9360909e-03  9.3765622e-05 -1.2577525e-03
  1.7052694e-04 -1.0470246e-03  9.1582153e-04 -1.1945128e-03
  1.2874184e-03 -3.1551000e-04 -1.1084992e-03  2.2345960e-04
  5.9021922e-04 -5.7232735e-04  1.6017178e-04 -1.0333696e-03
 -2.6842864e-04 -1.2489735e-03 -3.4248878e-05  2.0717620e-03
  1.0997808e-03  4.9419136e-04 -4.3252495e-04  7.6816598e-04
  3.0231036e-04  6.4548600e-04  2.5580439e-03 -1.2883682e-04
 -3.8391326e-04 -2.1800243e-04  6.5950496e-04 -2.8844117e-04
 -7.4177544e-04 -6.5318396e-04  1.4357771e-03  1.7945657e-03
  3.2790678e-03 -1.1300950e-03 -1.5527758e-04  4.3252096e-04
  2.0878548e-03  5.8326498e-04 -4.1506172e-04  1.1454885e-03
 -6.3745341e-05 -2.0422263e-03 -8.0344628e-04  2.0709851e-04
 -8.6796697e-04  7.6198514e-04 -3.0726698e-04  2.1699023e-04
 -1.4049197e-03 -1.9049532e-03 -1.1490833e-03 -3.2594264e-04
 -7.8721769e-04 -2.5946668e-03 -6.0526514e-04  9.3661918e-04
  5.8702513e-04  3.1111998e-04 -5.1438244e-04  4.9440534e-04
 -1.7251119e-03  5.4227427e-04 -7.4013631e-04 -4.8912101e-04
 -1.3722111e-03  2.1129930e-03  1.4438890e-03 -1.0972627e-03]
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/supervised_learning/word_embeddings#
```
<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 5. ELMo

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

When training an ELMo embedding model, you are training:

1. The internal weights of the BiLSTM
2. The character embedding layer
3. The weights applied to the hidden states

In the text file `5-elmo`, write the letter answer, followed by a newline, that lists the correct statements:

A. 1, 2, 3  
B. 1, 2  
C. 2, 3  
D. 1, 3  
E. 1  
F. 2  
G. 3  
H. None of the above

---
# 📄 Files

| Task Number | Task Title                   |File                 | Priority                                                             |
|-------------|------------------------------|---------------------|----------------------------------------------------------------------|
| 0           | 0. Bag Of Words        | `0-bag_of_words.py`   | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |
| 1           | 1. TF-IDF              | `1-tf_idf.py` | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |
| 2           | 2. Train Word2Vec           | `2-word2vec.py`     | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |
| 3           | 3. Extract Word2Vec           | `3-gensim_to_keras.py`    | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |
| 4           | 4. FastText            | `4-fasttext.py`   | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)  |
| 5           |5. ELMo       | `5-elmo` | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)    |

---

# 📊 Project Summary

This project explores word embeddings in Natural Language Processing (NLP), which are techniques to represent words as dense vectors that capture their meaning and context. It covers popular methods like **Word2Vec**, **GloVe**, and **FastText**, and **demonstrates** how these embeddings improve **NLP** tasks such as text classification and sentiment analysis by preserving semantic relationships between words.

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
