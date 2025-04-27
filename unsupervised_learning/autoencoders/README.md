<h1><p align="center"> Autoencoders </h1></p></font>


<p align="center">
  <img src="https://github.com/user-attachments/assets/51f5d4d3-c860-48b4-a975-760ecff94353" alt="Image"/>
</p>


# 📚 Resources

Read or watch:
- [Autoencoder - definition](https://www.youtube.com/watch?v=FzS3tMl4Nsc&t=73s)
- [Autoencoder - loss function](https://www.youtube.com/watch?v=xTU79Zs4XKY)
- [Deep learning - deep autoencoder](https://www.youtube.com/watch?v=z5ZYm_wJ37c)
- [Introduction to autoencoders](https://www.jeremyjordan.me/autoencoders/)
- [Variational Autoencoders - EXPLAINED!](https://www.youtube.com/watch?v=fcvYpzHmhvA) up to 12:55
- [Variational Autoencoders](https://www.youtube.com/watch?v=9zKuYvjFFS8)
- [Intuitively Understanding Variational Autoencoders](https://medium.com/data-science/intuitively-understanding-variational-autoencoders-1bfe67eb5daf)
- [Deep Generative Models](https://www.actian.com/glossary/deep-generative-models/)

Definitions to skim:
- [Kullback–Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) recall its use in t-SNE
- [Autoencoder](https://en.wikipedia.org/wiki/Autoencoder)
- [Generative model](https://en.wikipedia.org/wiki/Generative_model)

References:
- [The Deep Learning textbook - Chapter 14: Autoencoders](https://www.deeplearningbook.org/contents/autoencoders.html)
- [Reducing the Dimensionality of Data with Neural Networks 2006](https://www.cs.toronto.edu/~hinton/absps/science.pdf)


---

# 🎯 Learning Objectives

- What is an autoencoder?  
- What is latent space?  
- What is a bottleneck?  
- What is a sparse autoencoder?  
- What is a convolutional autoencoder?  
- What is a generative model?  
- What is a variational autoencoder?  
- What is the Kullback-Leibler divergence?  

---

# 🧾 Requirements

### General

- Allowed editors: `vi`, `vim`, `emacs`  
- All your files will be interpreted/compiled on **Ubuntu 20.04 LTS** using **python3 (version 3.9)**  
- Your files will be executed with **numpy (version 1.25.2)** and **tensorflow (version 2.15)**  
- All your files should end with a **new line**  
- The first line of all your files should be exactly: `#!/usr/bin/env python3`  
- A `README.md` file, at the root of the folder of the project, is mandatory  
- Your code should use the `pycodestyle` style (version 2.11.1)  
- All your modules should have documentation (`python3 -c 'print(__import__("my_module").__doc__)'`)  
- All your classes should have documentation (`python3 -c 'print(__import__("my_module").MyClass.__doc__)'`)  
- All your functions (inside and outside a class) should have documentation (`python3 -c 'print(__import__("my_module").my_function.__doc__)'` and `python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)'`)  
- Unless otherwise noted, you are not allowed to import any module except `import tensorflow.keras as keras`  
- All your files must be executable  

---

# ❓ Quiz:

### Question #0  
What is a “vanilla” autoencoder?

- A compression model ✔️ 
- Composed of an encoder and decoder ✔️  
- ~~A generative model~~  
- Learns a latent space representation ✔️  

### Question #1  
What is a bottleneck?

- ~~When you can no longer train your model~~  
- ~~The latent space representation~~   
- ~~The compressed input~~     
- A layer that is smaller than the previous and next layers ✔️  

### Question #2  
What is a VAE?

- ~~An adversarial network~~  
- A generative model ✔️  
- Composed of an encoder and decoder ✔️  
- ~~A compression model~~  

### Question #3  
What loss function(s) is/are used for training vanilla autoencoders?

- Mean Squared Error ✔️  
- ~~L2 Normalization~~  
- Cross Entropy  ✔️
- ~~Kullback-Leibler Divergence~~  

### Question #4  
What loss function(s) is/are used for training variational autoencoders?

- Mean Squared Error ✔️  
- ~~L2 Normalization~~  
- Cross Entropy ✔️  
- Kullback-Leibler Divergence ✔️  


---

# 📝 Tasks

### 0. "Vanilla" Autoencoder

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Write a function `def autoencoder(input_dims, hidden_layers, latent_dims):` that creates an autoencoder:

- `input_dims` is an integer containing the dimensions of the model input  
- `hidden_layers` is a list containing the number of nodes for each hidden layer in the encoder, respectively  
  - the hidden layers should be reversed for the decoder  
- `latent_dims` is an integer containing the dimensions of the latent space representation  

#### Returns:
`encoder`, `decoder`, `auto`  

- `encoder` is the encoder model  
- `decoder` is the decoder model  
- `auto` is the full autoencoder model  

The autoencoder model should be compiled using `adam` optimization and `binary cross-entropy` loss  
All layers should use a `relu` activation except for the last layer in the decoder, which should use `sigmoid`
```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/unsupervised_learning/autoencoders#./test_files/0-main.py
Epoch 1/50
235/235 [==============================] - 3s 10ms/step - loss: 0.2462 - val_loss: 0.1704
Epoch 2/50
235/235 [==============================] - 2s 10ms/step - loss: 0.1526 - val_loss: 0.1370
Epoch 3/50
235/235 [==============================] - 3s 11ms/step - loss: 0.1319 - val_loss: 0.1242
Epoch 4/50
235/235 [==============================] - 2s 10ms/step - loss: 0.1216 - val_loss: 0.1165
Epoch 5/50
235/235 [==============================] - 3s 11ms/step - loss: 0.1157 - val_loss: 0.1119

...

Epoch 46/50
235/235 [==============================] - 2s 11ms/step - loss: 0.0851 - val_loss: 0.0845
Epoch 47/50
235/235 [==============================] - 2s 11ms/step - loss: 0.0849 - val_loss: 0.0845
Epoch 48/50
235/235 [==============================] - 3s 12ms/step - loss: 0.0848 - val_loss: 0.0842
Epoch 49/50
235/235 [==============================] - 3s 13ms/step - loss: 0.0847 - val_loss: 0.0842
Epoch 50/50
235/235 [==============================] - 3s 12ms/step - loss: 0.0846 - val_loss: 0.0842
1/1 [==============================] - 0s 76ms/step
8.311438
1/1 [==============================] - 0s 80ms/step
```

<p align="center">
  <img src="https://github.com/user-attachments/assets/a92385b8-6666-4efe-97cd-2192c59ea03f" alt="Image"/>
</p>

<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 1. Sparse Autoencoder

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Write the function `def correlation(C):` that calculates a correlation matrix:
Write a function `def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):` that creates a sparse autoencoder:

- `input_dims` is an integer containing the dimensions of the model input  
- `hidden_layers` is a list containing the number of nodes for each hidden layer in the encoder, respectively  
  - the hidden layers should be reversed for the decoder  
- `latent_dims` is an integer containing the dimensions of the latent space representation  
- `lambtha` is the regularization parameter used for L1 regularization on the encoded output  

#### Returns:
`encoder`, `decoder`, `auto`  

- `encoder` is the encoder model  
- `decoder` is the decoder model  
- `auto` is the sparse autoencoder model  

The sparse autoencoder model should be compiled using `adam` optimization and `binary cross-entropy` loss  
All layers should use a `relu` activation except for the last layer in the decoder, which should use `sigmoid`

```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/unsupervised_learning/autoencoders#./test_files/1-main.py
Epoch 1/50
235/235 [==============================] - 4s 15ms/step - loss: 0.2467 - val_loss: 0.1715
Epoch 2/50
235/235 [==============================] - 3s 14ms/step - loss: 0.1539 - val_loss: 0.1372
Epoch 3/50
235/235 [==============================] - 2s 9ms/step - loss: 0.1316 - val_loss: 0.1242
Epoch 4/50
235/235 [==============================] - 2s 9ms/step - loss: 0.1218 - val_loss: 0.1166
Epoch 5/50
235/235 [==============================] - 2s 9ms/step - loss: 0.1157 - val_loss: 0.1122

...

Epoch 46/50
235/235 [==============================] - 3s 11ms/step - loss: 0.0844 - val_loss: 0.0844
Epoch 47/50
235/235 [==============================] - 3s 11ms/step - loss: 0.0843 - val_loss: 0.0840
Epoch 48/50
235/235 [==============================] - 3s 11ms/step - loss: 0.0842 - val_loss: 0.0837
Epoch 49/50
235/235 [==============================] - 3s 11ms/step - loss: 0.0841 - val_loss: 0.0837
Epoch 50/50
235/235 [==============================] - 3s 12ms/step - loss: 0.0839 - val_loss: 0.0835
1/1 [==============================] - 0s 85ms/step
3.0174155
1/1 [==============================] - 0s 46ms/step
```

<p align="center">
  <img src="https://github.com/user-attachments/assets/3748d787-fa37-4aa3-98bf-49d0a39561e6" alt="Image"/>
</p>

<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 2. Convolutional Autoencoder

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Write a function `def autoencoder(input_dims, filters, latent_dims):` that creates a convolutional autoencoder:

- `input_dims` is a tuple of integers containing the dimensions of the model input  
- `filters` is a list containing the number of filters for each convolutional layer in the encoder, respectively  
  - the filters should be reversed for the decoder  
- `latent_dims` is a tuple of integers containing the dimensions of the latent space representation  

Each convolution in the encoder should:  
- use a kernel size of `(3, 3)`  
- use `same` padding  
- use `relu` activation  
- be followed by max pooling of size `(2, 2)`

Each convolution in the decoder, except for the last two, should:  
- use a filter size of `(3, 3)`  
- use `same` padding  
- use `relu` activation  
- be followed by upsampling of size `(2, 2)`

The second to last convolution should instead use `valid` padding.  
The last convolution should:  
- have the same number of filters as the number of channels in `input_dims`  
- use `sigmoid` activation  
- no upsampling  

#### Returns:
`encoder`, `decoder`, `auto`  

- `encoder` is the encoder model  
- `decoder` is the decoder model  
- `auto` is the full autoencoder model  

The autoencoder model should be compiled using `adam` optimization and `binary cross-entropy` loss

```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/unsupervised_learning/autoencoders#./test_files/2-main.py
(60000, 28, 28, 1)
(10000, 28, 28, 1)
Epoch 1/50
235/235 [==============================] - 117s 457ms/step - loss: 0.2466 - val_loss: 0.1597
Epoch 2/50
235/235 [==============================] - 107s 457ms/step - loss: 0.1470 - val_loss: 0.1358
Epoch 3/50
235/235 [==============================] - 114s 485ms/step - loss: 0.1320 - val_loss: 0.1271
Epoch 4/50
235/235 [==============================] - 104s 442ms/step - loss: 0.1252 - val_loss: 0.1216
Epoch 5/50
235/235 [==============================] - 99s 421ms/step - loss: 0.1208 - val_loss: 0.1179

...

Epoch 46/50
235/235 [==============================] - 72s 307ms/step - loss: 0.0943 - val_loss: 0.0933
Epoch 47/50
235/235 [==============================] - 80s 339ms/step - loss: 0.0942 - val_loss: 0.0929
Epoch 48/50
235/235 [==============================] - 65s 279ms/step - loss: 0.0940 - val_loss: 0.0932
Epoch 49/50
235/235 [==============================] - 53s 225ms/step - loss: 0.0939 - val_loss: 0.0927
Epoch 50/50
235/235 [==============================] - 39s 165ms/step - loss: 0.0938 - val_loss: 0.0926
1/1 [==============================] - 0s 235ms/step
3.4141076
1/1 [==============================] - 0s 85ms/step
```

<p align="center">
  <img src="https://github.com/user-attachments/assets/4745f30e-37fb-432f-8193-fde7e100ea86" alt="Image"/>
</p>

<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 3. Variational Autoencoder

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Update the class `MultiNormal`:

Add the public instance method `def pdf(self, x):` that calculates the PDF at a data point:

- `x` is a numpy.ndarray of shape `(d, 1)` containing the data point whose PDF should be calculated  
  - `d` is the number of dimensions of the `Multinomial` instance  
- If `x` is not a numpy.ndarray, raise a `TypeError` with the message `x must be a numpy.ndarray`  
- If `x` is not of shape `(d, 1)`, raise a `ValueError` with the message `x must have the shape ({d}, 1)`  

#### Returns:
- The value of the PDF  

You are not allowed to use the function `numpy.cov`Write a function `def autoencoder(input_dims, hidden_layers, latent_dims):` that creates a variational autoencoder:

- `input_dims` is an integer containing the dimensions of the model input  
- `hidden_layers` is a list containing the number of nodes for each hidden layer in the encoder, respectively  
  - the hidden layers should be reversed for the decoder  
- `latent_dims` is an integer containing the dimensions of the latent space representation  

#### Returns:
`encoder`, `decoder`, `auto`  

- `encoder` is the encoder model, which should output the latent representation, the mean, and the log variance, respectively  
- `decoder` is the decoder model  
- `auto` is the full autoencoder model  

The autoencoder model should be compiled using `adam` optimization and `binary cross-entropy` loss  
All layers should use a `relu` activation except:  
- the mean and log variance layers in the encoder, which should use `None` activation  
- the last layer in the decoder, which should use `sigmoid`

```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/unsupervised_learning/autoencoders#./test_files/3-main.py
Epoch 1/50
235/235 [==============================] - 5s 17ms/step - loss: 212.1680 - val_loss: 175.3891
Epoch 2/50
235/235 [==============================] - 4s 17ms/step - loss: 170.0067 - val_loss: 164.9127
Epoch 3/50
235/235 [==============================] - 4s 18ms/step - loss: 163.6800 - val_loss: 161.2009
Epoch 4/50
235/235 [==============================] - 5s 21ms/step - loss: 160.5563 - val_loss: 159.1755
Epoch 5/50
235/235 [==============================] - 5s 22ms/step - loss: 158.5609 - val_loss: 157.5874

...

Epoch 46/50
235/235 [==============================] - 4s 19ms/step - loss: 143.8559 - val_loss: 148.1236
Epoch 47/50
235/235 [==============================] - 4s 19ms/step - loss: 143.7759 - val_loss: 148.0166
Epoch 48/50
235/235 [==============================] - 4s 19ms/step - loss: 143.6073 - val_loss: 147.9645
Epoch 49/50
235/235 [==============================] - 5s 19ms/step - loss: 143.5385 - val_loss: 148.1294
Epoch 50/50
235/235 [==============================] - 5s 20ms/step - loss: 143.3937 - val_loss: 147.9027
1/1 [==============================] - 0s 124ms/step
[[-4.4424314e-04  3.7557125e-05]
 [-2.3759568e-04  3.6484184e-04]
 [ 3.6569734e-05 -7.3342602e-04]
 [-5.5730779e-04 -6.3699216e-04]
 [-5.8648770e-04  8.7332644e-04]
 [ 1.7586297e-04 -8.7016745e-04]
 [-5.4950645e-04  6.9131691e-04]
 [-5.1684811e-04  3.8412266e-04]
 [-2.7567835e-04  5.2892545e-04]
 [-5.0945382e-04  1.0410405e-03]]
[[0.9501978  3.0150387 ]
 [1.1207044  0.6665632 ]
 [0.19164634 1.5250858 ]
 [0.9454097  0.45243642]
 [1.5451298  1.2251403 ]
 [0.28436017 1.3658737 ]
 [0.97746277 1.234872  ]
 [1.7042938  1.5537287 ]
 [1.2055128  1.1579443 ]
 [0.9644342  1.6614302 ]]
1/1 [==============================] - 0s 46ms/step
5/5 [==============================] - 0s 3ms/step
```

<p align="center">
  <img src="https://github.com/user-attachments/assets/8c961c0f-bdd8-49bd-ba6e-49b074159eea" alt="Image"/>
</p>

---
# 📄 Files

| Task Number | Task Title                   |File                 | Priority                                                             |
|-------------|------------------------------|---------------------|----------------------------------------------------------------------|
| 0           | 0. "Vanilla" Autoencoder        | `0-vanilla.py`   | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |
| 1           | 1. Sparse Autoencoder              | `1-sparse.py` | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |
| 2           | 2. Convolutional Autoencoder     | `2-convolutional.py`     | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |
| 3           | 3. Variational Autoencoder               | `3-variational.py`    | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |
 
---

# 📊 Project Summary

The main goal of this project is to design, train, and evaluate Autoencoders to perform tasks like data compression, image reconstruction, and anomaly detection. The project focuses on understanding how Autoencoders learn efficient data representations in an unsupervised manner.

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
