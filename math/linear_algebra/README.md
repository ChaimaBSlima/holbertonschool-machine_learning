<h1><p align="center"> Linear Algebra </h1></p></font>

*For this project, we expect you to look at this concept:*

*- [Using Vagrant on your personal computer](https://github.com/ChaimaBSlima/Valuable-IT-Concepts/blob/main/Vagrant.md)*

---

<p align="center">
  <img src="https://github.com/user-attachments/assets/b05329ca-44ff-4ba5-9988-2f8cae8cd864" alt="Image"/>
</p>


# 📚 Resources

Read or watch:

- [Introduction to vectors](https://www.youtube.com/watch?v=fNk_zzaMoSs)
- [What is a matrix?](https://math.stackexchange.com/questions/2782717/what-exactly-is-a-matrix) (*not [the matrix](https://www.imdb.com/title/tt0133093/)*)
- [Transpose](https://en.wikipedia.org/wiki/Transpose)
- [Understanding the dot product](https://betterexplained.com/articles/vector-calculus-understanding-the-dot-product/)
- [Matrix Multiplication](https://www.youtube.com/watch?v=BzWahqwaS8k)
- [What is the relationship between matrix multiplication and the dot product?](https://www.quora.com/What-is-the-relationship-between-matrix-multiplication-and-the-dot-product)
- [The Dot Product, Matrix Multiplication, and the Magic of Orthogonal Matrices](https://www.youtube.com/watch?v=rW2ypKLLxGk)
- [numpy tutorial](https://numpy.org/doc/stable/user/quickstart.html) *(until Shape Manipulation (excluded))*
- [numpy basics](https://www.oreilly.com/library/view/python-for-data/9781449323592/ch04.html) *(until Universal Functions (included))*
- [array indexing](https://docs.scipy.org/doc/numpy-1.15.0/reference/arrays.indexing.html#basic-slicing-and-indexing)
- [numerical operations on arrays](https://scipy-lectures.org/intro/numpy/operations.html)
- [Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [numpy mutations and broadcasting](https://medium.com/data-science/two-cool-features-of-python-numpy-mutating-by-slicing-and-broadcasting-3b0b86e8b4c7)
- [numpy.ndarray](https://docs.scipy.org/doc/numpy-1.15.0/reference/arrays.ndarray.html)
- [numpy.ndarray.shape](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.ndarray.shape.html#numpy.ndarray.shape)
- [numpy.transpose](https://numpy.org/doc/stable/reference/generated/numpy.transpose.html)
- [numpy.ndarray.transpose](https://intranet.hbtn.io/rltoken/VLP2xnn3VEob-A9upmkAaghttps://numpy.org/doc/stable/reference/generated/numpy.ndarray.transpose.html)
- [numpy.matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html)

---

# 🎯 Learning Objectives

At the end of this project, you are expected to be able to [explain to anyone](https://intranet.hbtn.io/rltoken/vdZL8qUVjpXNsz71U9mQWA), without the help of Google:

### General

- What is a vector?  
- What is a matrix?  
- What is a transpose?  
- What is the shape of a matrix?  
- What is an axis?  
- What is a slice?  
- How do you slice a vector/matrix?  
- What are element-wise operations?  
- How do you concatenate vectors/matrices?  
- What is the dot product?  
- What is matrix multiplication?  
- What is Numpy?  
- What is parallelization and why is it important?  
- What is broadcasting?  

---

# 🧾 Requirements

### Python Scripts

- Allowed editors: `vi`, `vim`, `emacs`  
- All your files will be interpreted/compiled on **Ubuntu 20.04 LTS** using **python3 (version 3.9)**  
- Your files will be executed with **numpy (version 1.25.2)**  
- All your files should end with a **new line**  
- The first line of all your files should be exactly: `#!/usr/bin/env python3`  
- A `README.md` file, at the root of the folder of the project, is mandatory  
- Your code should follow `pycodestyle` (version 2.11.1)  
- All your modules should have documentation (`python3 -c 'print(__import__("my_module").__doc__)'`)  
- All your classes should have documentation (`python3 -c 'print(__import__("my_module").MyClass.__doc__)'`)  
- All your functions (inside and outside a class) should have documentation (`python3 -c 'print(__import__("my_module").my_function.__doc__)'` and `python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)'`)  
- Unless otherwise noted, you are not allowed to import any module  
- All your files must be executable  
- The length of your files will be tested using `wc`  

---


# 📝 Tasks

### 0. Slice Me Up

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Complete the following source code (found below):

- `arr1` should be the first two numbers of `arr`
- `arr2` should be the last five numbers of `arr`
- `arr3` should be the 2nd through 6th numbers of `arr`
- You are not allowed to use any loops or conditional statements
- Your program should be exactly 8 lines

```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/linear_algebra#.cat 0-slice_me_up.py
#!/usr/bin/env python3
arr = [9, 8, 2, 3, 9, 4, 1, 0, 3]
arr1 =  # your code here
arr2 =  # your code here
arr3 =  # your code here
print("The first two numbers of the array are: {}".format(arr1))
print("The last five numbers of the array are: {}".format(arr2))
print("The 2nd through 6th numbers of the array are: {}".format(arr3))
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/linear_algebra#./0-slice_me_up.py 
The first two numbers of the array are: [9, 8]
The last five numbers of the array are: [9, 4, 1, 0, 3]
The 2nd through 6th numbers of the array are: [8, 2, 3, 9, 4]
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/linear_algebra# wc -l 0-slice_me_up.py
8 0-slice_me_up.py
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/linear_algebra#
```
<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 1. Trim Me Down

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Complete the following source code (found below):

- `the_middle` should be a 2D matrix containing the 3rd and 4th columns of `matrix`
- You are not allowed to use any conditional statements
- You are only allowed to use one `for` loop
- Your program should be exactly 6 lines

```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/linear_algebra# cat 1-trim_me_down.py 
#!/usr/bin/env python3
matrix = [[1, 3, 9, 4, 5, 8], [2, 4, 7, 3, 4, 0], [0, 3, 4, 6, 1, 5]]
the_middle = []
# your code here
print("The middle columns of the matrix are: {}".format(the_middle))
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/linear_algebra# ./1-trim_me_down.py
The middle columns of the matrix are: [[9, 4], [7, 3], [4, 6]]
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/linear_algebra# wc -l 1-trim_me_down.py
6 1-trim_me_down.py
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/linear_algebra#
```

<p align="center">⭐⭐⭐⭐⭐⭐</p>


### 2. Size Me Please 

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Write a function `def matrix_shape(matrix):` that calculates the shape of a matrix:

- You can assume all elements in the same dimension are of the same type/shape
- The shape should be returned as a list of integers
```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/linear_algebra# ./test_files/2-main.py
[2, 2]
[2, 3, 5]
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/linear_algebra# 
```

<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 3. Flip Me Over 

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)


Write a function `def matrix_transpose(matrix):` that returns the transpose of a 2D matrix, `matrix`:

- You must return a new matrix
- You can assume that `matrix` is never empty
- You can assume all elements in the same dimension are of the same type/shape

```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/linear_algebra# ./test_files/3-main.py
[[1, 2], [3, 4]]
[[1, 3], [2, 4]]
[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]
[[1, 6, 11, 16, 21, 26], [2, 7, 12, 17, 22, 27], [3, 8, 13, 18, 23, 28], [4, 9, 14, 19, 24, 29], [5, 10, 15, 20, 25, 30]]
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/linear_algebra# 
```

<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 4. Line Up
![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)


Write a function `def add_arrays(arr1, arr2):` that adds two arrays element-wise:

- You can assume that `arr1` and `arr2` are lists of ints/floats
- You must return a new list
- If `arr1` and `arr2` are not the same shape, return `None`
```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/linear_algebra# ./test_files/4-main.py
[6, 8, 10, 12]
[1, 2, 3, 4]
[5, 6, 7, 8]
None
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/linear_algebra# 
```

<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 5. Across The Planes

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Write a function `def add_matrices2D(mat1, mat2):` that adds two matrices element-wise:

- You can assume that `mat1` and `mat2` are 2D matrices containing ints/floats
- You can assume all elements in the same dimension are of the same type/shape
- You must return a new matrix
- If `mat1` and `mat2` are not the same shape, return `None`

```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/linear_algebra# ./test_files/5-main.py
[[6, 8], [10, 12]]
[[1, 2], [3, 4]]
[[5, 6], [7, 8]]
None
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/linear_algebra# 
```

<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 6. Howdy Partner
![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)


Write a function `def cat_arrays(arr1, arr2):` that concatenates two arrays:

- You can assume that `arr1` and `arr2` are lists of ints/floats
- You must return a new list
```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/linear_algebra# ./test_files/6-main.py 
[1, 2, 3, 4, 5, 6, 7, 8]
[1, 2, 3, 4, 5]
[6, 7, 8]
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/linear_algebra# 
```
<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 7. Gettin’ Cozy 

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)


Write a function `def cat_matrices2D(mat1, mat2, axis=0):` that concatenates two matrices along a specific axis:

- You can assume that `mat1` and `mat2` are 2D matrices containing ints/floats
- You can assume all elements in the same dimension are of the same type/shape
- You must return a new matrix
- If the two matrices cannot be concatenated, return `None`

```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/linear_algebra# ./test_files/7-main.py
[[1, 2], [3, 4], [5, 6]]
[[1, 2, 7], [3, 4, 8]]
[[9, 10], [3, 4, 5]]
[[1, 2], [3, 4], [5, 6]]
[[1, 2, 7], [3, 4, 8]]
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/linear_algebra# 
```

<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 8. Ridin’ Bareback
![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Write a function `def mat_mul(mat1, mat2):` that performs matrix multiplication:

- You can assume that `mat1` and `mat2` are 2D matrices containing ints/floats
- You can assume all elements in the same dimension are of the same type/shape
- You must return a new matrix
- If the two matrices cannot be multiplied, return `None`
```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/linear_algebra# ./test_files/8-main.py
[[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]]
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/linear_algebra# 
```

<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 9. Let The Butcher Slice It

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Complete the following source code (found below):

- `mat1` should be the middle two rows of `matrix`
- `mat2` should be the middle two columns of `matrix`
- `mat3` should be the bottom-right, square, 3x3 matrix of `matrix`
- You are not allowed to use any loops or conditional statements
- Your program should be exactly 10 lines

```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/linear_algebra# cat 9-let_the_butcher_slice_it.py 
#!/usr/bin/env python3
import numpy as np
matrix = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12],
                   [13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24]])
mat1 =  # your code here
mat2 =  # your code here
mat3 =  # your code here
print("The middle two rows of the matrix are:\n{}".format(mat1))
print("The middle two columns of the matrix are:\n{}".format(mat2))
print("The bottom-right, square, 3x3 matrix is:\n{}".format(mat3))
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/linear_algebra# ./9-let_the_butcher_slice_it.py 
The middle two rows of the matrix are:
[[ 7  8  9 10 11 12]
 [13 14 15 16 17 18]]
The middle two columns of the matrix are:
[[ 3  4]
 [ 9 10]
 [15 16]
 [21 22]]
The bottom-right, square, 3x3 matrix is:
[[10 11 12]
 [16 17 18]
 [22 23 24]]
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/linear_algebra#  wc -l 9-let_the_butcher_slice_it.py 
10 9-let_the_butcher_slice_it.py
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/linear_algebra#
```

<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 10. I'll Use My Scale

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Write a function `def np_shape(matrix):` that calculates the shape of a numpy.ndarray:

- You are not allowed to use any loops or conditional statements
- You are not allowed to use try/except statements
- The shape should be returned as a tuple of integers

```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/linear_algebra# ./test_files/10-main.py
(6,)
(0,)
(2, 2, 5)
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/linear_algebra# 
```

<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 11. The Western Exchange
![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)


Write a function `def np_transpose(matrix):` that transposes matrix:

- You can assume that matrix can be interpreted as a numpy.ndarray
- You are not allowed to use any loops or conditional statements
- You must return a new numpy.ndarray
```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/linear_algebra# ./test_files/11-main.py 
[1 2 3 4 5 6]
[1 2 3 4 5 6]
[]
[]
[[[ 1 11]
  [ 6 16]]

 [[ 2 12]
  [ 7 17]]

 [[ 3 13]
  [ 8 18]]

 [[ 4 14]
  [ 9 19]]

 [[ 5 15]
  [10 20]]]
[[[ 1  2  3  4  5]
  [ 6  7  8  9 10]]

 [[11 12 13 14 15]
  [16 17 18 19 20]]]
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/linear_algebra# 
```

<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 12. Bracing The Elements

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)


Write a function `def np_elementwise(mat1, mat2):` that performs element-wise addition, subtraction, multiplication, and division:

- You can assume that `mat1` and `mat2` can be interpreted as numpy.ndarrays
- You should return a tuple containing the element-wise sum, difference, product, and quotient, respectively
- You are not allowed to use any loops or conditional statements
- You can assume that `mat1` and `mat2` are never empty

```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/linear_algebra# ./test_files/12-main.py
[[11 22 33]
 [44 55 66]]
[[1 2 3]
 [4 5 6]]
Add:
 [[12 24 36]
 [48 60 72]] 
Sub:
 [[10 20 30]
 [40 50 60]] 
Mul:
 [[ 11  44  99]
 [176 275 396]] 
Div:
 [[11. 11. 11.]
 [11. 11. 11.]]
Add:
 [[13 24 35]
 [46 57 68]] 
Sub:
 [[ 9 20 31]
 [42 53 64]] 
Mul:
 [[ 22  44  66]
 [ 88 110 132]] 
Div:
 [[ 5.5 11.  16.5]
 [22.  27.5 33. ]]
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/linear_algebra# 
```

<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 13. Cat's Got Your Tongue
![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Write a function `def np_cat(mat1, mat2, axis=0):` that concatenates two matrices along a specific axis:

- You can assume that `mat1` and `mat2` can be interpreted as numpy.ndarrays
- You must return a new numpy.ndarray
- You are not allowed to use any loops or conditional statements
- You may use: `import numpy as np`
- You can assume that `mat1` and `mat2` are never empty
```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/linear_algebra# ./test_files/13-main.py
[[11 22 33]
 [44 55 66]
 [ 1  2  3]
 [ 4  5  6]]
[[11 22 33  1  2  3]
 [44 55 66  4  5  6]]
[[11 22 33  7]
 [44 55 66  8]]
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/linear_algebra# 
```

<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 14. Saddle Up

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

Write a function `def np_matmul(mat1, mat2):` that performs matrix multiplication:

- You can assume that `mat1` and `mat2` are numpy.ndarrays
- You are not allowed to use any loops or conditional statements
- You may use: `import numpy as np`
- You can assume that `mat1` and `mat2` are never empty

```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/linear_algebra# ./test_files/14-main.py
[[ 330  396  462]
 [ 726  891 1056]]
[[ 550]
 [1342]]
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/linear_algebra# 
```
<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 15. Slice Like A Ninja


![Advanced](https://img.shields.io/badge/advanced-🚀-blueviolet)

Write a function `def np_slice(matrix, axes={}):` that slices a matrix along specific axes:

- You can assume that `matrix` is a numpy.ndarray
- You must return a new numpy.ndarray
- `axes` is a dictionary where:
  - The key is an axis to slice along
  - The value is a tuple representing the slice to make along that axis
- You can assume that `axes` represents a valid slice
- [Hint](https://docs.python.org/3/library/functions.html#slice)

```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/linear_algebra# ./test_files/100-main.py 
[[2 3]
 [7 8]]
[[ 1  2  3  4  5]
 [ 6  7  8  9 10]]
[[[ 5  3  1]
  [10  8  6]]

 [[15 13 11]
  [20 18 16]]]
[[[ 1  2  3  4  5]
  [ 6  7  8  9 10]]

 [[11 12 13 14 15]
  [16 17 18 19 20]]

 [[21 22 23 24 25]
  [26 27 28 29 30]]]
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/linear_algebra# 
```

<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 16. Bracing The Elements

![Advanced](https://img.shields.io/badge/advanced-🚀-blueviolet)


Write a function `def add_matrices(mat1, mat2):` that adds two matrices:

- You can assume that `mat1` and `mat2` are matrices containing ints/floats
- You can assume all elements in the same dimension are of the same type/shape
- You must return a new matrix
- If matrices are not the same shape, return `None`
- You can assume that `mat1` and `mat2` will never be empty

```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/linear_algebra# ./test_files/101-main.py
[5, 7, 9]
[[6, 8], [10, 12]]
[[[[12, 14, 16, 18], [20, 22, 24, 26]], [[28, 120, 122, 124], [126, 128, 130, 132]], [[134, 136, 138, 140], [142, 144, 146, 148]]], [[[150, 152, 154, 156], [158, 160, 162, 164]], [[166, 168, 170, 172], [174, 176, 178, 180]], [[182, 184, 186, 188], [190, 192, 194, 196]]]]
None
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/linear_algebra# 
```

<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 17. Squashed Like Sardines

![Advanced](https://img.shields.io/badge/advanced-🚀-blueviolet)

Write a function `def cat_matrices(mat1, mat2, axis=0):` that concatenates two matrices along a specific axis:

- You can assume that `mat1` and `mat2` are matrices containing ints/floats
- You can assume all elements in the same dimension are of the same type/shape
- You must return a new matrix
- If you cannot concatenate the matrices, return `None`
- You can assume that `mat1` and `mat2` are never empty

**Note:** *The time difference between the standard Python3 library and the numpy library is an order of magnitude! When you have matrices with millions of data points, this time adds up!*
```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/linear_algebra# ./test_files/102-main.py
1.6927719116210938e-05
[1, 2, 3, 4, 5, 6]
4.76837158203125e-06 

1.8358230590820312e-05
[[1, 2], [3, 4], [5, 6], [7, 8]]
3.0994415283203125e-06 

1.7881393432617188e-05
[[1, 2, 5, 6], [3, 4, 7, 8]]
6.9141387939453125e-06 

0.00016427040100097656
[[[[1, 2, 3, 4, 11, 12, 13, 14], [5, 6, 7, 8, 15, 16, 17, 18]], [[9, 10, 11, 12, 19, 110, 111, 112], [13, 14, 15, 16, 113, 114, 115, 116]], [[17, 18, 19, 20, 117, 118, 119, 120], [21, 22, 23, 24, 121, 122, 123, 124]]], [[[25, 26, 27, 28, 125, 126, 127, 128], [29, 30, 31, 32, 129, 130, 131, 132]], [[33, 34, 35, 36, 133, 134, 135, 136], [37, 38, 39, 40, 137, 138, 139, 140]], [[41, 42, 43, 44, 141, 142, 143, 144], [45, 46, 47, 48, 145, 146, 147, 148]]]]
5.030632019042969e-05 

0.00020313262939453125
[[[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]], [[17, 18, 19, 20], [21, 22, 23, 24]], [[11, 12, 13, 14], [15, 16, 17, 18]], [[117, 118, 119, 120], [121, 122, 123, 124]]], [[[25, 26, 27, 28], [29, 30, 31, 32]], [[33, 34, 35, 36], [37, 38, 39, 40]], [[41, 42, 43, 44], [45, 46, 47, 48]], [[125, 126, 127, 128], [129, 130, 131, 132]], [[141, 142, 143, 144], [145, 146, 147, 148]]]]
1.5735626220703125e-05 

None
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/math/linear_algebra# 
```

---
# 📄 Files

| Task Number | Task Title                   |File                 | Priority                                                             |
|-------------|------------------------------|---------------------|----------------------------------------------------------------------|
| 0           | 0. Slice Me Up                | `0-slice_me_up.py`   | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |
| 1           | 1. Trim Me Down              | `1-trim_me_down.py` | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |
| 2           | 2. Size Me Please             | `2-size_me_please.py`     | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |
| 3           | 3. Flip Me Over                 | `3-flip_me_over.py`    | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |
| 4           | 4. Line Up               | `4-line_up.py`   | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |
| 5           | 5. Across The Planes              | `5-across_the_planes.py` | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |
| 6           | 6. Howdy Partner            | `6-howdy_partner.py`     | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |
| 7           | 7. Gettin’ Cozy                 | `7-gettin_cozy.py`    | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |
| 8           | 8. Ridin’ Bareback             | `8-ridin_bareback.py` | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |
| 9           | 9. Let The Butcher Slice It    | `9-let_the_butcher_slice_it.py`     | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |
| 10          | 10. I’ll Use My Scale               | `10-ill_use_my_scale.py`    | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |
| 11          |11. The Western Exchange          | `11-the_western_exchange.py`   | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |
|12           | 12. Bracing The Elements              | `12-bracin_the_elements.py` | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |
| 13          | 13. Cat's Got Your Tongue           | `13-cats_got_your_tongue.py`     | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |
| 14          | 14. Saddle Up                 | `14-saddle_up.py`    | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |
| 15          |15. Slice Like A Ninja     | `100-slice_like_a_ninja.py` | ![Advanced](https://img.shields.io/badge/advanced-🚀-blueviolet)    |
| 16          |16. The Whole Barn     | `101-the_whole_barn.py` | ![Advanced](https://img.shields.io/badge/advanced-🚀-blueviolet)    |
| 17          |17. Squashed Like Sardines   | `102-squashed_like_sardines.py` | ![Advanced](https://img.shields.io/badge/advanced-🚀-blueviolet)    |
---

# 📊 Project Summary

This project teaches core linear algebra **concepts (vectors, matrices, operations)** using **NumPy**, focusing on efficient array computations. Students implement operations like dot products and matrix multiplication while learning NumPy's broadcasting and vectorization features, following strict Python coding standards.

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

