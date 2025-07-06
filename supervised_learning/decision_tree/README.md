<h1><p align="center"> Decision Tree & Random Forest  </h1></p></font>

*For this project, we expect you to look at this concept:*

- *[What is a decision tree?](https://github.com/ChaimaBSlima/Valuable-IT-Concepts/blob/main/decision_tree_predVSpredict.md)*
- *[Decision_Tree.pred vs Decision_Tree.predict](https://github.com/ChaimaBSlima/Valuable-IT-Concepts/blob/main/decision_tree_predVSpredict.md)*

---

<p align="center">
  <img src="https://github.com/user-attachments/assets/e621c6dd-724b-47fb-b298-0dc983efca52" alt="Image"/>
</p>


# 📚 Resources

Read or watch:


- [Rokach and Maimon (2002) : Top-down induction of decision trees classifiers : a survey](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=cef4d6ec6505d3d3dcbc9365802947dda107dba2)
- [Ho et al. (1995) : Random Decision Forests](https://web.archive.org/web/20160417030218/http://ect.bell-labs.com/who/tkh/publications/papers/odt.pdf)
- [Fei et al. (2008) : Isolation forests](https://www.lamda.nju.edu.cn/publication/icdm08b.pdf)
- [Gini and Entropy clearly explained : Handling Continuous features in Decision Trees](https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/misc/2024/4/ba488060b38f19d4d174fa4e377e97139ae0737b.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20250704%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20250704T141813Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=60ee75d76f98b1018ad951e7c3802b7857341a30cf5ec0e560ba9745f047c1dc)
- [Abspoel and al. (2021) : Secure training of decision trees with continuous attributes](https://eprint.iacr.org/2020/1130.pdf)
- [Threshold Split Selection Algorithm for Continuous Features in Decision Tree](https://www.youtube.com/watch?v=asf1h2Onq4A)
- [Splitting Continuous Attribute using Gini Index in Decision Tree](https://www.youtube.com/watch?v=41SHQjwuQ5o)
- [How to handle Continuous Valued Attributes in Decision Tree](https://www.youtube.com/watch?v=2vIvM4zmyf4)
- [Decision Tree problem based on the Continuous-valued attribute](https://www.youtube.com/watch?v=J_HEu5WqHao)
- [How to Implement Decision Trees in Python using Scikit-Learn(sklearn)](https://www.youtube.com/watch?v=wxS5P7yDHRA)
- [Matching and Prediction on the Principle of Biological Classification by William A. Belson](https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/misc/2023/11/a4c3869a9204cf142a286d545f899b720bf1e685.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20250704%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20250704T142213Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=723d59831bd5a0cc6869e72c3cb102793260b4d88335e13fd3b4bf926fc97b37)
- [DecisionTree.pred vs DecisionTree.predict](https://intranet.hbtn.io/concepts/1200)
- [DecisionTree.pred vs DecisionTree.predict](https://intranet.hbtn.io/concepts/1200)

### Notes

- This project aims to implement decision trees from scratch. It is important for engineers to understand how the tools we use are built for two reasons.  
First, it gives us confidence in our skills. Second, it helps us when we need to build our own tools to solve unsolved problems.

- The first three references point to historical papers where the concepts were first studied.  
- References 4 to 9 can help if you feel you need some more explanation about the way we split nodes.  
- William A. Belson is usually credited for the invention of decision trees (read reference 11).  

- Despite our efforts to make it efficient, we cannot compete with Sklearn’s implementations (since they are done in C).  
- In real life, it is thus recommended to use Sklearn’s tools.  
- In this regard, it is warmly recommended to watch the video referenced as (10) above. It shows how to use Sklearn’s decision trees and insists on the methodology.

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
# 🧪 Tasks

We will progressively add methods in the following 3 classes:

```python
class Node:
    def __init__(self, feature=None, threshold=None, left_child=None, right_child=None, is_root=False, depth=0):
        self.feature                  = feature
        self.threshold                = threshold
        self.left_child               = left_child
        self.right_child              = right_child
        self.is_leaf                  = False
        self.is_root                  = is_root
        self.sub_population           = None    
        self.depth                    = depth
                
class Leaf(Node):
    def __init__(self, value, depth=None) :
        super().__init__()
        self.value   = value
        self.is_leaf = True
        self.depth   = depth

class Decision_Tree() :
    def __init__(self, max_depth=10, min_pop=1, seed=0,split_criterion="random", root=None) :
        self.rng               = np.random.default_rng(seed)
        if root :
            self.root          = root
        else :
            self.root          = Node(is_root=True)
        self.explanatory       = None
        self.target            = None
        self.max_depth         = max_depth
        self.min_pop           = min_pop
        self.split_criterion   = split_criterion
        self.predict           = None
```
- Once built, decision trees are **binary trees**: a node either is a **leaf** or has **two children**.  
It never happens that a node for which `is_leaf` is `False` has its `left_child` or `right_child` left unspecified.

- The first three tasks are a warm-up designed to review the basics of **class inheritance and recursion**  
(nevertheless, the functions coded in these tasks will be reused in the rest of the project).

- Our first objective will be to write a `Decision_Tree.predict` method that takes the **explanatory features** of a set of individuals  
and returns the **predicted target value** for these individuals.

- Then we will write a method `Decision_Tree.fit` that takes the **explanatory features** and the **targets** of a set of individuals,  
and grows the tree from the **root** to the **leaves** to make it into an efficient prediction tool.

- Once these tasks are accomplished, we will introduce a new class `Random_Forest` that will also be a **powerful prediction tool**.

- Finally, we will write a variation on `Random_Forest`, called `Isolation_Random_forest`, that will be a tool to **detect outliers**.

---

# 🧾 Requirements

- **You should carefully read all the concept pages attached above.**  
- All your files will be interpreted/compiled on Ubuntu 20.04 LTS using `python3` (version 3.9)
- Your files will be executed with `numpy` (version 1.25.2)  
- All your files should end with a new line  
- Your code should use the `pycodestyle` style (version 2.11.1)  
- The first line of all your files should be exactly: `#!/usr/bin/env python3`  
- A `README.md` file, at the root of the folder of the project, is mandatory  
- All your modules should have documentation (`python3 -c 'print(__import__("my_module").__doc__)'`)  
- All your classes should have documentation (`python3 -c 'print(__import__("my_module").MyClass.__doc__)'`)  
- All your functions (inside and outside a class) should have documentation (`python3 -c 'print(__import__("my_module").my_function.__doc__)'` and `python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)'`)  
- All your files must be executable

---


# 📝 Tasks

### 0. Depth of a decision tree

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

All the nodes of a decision tree have their `depth` attribute. The depth of the root is `0`, while the children of a node at depth k have a depth of `k+1`. We want to find the maximum of the depths of the nodes (including the leaves) in a decision tree. In order to do so, we added a method `def depth(self)`: in the `Decision_Tree` class, a method `def max_depth_below(self)`: in the `Leaf` class.

**Task:** Update the class `Node` by adding the method `def max_depth_below(self):`.

Down below is the content of the file `0-build_decision_tree.py`.

```python
#!/usr/bin/env python3

import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left_child=None, right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self) :

            ####### FILL IN THIS METHOD

class Leaf(Node):
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self) :
        return self.depth

class Decision_Tree():
    def __init__(self, max_depth=10, min_pop=1, seed=0, split_criterion="random", root=None):
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self) :
        return self.root.max_depth_below()
```
**Main to test your work**

```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/supervised_learning/decision_tree# ./test_files/0-main.py
2
5
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/supervised_learning/decision_tree# 
```

<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 1. Number of nodes/leaves in a decision tree

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

We now want to count the number of nodes in a decision tree, potentially excluding the root and internal nodes to count only the leaves. In order to do so, we added a method `def count_nodes(self, only_leaves=False):` in the `Decision_Tree` class:

```
def count_nodes(self, only_leaves=False) :
    return self.root.count_nodes_below(only_leaves=only_leaves)
```
we added a method `def count_nodes_below(self, only_leaves=False):` in the `Leaf` class:

```
def count_nodes_below(self, only_leaves=False) :
    return 1
```
**Task:** Update the class Node by adding the method `def count_nodes_below(self, only_leaves=False):`.

**Main to test your work**
```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/supervised_learning/decision_tree#./test_files/1-main.py
Number of nodes  in example 0 : 5
Number of leaves in example 0 : 3
Number of nodes  in example 1 : 31
Number of leaves in example 1 : 16
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/supervised_learning/decision_tree# 
```

<p align="center">⭐⭐⭐⭐⭐⭐</p>


### 2. Let's print our Tree

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)

In this task, we give you the `def __str__(self) :` method for the `Decision_Tree` class :
```python
def __str__(self):
    return self.root.__str__()
```
and the `def __str__(self) :` method for the `Leaf` class :

```python
def __str__(self):
    return (f"-> lef [value={self.value}]")
```
**Task:** Insert the above declarations in the respective classes, and update the class `Node` by adding the method `def __str__(self) :`

**Hint 1:** You might need some functions `def left_child_add_prefix(text):` and `def right_child_add_prefix(text):` at some point.
**Hint 2:** In order to help you in this task, we gave you the function `def left_child_add_prefix(text):` and now your mission is to add the function `def right_child_add_prefix(text):` to be able to implement the method `def __str__(self)/`

```python
def left_child_add_prefix(self,text):
    lines=text.split("\n")
    new_text="    +--"+lines[0]+"\n"
    for x in lines[1:] :
        new_text+=("    |  "+x)+"\n"
    return (new_text)
```
**Main to test your work**
```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/supervised_learning/decision_tree#./test_files/2-main.py
root [feature=0, threshold=0.5]
    +---> leaf [value=0]
    +---> node [feature=1, threshold=30000]
           +---> leaf [value=0]
           +---> leaf [value=1]

root [feature=0, threshold=7.5]
    +---> node [feature=0, threshold=11.5]
    |      +---> node [feature=0, threshold=13.5]
    |      |      +---> node [feature=0, threshold=14.5]
    |      |      |      +---> leaf [value=15]
    |      |      |      +---> leaf [value=14]
    |      |      +---> node [feature=0, threshold=12.5]
    |      |             +---> leaf [value=13]
    |      |             +---> leaf [value=12]
    |      +---> node [feature=0, threshold=9.5]
    |             +---> node [feature=0, threshold=10.5]
    |             |      +---> leaf [value=11]
    |             |      +---> leaf [value=10]
    |             +---> node [feature=0, threshold=8.5]
    |                    +---> leaf [value=9]
    |                    +---> leaf [value=8]
    +---> node [feature=0, threshold=3.5]
           +---> node [feature=0, threshold=5.5]
           |      +---> node [feature=0, threshold=6.5]
           |      |      +---> leaf [value=7]
           |      |      +---> leaf [value=6]
           |      +---> node [feature=0, threshold=4.5]
           |             +---> leaf [value=5]
           |             +---> leaf [value=4]
           +---> node [feature=0, threshold=1.5]
                  +---> node [feature=0, threshold=2.5]
                  |      +---> leaf [value=3]
                  |      +---> leaf [value=2]
                  +---> node [feature=0, threshold=0.5]
                         +---> leaf [value=1]
                         +---> leaf [value=0]
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/supervised_learning/decision_tree# 
```

<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 3. Towards the predict function (1) : the get_leaves method

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)


Task: Insert the following declarations in their respective classes, and update the class `Node` by adding the method `def get_leaves_below(self) :` that returns the list of all leaves of the tree.

 - Add in class `Leaf`:
```python
def get_leaves_below(self) :
    return [self]

```
 - Add in class Decision_Tree:

```python
def get_leaves(self) :
    return self.root.get_leaves_below()
```

**Main to test your work**
```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/supervised_learning/decision_tree#./test_files/3-main.py
## Tree 1
-> leaf [value=0] 
-> leaf [value=0]
-> leaf [value=1]
## Tree 2
-> leaf [value=7]
-> leaf [value=6]
-> leaf [value=5]
-> leaf [value=4]
-> leaf [value=3]
-> leaf [value=2]
-> leaf [value=1]
-> leaf [value=0]
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/supervised_learning/decision_tree# 
```

<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 4. Towards the predict function (2) : the update_bounds method

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)


**Task:** Insert the following declarations in their respective classes, and update the class `Node` by completing the method `def get_leaves_below(self) :`

- This method should recursively compute, for each node, two dictionaries stored as attributes `Node.lower` and `Node.upper`.
- These dictionaries should contain the bounds of the node for each feature.
- The lower and upper bounds representa the minimum and maximum values, respectively, observed in the data subset associated with that node.
- The keys in the dictionary represent the features.

- Add in class `Leaf`:
```python
    def update_bounds_below(self) :
        pass 
```
- Add in class  `Decision_Tree`:
```python
    def update_bounds(self) :
        self.root.update_bounds_below() 
```
- Fill in `def update_bounds_below(self) :` in class `Node`:
```python
    def update_bounds_below(self) :
        if self.is_root : 
            self.upper = { 0:np.inf }
            self.lower = {0 : -1*np.inf }

        for child in [self.left_child, self.right_child] :

                         # To Fill : compute and attach the lower and upper dictionaries to the children

        for child in [self.left_child, self.right_child] :
            child.update_bounds_below()
```
**Main to test your work**
```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/supervised_learning/decision_tree#./test_files/4-main.py
example_ 0
  leaf number  0
    lower : {0: 0.5}
    upper : {0: inf}
  leaf number  1
    lower : {0: -inf, 1: 30000}
    upper : {0: 0.5}
  leaf number  2
    lower : {0: -inf}
    upper : {0: 0.5, 1: 30000}
example_ 1
  leaf number  0
    lower : {0: 30.5}
    upper : {0: inf}
  leaf number  1
    lower : {0: 29.5}
    upper : {0: 30.5}
  leaf number  2
    lower : {0: 28.5}
    upper : {0: 29.5}
  leaf number  3
    lower : {0: 27.5}
    upper : {0: 28.5}
  leaf number  4
    lower : {0: 26.5}
    upper : {0: 27.5}
  leaf number  5
    lower : {0: 25.5}
    upper : {0: 26.5}
  leafper : {0: 16.5}
  leaf number  16
    lower : {0: 14.5}
    upper : {0: 15.5}
  leaf number  17
    lower : {0: 13.5}
    upper : {0: 14.5}
  leaf number  18
    lower : {0: 12.5}
    upper : {0: 13.5}
  leaf number  19
    lower : {0: 11.5}
    upper : {0: 12.5}
  leaf number  20
    lower : {0: 10.5}
    upper : {0: 11.5}
  leaf number  21
    lower : {0: 9.5}
    upper : {0: 10.5}
  leaf number  22
    lower : {0: 8.5}
    upper : {0: 9.5}
  leaf number  23
    lower : {0: 7.5}
    upper : {0: 8.5}
  leaf number  24
    lower : {0: 6.5}
    upper : {0: 7.5}
  leaf number  25
    lower : {0: 5.5}
    upper : {0: 6.5}
  leaf number  26
    lower : {0: 4.5}
    upper : {0: 5.5}
  leaf number  27
    lower : {0: 3.5}
    upper : {0: 4.5}
  leaf number  28
    lower : {0: 2.5}
    upper : {0: 3.5}
  leaf number  29
    lower : {0: 1.5}
    upper : {0: 2.5}
  leaf number  30
    lower : {0: 0.5}
    upper : {0: 1.5}
  leaf number  31
    lower : {0: -inf}
    upper : {0: 0.5}
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/supervised_learning/decision_tree# 
```

<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 5. Towards the predict function (3): the update_indicator method

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)


Consider the indicator function for a given node, denoted as "n." This function is defined as follows:

- It takes a 2D NumPy array, denoted as `A`, of shape `(n_individuals, n_features)`.
- The output of the indicator function is a 1D NumPy array, of size equals to the number of individuals (`n_individuals`), containing boolean values.
- The `i`-th element of this output array is set to `True` if the corresponding `i`-th individual meets the conditions specified by the node "n"; otherwise, it is set to `False`.

**Task:** Write a method `Node.update_indicator` that computes the indicator function from the `Node.lower` and `Node.upper` dictionaries and stores it in an attribute `Node.indicator`:

Fill in `def update_indicator(self):` in class `Node`:

```python
def update_indicator(self) :

        def is_large_enough(x):

                #<- fill the gap : this function returns a 1D numpy array of size 
                #`n_individuals` so that the `i`-th element of the later is `True` 
                # if the `i`-th individual has all its features > the lower bounds

        def is_small_enough(x):

                #<- fill the gap : this function returns a 1D numpy array of size 
                #`n_individuals` so that the `i`-th element of the later is `True` 
                # if the `i`-th individual has all its features <= the lower bounds

        self.indicator = lambda x : np.all(np.array([is_large_enough(x),is_small_enough(x)]),axis=0)
```
**Hint:** you might want to consider something like `np.array([np.greater(A[:,key],self.lower[key]) for key in list(self.lower.keys())]` at some point.

**Main to test your work**
```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/supervised_learning/decision_tree#./test_files/5-main.py
For example_0()
A=
 [[    1 22000]
 [    1 44000]
 [    0 22000]
 [    0 44000]]
values of indicators of leaves :
 [[ True  True False False]
 [False False False  True]
 [False False  True False]]


For example_1(4)
A=
 [[11.65 ]
 [ 6.917]]
values of indicators of leaves :
 [[False False]
 [False False]
 [False False]
 [ True False]
 [False False]
 [False False]
 [False False]
 [False False]
 [False  True]
 [False False]
 [False False]
 [False False]
 [False False]
 [False False]
 [False False]
 [False False]]
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/supervised_learning/decision_tree# 
```

<p align="center">⭐⭐⭐⭐⭐⭐</p>

### 6. The predict function

![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen)


We are now in a position to write our efficient `Decision_Tree.predict` function.

**Task:** Write a method `Decision_Tree.update_predict` that computes the prediction function:

Fill in `def update_predict(self):` in class `Decision_Tree`:

```python
def update_predict(self) :
        self.update_bounds()
        leaves=self.get_leaves()
        for leaf in leaves :
            leaf.update_indicator()          
        self.predict = lambda A: #<--- To be filled
```
In this concept page: [DecisionTree.pred vs DecisionTree.predict](https://github.com/ChaimaBSlima/Valuable-IT-Concepts/blob/main/decision_tree_predVSpredict.md), we introduced an additional approach for implementing a prediction function known as `Decision_Tree.pred`.

As part of the testing process, insert the following methods into their respective classes:

- Add `def pred(self, x):` in class `Leaf`:
```python
   def pred(self,x) :
        return self.value
```
- add `def pred(self,x):` in class `Node`:
```python
    def pred(self,x) :
        if x[self.feature]>self.threshold :
            return self.left_child.pred(x)
        else :
            return self.right_child.pred(x)
```
- add def `pred(self,x):` in class `Decision_Tree`:
```python
    def pred(self,x) :
            return self.root.pred(x)
```
Now, to validate whether `Decision_Tree.pred` performs similarly to the existing `Decision_Tree.predict`, we are creating a generator for random trees. We will compare the behavior of `Decision_Tree.predict` and `Decision_Tree.pred` on a sample explanatory array.
**Main to test your work**

```
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/supervised_learning/decision_tree#./test_files/6-main.py
root [feature=2, threshold=90.09]
    +---> node [feature=2, threshold=91.52]
    |      +---> node [feature=4, threshold=-37.63]
    |      |      +---> node [feature=4, threshold=20.63]
    |      |      |      +---> leaf [value=0]
    |      |      |      +---> leaf [value=2]
    |      |      +---> node [feature=1, threshold=9.92]
    |      |             +---> leaf [value=1]
    |      |             +---> leaf [value=0]
    |      +---> node [feature=0, threshold=50.7]
    |             +---> node [feature=4, threshold=-34.05]
    |             |      +---> leaf [value=1]
    |             |      +---> leaf [value=1]
    |             +---> node [feature=3, threshold=-39.36]
    |                    +---> leaf [value=0]
    |                    +---> leaf [value=1]
    +---> node [feature=4, threshold=-19.38]
           +---> node [feature=0, threshold=-59.31]
           |      +---> node [feature=2, threshold=42.64]
           |      |      +---> leaf [value=0]
           |      |      +---> leaf [value=0]
           |      +---> node [feature=1, threshold=-2.96]
           |             +---> leaf [value=0]
           |             +---> leaf [value=2]
           +---> node [feature=3, threshold=44.96]
                  +---> node [feature=4, threshold=-56.37]
                  |      +---> leaf [value=2]
                  |      +---> leaf [value=0]
                  +---> node [feature=3, threshold=40.6]
                         +---> leaf [value=0]
                         +---> leaf [value=1]

T.pred(A) :
 [0 0 0 0 1 1 2 0 1 0 0 0 2 0 2 0 1 0 1 1 0 1 0 0 0 2 1 0 0 0 1 0 0 0 0 1 0
 1 0 1 0 1 1 0 1 2 0 1 1 0 0 1 0 1 0 1 0 1 1 1 0 0 0 2 0 0 0 1 0 2 2 0 0 0
 0 0 1 0 0 1 0 1 0 1 1 0 0 0 0 0 0 1 0 0 1 1 0 0 0 1]
T.predict(A) :
 [0 0 0 0 1 1 2 0 1 0 0 0 2 0 2 0 1 0 1 1 0 1 0 0 0 2 1 0 0 0 1 0 0 0 0 1 0
 1 0 1 0 1 1 0 1 2 0 1 1 0 0 1 0 1 0 1 0 1 1 1 0 0 0 2 0 0 0 1 0 2 2 0 0 0
 0 0 1 0 0 1 0 1 0 1 1 0 0 0 0 0 0 1 0 0 1 1 0 0 0 1]
Predictions are the same on the explanatory array A : True
root@CHAIMA-LAPTOP:~/holbertonschool-machine_learning/supervised_learning/decision_tree# 
```
**NOTE:** Read the concept page [DecisionTree.pred vs DecisionTree.predict](https://github.com/ChaimaBSlima/Valuable-IT-Concepts/blob/main/decision_tree_predVSpredict.md) to understand how `Decision_Tree.predict` is better than `Decision_Tree.pred` when it comes to computational efficiency.
<p align="center">⭐⭐⭐⭐⭐⭐</p>


---
# 📄 Files

| Task Number | Task Title                   |File                 | Priority                                                             |
|-------------|------------------------------|---------------------|----------------------------------------------------------------------|
| 0           | 0. Depth of a decision tree                | `0-build_decision_tree.py`   | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |
| 1           | 1. Number of nodes/leaves in a decision tree              | `1-build_decision_tree.py` | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |
| 2           | 2. Let's print our Tree             | `2-build_decision_tree.py`     | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |
| 3           | 3. Towards the predict function (1) : the get_leaves method                 | `3-build_decision_tree.py`    | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |
| 4           | 4. Towards the predict function (2) : the update_bounds method | `4-build_decision_tree.py`   | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |
| 5           | 5. Towards the predict function (3): the update_indicator method              | `5-build_decision_tree.py` | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |
| 6           | 6. The predict function            | `6-build_decision_tree.py`     | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |
| 7           | 7. Training decision trees                 | `7-build_decision_tree.py`    | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |
| 8           | 8. Using Gini impurity function as a splitting criterion            | `8-build_decision_tree.py` | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |
| 9           | 9. Random forests    | `9-random_forest.py`     | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |
| 10          | 10. IRF 1 : isolation random trees               | `10-isolation_tree.py`    | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |
| 11          |11. IRF 2 : isolation random forests         | `11-isolation_forest.py`   | ![Mandatory](https://img.shields.io/badge/mandatory-✅-brightgreen) |

---

# 📊 Project Summary

This project dives into **decision tree algorithms**, covering core concepts like `Gini impurity`, `entropy`, and `continuous feature splitting`. You'll implement a decision tree from scratch while studying foundational papers and modern techniques. The project contrasts manual implementation with scikit-learn's optimized version, emphasizing both theoretical understanding and practical application in machine learning.

Key aspects:

 - Node splitting strategies for continuous features

 - Comparison of `.pred` vs `.predict`methods

 - Historical context (Belson's work) and modern implementations

 - Practical guidance on using scikit-learn's decision trees

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

