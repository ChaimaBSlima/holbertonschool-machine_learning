#!/usr/bin/env python3
""" Task 4: 4. Frequency """
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """
    Generate and display a histogram of student grades.

    Args:
        None

    Returns:
        None
    """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))
    plt.hist(student_grades, bins=np.arange(0, 110, 10), edgecolor='black')
    plt.xlim((0, 100))
    plt.xticks(np.arange(0, 110, 10))
    plt.ylim((0, 30))
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title("Project A")
    plt.show()
