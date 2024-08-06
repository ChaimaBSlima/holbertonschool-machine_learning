#!/usr/bin/env python3
""" Task 5: 5. All in One """
import numpy as np
import matplotlib.pyplot as plt


def all_in_one():
    """
    Generate and display multiple plots in a single figure.
    1. A plot of the cubic function.
    2. A scatter plot of men's height vs. weight.
    3. A plot of the exponential decay of Carbon-14 with a logarithmic y-scale.
    4. A plot comparing the exponential decay of Carbon-14 and Radium-226.
    5. A histogram of student grades.

    Args:
        None

    Returns:
        None
    """
    y0 = np.arange(0, 11) ** 3

    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180

    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)

    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    fig = plt.figure()
    # Plot 0 Task 0 #
    plot0 = fig.add_subplot(3, 2, 1)
    plot0.plot(y0, 'r-')
    plot0.set_xlim((0, 10))
    # Plot 1 Task 1 #
    plot1 = fig.add_subplot(3, 2, 2)
    plot1.scatter(x1, y1, c='m')
    plot1.set_xlabel('Height (in)', fontsize='x-small')
    plot1.set_ylabel('Weight (lbs)', fontsize='x-small')
    plot1.set_title("Men's Height vs Weight", fontsize='x-small')
    # Plot 2 Task 2 #
    plot2 = fig.add_subplot(3, 2, 3)
    plot2.plot(x2, y2)
    plot2.set_xlim((0, 28650))
    plot2.set_yscale("log")
    plot2.set_xlabel('Time (years)', fontsize='x-small')
    plot2.set_ylabel('Fraction Remaining', fontsize='x-small')
    plot2.set_title("Exponential Decay of C-14", fontsize='x-small')
    # Plot 3 Task 3 #
    plot3 = fig.add_subplot(3, 2, 4)
    plot3.plot(x3, y31, "r--", label='C-14')
    plot3.plot(x3, y32, "g", label='Ra-226')
    plot3.set_xlim((0, 20000))
    plot3.set_ylim((0, 1))
    plot3.set_xlabel('Time (years)', fontsize='x-small')
    plot3.set_ylabel('Fraction Remaining', fontsize='x-small')
    plot3.set_title("Exponential Decay of Radioactive Elements",
                    fontsize='x-small')
    plot3.legend(loc='upper right',  fontsize='x-small')
    # Plot 4 Task 4 #
    plot4 = fig.add_subplot(3, 1, 3)
    plot4.hist(student_grades, bins=np.arange(0, 110, 10), edgecolor='black')
    plot4.set_xlim((0, 100))
    plot4.set_xticks(np.arange(0, 110, 10))
    plot4.set_ylim((0, 30))
    plot4.set_xlabel('Grades', fontsize='x-small')
    plot4.set_ylabel('Number of Students', fontsize='x-small')
    plot4.set_title("Project A", fontsize='x-small')

    fig.suptitle("All in One")
    plt.tight_layout()
    plt.show()
