#!/usr/bin/env python3
""" Task 6: 6. Stacking Bars """
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    Generate and display a stacked bar chart of fruit quantities per person.

    Args:
        None

    Returns:
        None
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))
    plt.ylim((0, 80))
    persons = ["Farrah", "Fred", "Felicia"]
    food = ["apples", "bananas", "oranges", "peaches"]
    colors = ["red",  "yellow", "#ff8000", "#ffe5b4"]
    bar = 0
    for i in range(4):
        bottom = 0
        for j in range(i):
            bottom += fruit[j]
        plt.bar(
            np.arange(len(persons)),
            fruit[bar],
            width=0.5,
            bottom=bottom,
            color=colors[i],
            label=food[i])
        bar += 1
    plt.xticks(np.arange(len(persons)), persons)
    plt.ylabel('Quantity of Fruit')
    plt.title("Number of Fruit per Person")
    plt.legend(loc='upper right')
    plt.show()
