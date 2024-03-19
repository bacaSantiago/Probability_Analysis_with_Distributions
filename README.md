# Probability Analysis with Distributions

## Overview

This Python program explores the probabilities associated with rainy days, defined as days with precipitation exceeding 5 mm, in a certain city. The analysis employs various probability distributions, including Bernoulli, Binomial, Poisson, Geometric, and Exponential, to model and evaluate the likelihood of different scenarios.

## Required Libraries
- **NumPy:** Used for numerical operations and array manipulations.
- **Matplotlib:** Employed for graphical representation of probability distributions.
- **SciPy (scipy.special, scipy.stats):** Utilized for mathematical functions and statistical calculations.
- **Warnings:** Used to suppress unnecessary warnings during the analysis.

## Introduction

Let \( p = 0.12 \) represent the probability of having a rainy day in the city. The program aims to provide a comprehensive understanding of the probabilistic nature of rainy days through various distribution models.

## Exercise 1: Bernoulli Distribution

The program begins by introducing the Bernoulli distribution, a fundamental probability distribution for binary outcomes. It calculates the probability of having a rainy day using the Bernoulli distribution formula.

## Exercise 2: Binomial Distribution in 1 week

This section explores the Binomial distribution to model the probability of having a specific number of rainy days in a week. The program not only calculates probabilities for individual scenarios but also graphically represents the Probability Mass Function (pmf) and Cumulative Distribution Function (cdf) of the distribution.

## Exercise 3: Poisson distribution in 1 week

The Poisson distribution is introduced as an alternative approximation to the Binomial distribution in the context of weekly rainy days. Probabilities are calculated and visually compared to the Binomial distribution.

## Exercise 4: Binomial Distribution in 1 year

Extending the analysis to a yearly timeframe, the program utilizes the Binomial distribution to model the probability of different numbers of rainy days. The probabilities of specific scenarios, such as exactly 100, at most 200, and at least 50 rainy days, are calculated and graphically represented.

## Exercise 5: Poisson distribution in 1 year

Similar to Exercise 4, this section uses the Poisson distribution as an approximation to the Binomial distribution for yearly rainy days. The program calculates probabilities and compares the two distributions graphically.

## Exercise 6: Distributions with different probability values in 1 year

The impact of different probability values (p) on the Binomial and Poisson distributions is investigated. The program compares these distributions for three probability values (p=0.05, p=0.5, p=0.95) and analyzes their correspondence.

## Exercise 7: Geometric Distribution

The Geometric distribution is introduced to model the number of days until the first rainy day occurs. Probabilities for specific scenarios, such as the first rainy day being on the 10th day or waiting at most 31 days for rain, are calculated and graphically presented.

## Exercise 8: Exponential Distribution

In this section, the Exponential distribution is employed to model the time until the first rainy day in a continuous context. The program calculates probabilities and compares the results with the Geometric distribution.

## Conclusion

The program concludes with a summary of the findings, emphasizing the significance of the chosen probability distributions in modeling rainy days. It highlights the best correspondence between distributions for different probability values and provides insights into the behavior of these distributions in various temporal contexts.

**Note:** Before running the program, ensure you have the required libraries installed to facilitate a smooth execution.
