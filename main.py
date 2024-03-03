# Required Libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sps
import scipy.stats as stats
import warnings

"""_Introduction_
    Let the probability of having a rainy day (defined as a day with precipitation above 5 mm) be p = 0.12 in a certain city.
"""


    # Exercise 1: Bernoulli Distribution
    
# Probability mass function pmf_x (Bernoulli)
def bernoulli(x, p):
    if x == 0 or x == 1:
        pmf_x = (p**x) * ((1 - p)**(1 - x))
    else:
        print("X value outside statistical support")
    return pmf_x

# Calculate the probability of having a rainy day
p = 0.12 # Probability of a rainy day
pmf_bernoulli = bernoulli(1,p)
print(f"\nWith Bernoulli distribution(p={p},x=1):")
print(f"P[X=1] = {pmf_bernoulli}")



    # Exercise 2: Binomial Distribution in 1 week
    
# Probability mass function pmf_x (Binomial)
def binomial(x,N,p):
    pmf_x = sps.comb(N,x) * p**x * (1 - p)**(N - x)
    pmf_x = np.where(np.isnan(pmf_x) | np.isinf(pmf_x), 0.0, pmf_x)  # Replace "nan" and "inf" with 0.0
    return pmf_x

# Array of numbers to calculate pmf_x (Binomial) with N repetitions
N = 7 # Number of days in 1 week
x = np.arange(0,N+1,1)

# Calculate probabilities for each x in the array
pmf_binomial = binomial(x,N,p)

# Graph pmf_x (Binomial)
plt.figure('Exercise 2')
plt.stem(x, pmf_binomial)
plt.xlabel('Number of rainy days')
plt.ylabel('Probability (pmf_x)')
plt.title('Binomial Distribution of X rainy days in 1 week', fontweight="bold")
plt.grid(True)
plt.show()

# Calculate the cumulative probability function cdf_x
cdf_x_binomial = np.cumsum(pmf_binomial)

# a) Probability of having exactly 3 rainy days
pmf_3 = binomial(3,N,p)
print(f"\nWith Binomial distribution(x={x},N={N},p={p})")
print(f"P[X=3] = {pmf_3}")

# b) Probability of having at most 3 rainy days (x≤3)
print(f"P[X≤3] = {cdf_x_binomial[3]}") 

# c) Probability of having at least 2 rainy days (x≥2)
print(f"P[X≥2] = {1 - cdf_x_binomial[1]}") 



    # Exercise 3: Poisson distribution in 1 week
    
# Probability mass function pmf_x (Poisson)
def poisson(x,mu):
    pmf_x = (mu**x * np.exp(- mu)) / sps.factorial(x)
    pmf_x = np.where(np.isnan(pmf_x) | np.isinf(pmf_x), 0.0, pmf_x)  # Replace "nan" and "inf" with 0.0
    return pmf_x

# Calculate probabilities for each x in the array and show descriptive graph
mu = N*p # E[X] of the Binomial
pmf_poisson = poisson(x,mu)

# Plot pmf_x (Poisson)
plt.figure('Exercise 3')
plt.stem(x, pmf_poisson, )
plt.xlabel('Number of rainy days')
plt.ylabel('Probability (pmf_x)')
plt.title(f'Poisson distribution of X rainy days in 1 week with mu = %f' %mu, fontweight="bold")
plt.grid(True)
plt.show()

# Calculate the cumulative probability function cdf_x
cdf_x_poisson = np.cumsum(pmf_poisson)

# a) Probability of having exactly 3 rainy days
pmf_3 = poisson(3,mu)
print(f"\nWith Poisson distribution(x={x},mu={mu})")
print(f"P[X=3] = {pmf_3}")

# b) Probability of having at most 3 rainy days (x≤3)
print(f"P[X≤3] = {cdf_x_poisson[3]}") 

# c) Probability of having at least 2 rainy days (x≥2)
print(f"P[X≥2] = {1 - cdf_x_poisson[1]}") 

# Comparar Binomial vs Poisson 
plt.figure('Exercise 3.2')
plt.plot(x,pmf_binomial, label = 'Binomial', marker='o', color="darkturquoise")
plt.plot(x,pmf_poisson, label = 'Poisson', marker='o', color='crimson')
plt.xlabel('Number of rainy days')
plt.ylabel('Probability (pmf_x)')
plt.title("Comparison between distributions of X rainy days in 1 week", fontweight="bold")
plt.grid(True)
plt.legend()
plt.show()



    # Exercise 4: Binomial Distribution in 1 year

# Array of numbers to calculate pmf_x (Binomial) with N repetitions
N = 365 # Number of days in 1 year
x = np.arange(0,N+1,1)

# Calculate probabilities for each x in the array
pmf_binomial_y = binomial(x,N,p)

# Graph pmf_x (Binomial)
plt.figure('Exercise 4')
plt.stem(x, pmf_binomial_y)
plt.xlabel('Number of rainy days')
plt.ylabel('Probability (pmf_x)')
plt.title('Binomial Distribution of X rainy days in 1 year', fontweight="bold")
plt.grid(True)
plt.show()

# Calculate the cumulative probability function cdf_x
cdf_x_binomial_y = np.cumsum(pmf_binomial_y)

# a) Probability of having exactly 100 rainy days
pmf_100 = binomial(100,N,p)
print(f"\nWith Binomial distribution(x=[0,1,2,...,365],N={N},p={p})")
print(f"P[X=100] = {pmf_100}")

# b) Probability of having at most 200 rainy days (x≤200)
print(f"P[X≤200] = {cdf_x_binomial_y[200]}") 

# c) Probability of having at least 50 rainy days (x≥50)
print(f"P[X≥50] = {1 - cdf_x_binomial_y[49]}") 



    # Exercise 5: Poisson distribution in 1 year
warnings.filterwarnings("ignore")
    
# Calculate probabilities for each x in the array
mu = N*p # E[X] of the Binomial
pmf_poisson_y = poisson(x,mu)

# Graficar la pmf_x (Poisson)
plt.figure('Exercise 5')
plt.stem(x, pmf_poisson_y)
plt.xlabel('Number of rainy days')
plt.ylabel('Probability (pmf_x)')
plt.title(f'Poisson distribution of X rainy days in 1 year with mu = %f' %mu, fontweight="bold")
plt.grid(True)
plt.show()

# Calculate the cumulative probability function cdf_x
cdf_x_poisson_y = np.cumsum(pmf_poisson_y)

# a) Probability of having exactly 100 rainy days
pmf_100 = poisson(100,mu)
print(f"\nWith Poisson distribution(x=[0,1,2,...,365],mu={mu})")
print(f"P[X=100] = {pmf_100}")

# b) Probability of having at most 200 rainy days (x≤200)
print(f"P[X≤200] = {cdf_x_poisson_y[200]}") 

# c) Probability of having at least 50 rainy days (x≥50)
print(f"P[X≥50] = {1 - cdf_x_poisson_y[49]}") 

# Compare Binomial vs Poisson 
plt.figure('Exercise 5.2')
plt.plot(x,pmf_binomial_y, label = 'Binomial', marker='o', color='darkturquoise')
plt.plot(x,pmf_poisson_y, label = 'Poisson', marker='o', color='crimson')
plt.xlabel('Number of rainy days')
plt.ylabel('Probability (pmf_x)')
plt.title("Comparison between distributions of X rainy days in 1 year", fontweight="bold")
plt.grid(True)
plt.legend()
plt.show()



    # Exercise 6: Distributions with different probability values in 1 year
    
# Define our probabilities and calculate each Binomial pmf with them
p1, p2, p3 = 0.05, 0.5, 0.95
pmf_binomial_p1 = binomial(x,N,p1)
pmf_binomial_p2 = binomial(x,N,p2)
pmf_binomial_p3 = binomial(x,N,p3)

# Define mu for the Binomial pmf, and calculate each Poisson pmf with these
mu1, mu2, mu3 = N*p1, N*p2, N*p3 # E[X] of each binomial distribution
pmf_poisson_p1 = stats.poisson.pmf(x,mu1)
pmf_poisson_p2 = stats.poisson.pmf(x,mu2)
pmf_poisson_p3 = stats.poisson.pmf(x,mu3)

# Create a figure with three subplots
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
p_values = [0.05, 0.5, 0.95]
pmfs_binomial = [pmf_binomial_p1, pmf_binomial_p2, pmf_binomial_p3]
pmfs_poisson = [pmf_poisson_p1, pmf_poisson_p2, pmf_poisson_p3]

# Loop to graph the pmf_x and compare Binomial vs Poisson with each p value
for i in range(3):
    axs[i].plot(x, pmfs_binomial[i], label='Binomial', marker='o', color='darkturquoise')
    axs[i].plot(x, pmfs_poisson[i], label='Poisson', marker='o', color='crimson')
    axs[i].set_xlabel('Number of rainy days')
    axs[i].set_ylabel('Probability (pmf_x)')
    axs[i].set_title(f"p = {p_values[i]}", fontstyle="italic")
    axs[i].grid(True)
    axs[i].legend()

# Format subplots
plt.subplots_adjust(top=0.85, hspace=0.3, wspace=0.3)
plt.suptitle("Comparison between distributions with different probabilities of rain in 1 year\n", fontweight="bold")
plt.show()

print(f"Therefore, the best correspondence occurs in p={p1}, and the worst in p={p3}. \nThat is, the lower the value of p, the better the correspondence.")




    # Exercise 7: Geometric Distribution
    
# Probability mass function pmf_y (Geometric)
def geometric(y, p):
    pmf_y = p * ((1 - p)**(y - 1))
    return pmf_y

# Number array to calculate pmf_y (Geometric) with N repetitions
N = 100 # Arbitrary number of rainy days
y = np.arange(1,N+1,1)

# Calculate probabilities for each y in the array
pmf_geometric = geometric(y,p) # Remembering that p=0.12

# Graph the pmf_x (Geometric)
plt.figure('Exercise 7')
plt.stem(y, pmf_geometric)
plt.xlabel('Number of rainy days')
plt.ylabel('Probability (pmf_y)')
plt.title('Geometric Distribution of Y rainy days in 100 days', fontweight="bold")
plt.grid(True)
plt.show()

# Calculate the cumulative probability function cdf_y
cdf_y_geometric = np.cumsum(pmf_geometric)

# a) Probability that the first rainy day will be 10
pmf_10 = geometric(10,p) 
print(f"\nWith Geometric distribution(y=[0,1,2,...,100],p={p})")
print(f"P[Y=10] = {pmf_10}")

# b) Probability that the first rainy day is not equal to or greater than 11: [1 - (y≥11)] = (y<11) = y(≤10)
print(f"P[Y<11] = {cdf_y_geometric[10-1]}") # - 1 because our array does not start at 0

# c) Probability of having to wait at most 31 days (inclusive) to have a rainy day p(≤31)
print(f"P[Y≤31] = {cdf_y_geometric[31-1]}") # - 1 because our array does not start at 0



    # Exercise 8: Exponential Distribution
    
# Probability density function pdf_y (Exponential)
def exponential(y, theta):
    pmf_y = 1 / theta * np.exp(- y / theta)
    return pmf_y

# Cumulative probability function cdf_y (Exponential)
def cdf_exponential(y,theta):
    cdf_y = 1 - np.exp(- y / theta)
    return cdf_y
    
# Define θ and create an array of numbers to calculate pdf_y (Exponential) with N repetitions
mu = p # E[Y] of the Binomial with N=1
theta = 1 / mu  # Binomial - Exponential Relationship
y = np.arange(0,N+1,.01) # N = 100 enough to model behavior at infinity

# Calculate the densities for each y in the array
pdf_y = exponential(y,theta)

# Graph the pdf_y (Exponential)
plt.figure('Exercise 8')
plt.plot(y, pdf_y, color='darkorchid')
plt.xlabel('Number of rainy days')
plt.ylabel('Density (pmf_y)')
plt.title('Exponential Distribution of Y rainy days in 100 days', fontweight="bold")
plt.grid(True)
plt.show() 

# Calculate the cumulative probability function cdf_y
cdf_y_exp = cdf_exponential(y,theta)

# a) Probability that the first rainy day will be 10
print(f"\nWith Exponential distribution(y=[0,1,2,...,100],p={p})")
print(f"P[Y=10] = {0}. Since punctual probabilities do not exist.")

# b) Probability that the first rainy day is not equal to or greater than 11: [1 - (y≥11)] = (y<11) = y(≤11)
print(f"P[Y≤11] = {cdf_y_exp[10*100]}") # * 100 because the array step is 0.01

# c) Probability of having to wait at most 31 days (inclusive) to have a rainy day p(≤31)
print(f"P[Y≤31] = {cdf_y_exp[31*100]}") # * 100 because the array step is 0.01

print("\nSince, the statistical support of our functions, Geometric and Exponential, differ. As the first starts at 1 and the second at 0.")
print("We can notice that the calculations of the probabilities in smaller numbers of rainy days, close to the lower limit, differ to a greater extent.") 
print("On the contrary, with greater numbers of rainy days, these differences are reduced, so the approximations are practically the same.")