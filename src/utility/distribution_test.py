import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def normal_distribution_test(sample, title='Sample Distribution', xlabel='Value'):
    # Plot histogram of sample
    plt.hist(sample, bins=30, density=True, alpha=0.6, color='g')
    plt.title(f'Histogram of {title}')
    plt.xlabel(xlabel)
    plt.ylabel('Density')

    # Fit a normal distribution to the data
    mu, std = stats.norm.fit(sample)

    # Plot the PDF of the fitted normal distribution
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    plt.show()

    # Perform normality test (Shapiro-Wilk)
    stat, p_value = stats.shapiro(sample)
    print(f'Shapiro-Wilk test statistic: {stat}, p-value: {p_value}')
    if p_value > 0.05:
        print(f"{title} appears to be normally distributed (fail to reject H0)")
    else:
        print(f"{title} does not appear to be normally distributed (reject H0)")
