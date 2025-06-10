import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def logistic(k, A, B, k_star, w):
    return A + B / (1 + np.exp(-(k - k_star) / w))

def fit_k(k, acc):
    popt, pcov = curve_fit(logistic, k, acc, 
                        p0=[0, 1, k.mean(), 1]) 
    A, B, k_star, w = popt
    return popt

def plot_fit(k, acc, popt):
    k_smooth = np.linspace(k.min(), k.max(), 100)
    plt.scatter(k, acc, label='Data')
    plt.plot(k_smooth, logistic(k_smooth, *popt), 'r-', label='Fit')
