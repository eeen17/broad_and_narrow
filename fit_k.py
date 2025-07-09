import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import bootstrap
import pandas as pd
from typing import Tuple, Dict, List

class PhaseTransitionAnalyzer:
  
    def __init__(self):
        self.results = {}
    
    def logistic_function(self, k, L, alpha, k0):
        return L / (1 + np.exp(-alpha * (k - k0)))
    
    def fit_logistic_curve(self, k_values, accuracy_values, 
                          initial_guess=None, bounds=None):
        """
        Fit logistic curve to accuracy vs k data
        
        Returns:
        - popt: optimal parameters [L, alpha, k0]
        - pcov: covariance matrix
        - r_squared: goodness of fit
        """
        if initial_guess is None:
            # Smart initial guess
            L_guess = np.max(accuracy_values)
            k0_guess = k_values[np.argmin(np.abs(accuracy_values - L_guess/2))]
            alpha_guess = 1.0
            initial_guess = [L_guess, alpha_guess, k0_guess]
        
        if bounds is None:
            # Reasonable bounds
            bounds = ([0, 0,  0], 
                     [1.5*np.max(accuracy_values), 100, np.max(k_values)])
        
        try:
            popt, pcov = curve_fit(self.logistic_function, k_values, accuracy_values,
                                 p0=initial_guess, bounds=bounds, maxfev=5000)
            
            # Calculate R-squared
            y_pred = self.logistic_function(k_values, *popt)
            ss_res = np.sum((accuracy_values - y_pred) ** 2)
            ss_tot = np.sum((accuracy_values - np.mean(accuracy_values)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            return popt, pcov, r_squared
            
        except Exception as e:
            print(f"Fitting failed: {e}")
            return None, None, None
    
    def compute_transition_width(self, alpha, method='10_90'):
        if method == '10_90':
            # Width where logistic goes from 10% to 90% of its range
            return np.log(81) / alpha  # ln(9/0.1) / alpha ≈ 4.4/alpha
        elif method == 'derivative':
            # FWHM of derivative (steepest part)
            return 2 * np.log(2 + np.sqrt(3)) / alpha  # ≈ 2.63/alpha
        else:
            raise ValueError("Method must be '10_90' or 'derivative'")
    
    
    
    def analyze_single_dataset(self, k_values, accuracy_values, 
                              dataset_name="", plot=True, bootstrap_ci=True):

        popt, pcov, r_squared = self.fit_logistic_curve(k_values, accuracy_values)
        
        if popt is None:
            print(f"Failed to fit {dataset_name}")
            return None
        
        L, alpha, k0 = popt
        width = self.compute_transition_width(alpha)
        
        param_errors = np.sqrt(np.diag(pcov)) if pcov is not None else [0, 0, 0]
        
        ci_results = None
        if bootstrap_ci:
            try:
                ci_results = self.bootstrap_confidence_intervals(k_values, accuracy_values)
            except:
                print("Bootstrap CI computation failed")
        
        result = {
            'L': L,
            'alpha': alpha, 
            'k0': k0,
            'width': width,
            'r_squared': r_squared,
            'param_errors': param_errors,
            'confidence_intervals': ci_results
        }
        
        self.results[dataset_name] = result
        
        if plot:
            self.plot_fit(k_values, accuracy_values, popt, dataset_name)
        
        return result, popt
    
    def analyze_scaling_with_n(self, data_dict, n_values=None, plot=True):
        
        if n_values is None:
            n_values = list(data_dict.keys())
        
        widths = []
        width_errors = []
        
        for n in n_values:
            k_vals, acc_vals = data_dict[n]
            result = self.analyze_single_dataset(k_vals, acc_vals, 
                                               dataset_name=f"n={n}", plot=False)
            if result:
                widths.append(result['width'])
                # Use bootstrap CI for error estimate
                if result['confidence_intervals']:
                    ci = result['confidence_intervals']['width']
                    width_errors.append((ci[1] - ci[0]) / 2)  # Half of CI range
                else:
                    width_errors.append(0)
        
        widths = np.array(widths)
        width_errors = np.array(width_errors)
        n_values = np.array(n_values)
        
        # Fit 1/sqrt(n) scaling
        def scaling_func(n, A):
            return A / np.sqrt(n)
        
        try:
            popt_scaling, _ = curve_fit(scaling_func, n_values, widths, 
                                      sigma=width_errors if np.any(width_errors) else None)
            A_fitted = popt_scaling[0]
        except:
            A_fitted = None
            print("Failed to fit")
        
        
        if plot:
            self.plot_scaling(n_values, widths, width_errors, A_fitted)
        
        return {
            'n_values': n_values,
            'widths': widths,
            'width_errors': width_errors,
            'scaling_coefficient': A_fitted
        }

    def getsub(self, k_values, accuracy_values, popt, title=""):
        # Plot fit
        k_fine = np.linspace(np.min(k_values), np.max(k_values), 1000)
        y_fit = self.logistic_function(k_fine, *popt)
        return k_fine, y_fit
    
    def plot_fit(self, k_values, accuracy_values, popt, title=""):
        plt.figure(figsize=(10, 6))
        
        plt.scatter(k_values, accuracy_values, alpha=0.7, label='Data', s=50)
        
        k_fine = np.linspace(np.min(k_values), np.max(k_values), 1000)
        y_fit = self.logistic_function(k_fine, *popt)
        plt.plot(k_fine, y_fit, 'r-', linewidth=2, label='Logistic fit')
        
        L, alpha, k0 = popt
        width = self.compute_transition_width(alpha)
        
        y_10 = 0.1 * L
        y_90 = 0.9 * L
        k_10 = k0 - np.log(9) / alpha
        k_90 = k0 + np.log(9) / alpha
        
        plt.axhline(y=y_10, color='gray', linestyle='--', alpha=0.5)
        plt.axhline(y=y_90, color='gray', linestyle='--', alpha=0.5)
        plt.axvline(x=k_10, color='gray', linestyle='--', alpha=0.5)
        plt.axvline(x=k_90, color='gray', linestyle='--', alpha=0.5)
        
        plt.annotate('', xy=(k_10, y_10 - 0.05), xytext=(k_90, y_10 - 0.05),
                    arrowprops=dict(arrowstyle='<->', color='red', lw=2))
        plt.text((k_10 + k_90)/2, y_10 - 0.1, f'Width ≈ {width:.3f}', 
                ha='center', va='top', color='red', fontweight='bold')
        
        plt.xlabel('k')
        plt.ylabel('Accuracy')
        plt.title(f'{title}\nR² = {self.results.get(title, {}).get("r_squared", 0):.4f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_scaling(self, n_values, widths, width_errors, A_fitted=None):
        plt.figure(figsize=(10, 6))
        
        plt.errorbar(n_values, widths, yerr=width_errors, 
                    fmt='o', capsize=5, capthick=2, label='Measured widths')
        
        if A_fitted:
            n_fine = np.linspace(np.min(n_values), np.max(n_values), 1000)
            width_theory = A_fitted / np.sqrt(n_fine)
            plt.plot(n_fine, width_theory, 'r-', linewidth=2, 
                    label=f'Fit: {A_fitted:.3f}/√n')
        
        plt.xlabel('n (system size)')
        plt.ylabel('Transition Width')
        plt.title('Phase Transition Width Scaling')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.loglog()  # Log-log scale often clearer for power laws
        plt.tight_layout()
        plt.show()
    
    def print_summary(self, dataset_name=""):
        if dataset_name in self.results:
            result = self.results[dataset_name]
            print(f"\n=== Results for {dataset_name} ===")
            print(f"Maximum accuracy (L): {result['L']:.4f}")
            print(f"Steepness (α): {result['alpha']:.4f}")
            print(f"Midpoint (k₀): {result['k0']:.4f}")
            print(f"Transition width: {result['width']:.4f}")
            print(f"R-squared: {result['r_squared']:.4f}")
            
            if result['confidence_intervals']:
                ci = result['confidence_intervals']
                print(f"\n95% Confidence Intervals:")
                print(f"Width: [{ci['width'][0]:.4f}, {ci['width'][1]:.4f}]")
                print(f"α: [{ci['alpha'][0]:.4f}, {ci['alpha'][1]:.4f}]")



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


