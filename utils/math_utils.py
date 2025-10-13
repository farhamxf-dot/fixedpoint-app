import numpy as np
import torch
import math
from scipy.optimize import fsolve
import scipy.integrate as integrate

# Global Variable for bifurcation override
OVERRIDE_BIFURCATION_PARAMS = None  # If set, should be a tuple (a1, a2, a3, b1, b2, b3)

# Custom Sigmoid Function
def sigmoid(x):
    # Stable sigmoid with clipping to prevent overflow
    return torch.clamp(1 / (1 + torch.exp(-torch.clamp(x, -88, 88))), 1e-7, 1-1e-7)

def sigmoid_da(x, s=1.0):
    # Derivative of sigmoid with scale parameter
    sig = sigmoid(s * x)
    return s * sig * (1 - sig)

def r2_score_func(y_true, y_pred):
    """Calculate R² score"""
    SS_res = np.sum(np.square(y_true - y_pred))
    SS_tot = np.sum(np.square(y_true - np.mean(y_true)))
    r2 = 1 - (SS_res / (SS_tot + 1e-8))  # Add small epsilon to prevent division by zero
    return np.clip(r2, -1, 1)  # Clip to valid range

def r2_score_torch(y_true, y_pred):
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    if y_true.size(0) < 2:
        return None
    ss_res = torch.sum((y_true - y_pred)**2)
    ss_tot = torch.sum((y_true - torch.mean(y_true))**2)
    return 1 - ss_res/ss_tot if ss_tot != 0 else None

# Fixed Point Solver using fsolve
def fsolve_fixed_point(f, a, b, tol=1e-8, maxfev=10000):
    def g(x):
        return f(x) - x
    x0 = (a + b) / 2.0
    fixed_pt, info, ier, mesg = fsolve(g, x0, xtol=tol, maxfev=maxfev, full_output=True)
    if ier == 1:
        return fixed_pt[0]
    else:
        return None

def find_interval(f, domain, num_points=1000):
    xs = np.linspace(domain[0], domain[1], num_points)
    g_vals = [f(x) - x for x in xs]
    for i in range(len(g_vals)-1):
        if g_vals[i] * g_vals[i+1] < 0:
            return xs[i], xs[i+1]
    return None

def progressive_find_interval(f, domain, initial_points=100, max_points=10000, factor=2):
    n_points = initial_points
    while n_points <= max_points:
        interval = find_interval(f, domain, num_points=n_points)
        if interval is not None:
            return interval
        n_points *= factor
    return None

# Define complex_function for testing
def complex_function(x):
    return 2*(x**3-1)/3

# NEW: Volterra integral equation solver using fixed point method
def volterra_kernel(x, t):
    """Simple kernel function K(x,t) = 0.5 for the Volterra equation."""
    return 0.5

def volterra_source_function(x):
    """Source function f(x) = 1 for the Volterra equation."""
    return 1.0

def volterra_operator(u_func, x_points):
    """
    Volterra integral operator T: u -> Tu
    where (Tu)(x) = f(x) + ∫[0 to x] K(x,t) * u(t) dt
    """
    result = np.zeros_like(x_points)
    
    for i, x in enumerate(x_points):
        if x == 0:
            result[i] = volterra_source_function(x)  # f(0) = 1
        else:
            # Compute integral ∫[0 to x] K(x,t) * u(t) dt using trapezoidal rule
            t_points = x_points[:i+1]  # Points from 0 to x
            integrand = np.array([volterra_kernel(x, t) * u_func[j] for j, t in enumerate(t_points)])
            integral = np.trapz(integrand, t_points)
            result[i] = volterra_source_function(x) + integral
    
    return result

def solve_volterra_fixed_point(x_max=2.0, n_points=50, max_iterations=100, tolerance=1e-8):
    """
    Solve Volterra integral equation u(x) = 1 + 0.5 * ∫[0 to x] u(t) dt
    using fixed point iteration.
    
    Analytical solution: u(x) = exp(0.5*x)
    """
    # Create discretization
    x_points = np.linspace(0, x_max, n_points)
    
    # Initial guess: u^(0)(x) = 1 (constant function)
    u_current = np.ones_like(x_points)
    
    for iteration in range(max_iterations):
        # Apply Volterra operator: u^(k+1) = T(u^(k))
        u_new = volterra_operator(u_current, x_points)
        
        # Check convergence
        error = np.max(np.abs(u_new - u_current))
        if error < tolerance:
            break
        
        u_current = u_new.copy()
    else:
        print(f"Warning: Maximum iterations ({max_iterations}) reached. Final error: {error:.2e}")
    
    # Analytical solution for comparison
    analytical_solution = np.exp(0.5 * x_points)
    
    return x_points, u_current, analytical_solution, iteration + 1, error

# NEW: Function to create test data from Volterra equation for neural network training
def volterra_test_function(x):
    """
    Test function based on solution of Volterra integral equation.
    This creates a smooth, well-behaved function for fixed point testing.
    
    The function u(x) = exp(0.5*x) has fixed point where u(x) = x
    This occurs when exp(0.5*x) = x
    """
    return np.exp(0.5 * x)

def logistic_system_function(x):
    """
    Logistic dynamical system: dx/dt = x(1-x)
    Fixed points: x=0 (unstable), x=1 (stable)
    """
    return x + 0.1 * x * (1 - x)