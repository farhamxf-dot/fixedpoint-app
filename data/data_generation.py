import random
import math
import csv
import numpy as np
import streamlit as st
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.math_utils import find_interval, fsolve_fixed_point, progressive_find_interval

def generate_random_function():
    """Generate a random function of various types with domain."""
    func_type = random.choice([1, 2, 3, 4, 5])
    if func_type == 1:
        a = random.uniform(-10, 10)
        b = random.uniform(0.1, 10)
        c = random.uniform(-10, 10)
        d = random.uniform(0.1, 10)
        e = random.uniform(-10, 10)
        def trig_function(x): 
            return a * math.cos(b * x) + c * math.sin(d * x) + e
        domain = (-10, 10)
        return trig_function, domain
    elif func_type == 2:
        a = random.uniform(0.1, 10)
        b = random.uniform(-10, -0.1)
        c = random.uniform(-10, 10)
        def exp_function(x): 
            return -a * math.exp(b * x) + c
        domain = (-10, 10)
        return exp_function, domain
    elif func_type == 3:
        a = random.uniform(-10, -0.1)
        b_param = random.uniform(1, 10)
        c = random.uniform(-10, 10)
        def log_function(x): 
            return a * math.log(b_param + x) + c
        domain = (0.1, 10)
        return log_function, domain
    elif func_type == 4:
        a = random.uniform(-10, 10)
        b_param = random.uniform(-10, 10)
        k = random.uniform(0.1, 10)
        def rational_function(x): 
            return ((a * x + b_param) / (x**2 + 1)) * math.exp(-k * x**2)
        domain = (-10, 10)
        return rational_function, domain
    elif func_type == 5:
        a = random.uniform(-10, 10)
        b = random.uniform(0.1, 10)
        c = random.uniform(-10, 10)
        def tanh_function(x): 
            return a * math.tanh(b * x) + c
        domain = (-10, 10)
        return tanh_function, domain

def generate_row_progressive(n_points=11, tol=1e-6):
    """Generate a row of data by finding a function with a fixed point."""
    for _ in range(200):
        result = generate_random_function()
        if result is None:
            continue
        f, domain = result
        interval = progressive_find_interval(f, domain, initial_points=100, max_points=10000, factor=2)
        if interval is not None:
            a_int, b_int = interval
            fixed_pt = fsolve_fixed_point(f, a_int, b_int, tol=tol)
            if fixed_pt is not None:
                xs = np.linspace(a_int, b_int, n_points)
                features = [f(x) for x in xs]
                return features, fixed_pt
    return None, None

def generate_data_csv(n_rows, n_points, output_file='functions_fixed_points_progressive.csv'):
    """Generate a CSV file with functions and their fixed points."""
    st.write("Starting data generation...")
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = [f'f_x{i}' for i in range(n_points)] + ['fixed_point']
        writer.writerow(header)
        count = 0
        while count < n_rows:
            features, fixed_pt = generate_row_progressive(n_points=n_points)
            if features is not None:
                writer.writerow(features + [fixed_pt])
                count += 1
                if count % 100 == 0:
                    st.write(f"Generated {count} rows...")
    st.write("Finished generating CSV file:", output_file)
