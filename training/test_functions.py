import torch
import numpy as np
import streamlit as st
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.math_utils import (progressive_find_interval, fsolve_fixed_point, 
                             solve_volterra_fixed_point, volterra_test_function,
                             logistic_system_function)
from utils.visualization import create_test_results_visualization

def test_complex_function_enhanced(model, device, complex_function, ensemble_predict=None, ensemble_scaler=None, 
                                  test_n_points=31, scaler=None, y_scaler=None):
    """
    Test the model on a complex function to find fixed points.
    
    Args:
        model: The neural network model
        device: The device to run the model on (CPU or GPU)
        complex_function: The function to test
        ensemble_predict: Optional ML ensemble predictor
        ensemble_scaler: Scaler for the ML ensemble
        test_n_points: Number of points to sample from the function
        scaler: Scaler for input features
        y_scaler: Scaler for output labels
        
    Returns:
        final_prediction: The final predicted fixed point
        true_fixed_point: The true fixed point
        final_error: The error between the predicted and true fixed point
    """
    model.eval()
    domain = (-10, 10)
    interval = progressive_find_interval(complex_function, domain, initial_points=100, max_points=10000, factor=2)
    if interval is None:
        st.write("Could not find a valid interval for the complex function.")
        return None, None, None
    
    a_int, b_int = interval
    true_fixed_point = fsolve_fixed_point(complex_function, a_int, b_int, tol=1e-6)
    
    if true_fixed_point is not None and abs(complex_function(true_fixed_point) - true_fixed_point) < 1e-6:
        st.write("✅ Verified: The computed true fixed point satisfies f(x)=x within tolerance.")
    else:
        st.write("⚠️ Warning: The computed true fixed point does not satisfy f(x)=x within tolerance.")
    
    xs = np.linspace(a_int, b_int, test_n_points)
    features = np.array([complex_function(x) for x in xs]).reshape(1, test_n_points)
    
    # Neural network prediction
    if scaler is not None:
        features_scaled = scaler.transform(features)
    else:
        features_scaled = features
    
    features_tensor = torch.from_numpy(features_scaled.reshape(1, test_n_points, 1)).to(device).float()    
    
    with torch.no_grad():
        predicted_fixed_point = model(features_tensor).cpu().item()
        # Inverse transform the prediction if scalers are provided
        if y_scaler is not None:
            predicted_fixed_point = y_scaler.inverse_transform([[predicted_fixed_point]])[0][0]
    
    # ML ensemble prediction if available
    ml_ensemble_prediction = None
    if ensemble_predict is not None and ensemble_scaler is not None:
        try:
            features_flat = features.reshape(1, -1)
            # Apply the ensemble scaler if provided
            if ensemble_scaler is not None:
                features_flat = ensemble_scaler.transform(features_flat)
            ml_ensemble_prediction = ensemble_predict(features_flat)[0]
            st.write(f"✅ ML Ensemble prediction: {ml_ensemble_prediction:.6f}")
        except Exception as e:
            st.error(f"Error in ML ensemble prediction: {str(e)}")
            ml_ensemble_prediction = None
    
    # Calculate errors
    nn_error = abs(predicted_fixed_point - true_fixed_point) if true_fixed_point is not None else float('inf')
    ml_error = abs(ml_ensemble_prediction - true_fixed_point) if ml_ensemble_prediction is not None and true_fixed_point is not None else None
    
    # Final hybrid prediction - take the average if both models are available
    final_prediction = predicted_fixed_point
    if ml_ensemble_prediction is not None:
        final_prediction = 0.7 * predicted_fixed_point + 0.3 * ml_ensemble_prediction
        st.write(f"✅ Hybrid prediction (0.7*NN + 0.3*ML): {final_prediction:.6f}")
    
    final_error = abs(final_prediction - true_fixed_point) if true_fixed_point is not None else float('inf')
    
    # Create a two-column layout for results and visualization
    results_col, viz_col = st.columns([1, 1])
    
    # Prepare model predictions dictionary
    model_predictions = {
        "Neural Network": predicted_fixed_point
    }
    
    if ml_ensemble_prediction is not None:
        model_predictions["ML Ensemble"] = ml_ensemble_prediction
        model_predictions["Hybrid Model"] = final_prediction
    
    # Create tables and visualizations
    tables, figures = create_test_results_visualization(
        model_predictions, true_fixed_point, complex_function, interval
    )
    
    if tables and figures:
        results_df, verification_df = tables
        fig1, fig2 = figures
        
        # Display results in the first column
        with results_col:
            st.write("## Fixed Point Test Results")
            st.table(results_df)
            st.write("### Function Verification")
            st.table(verification_df)
        
        # Display visualizations in the second column
        with viz_col:
            st.write("## Fixed Point Visualization")
            st.pyplot(fig1)
            
            # Show zoomed view if available
            if fig2 is not None:
                st.write("### Zoomed View of Fixed Point")
                st.pyplot(fig2)
    
    return final_prediction, true_fixed_point, final_error

def test_volterra_equation_with_model(model, device, test_n_points=31, scaler=None, y_scaler=None):
    """
    Test the model on a Volterra integral equation solution to find fixed points.
    
    The Volterra equation: u(x) = 1 + 0.5 * ∫[0 to x] u(t) dt
    Has analytical solution: u(x) = exp(0.5*x)
    We find where exp(0.5*x) = x (the fixed point)
    """
    st.write("## Volterra Integral Equation Test")
    st.write("Testing Volterra equation: u(x) = 1 + 0.5 * ∫[0 to x] u(t) dt")
    st.write("Solution: u(x) = exp(0.5*x)")
    st.write("Finding fixed point where exp(0.5*x) = x")
    
    # First, solve the Volterra equation using fixed point method
    st.write("### Solving Volterra Equation using Fixed Point Iteration")
    x_points, numerical_sol, analytical_sol, iterations, convergence_error = solve_volterra_fixed_point()
    
    st.write(f"Volterra equation solved in {iterations} iterations with error {convergence_error:.2e}")
    solution_accuracy = np.max(np.abs(numerical_sol - analytical_sol))
    st.write(f"Solution accuracy: {solution_accuracy:.2e}")
    
    # Now use the neural network to find the fixed point of the solution function
    model.eval()
    domain = (0, 3)  # Reasonable domain for exp(0.5*x) = x
    interval = progressive_find_interval(volterra_test_function, domain, initial_points=100, max_points=5000, factor=2)
    
    if interval is None:
        st.write("Could not find a valid interval for the Volterra solution function.")
        return None, None, None
    
    a_int, b_int = interval
    true_fixed_point = fsolve_fixed_point(volterra_test_function, a_int, b_int, tol=1e-6)
    
    if true_fixed_point is not None:
        verification_error = abs(volterra_test_function(true_fixed_point) - true_fixed_point)
        if verification_error < 1e-6:
            st.write(f"✅ Found fixed point at x = {true_fixed_point:.6f}")
            st.write(f"✅ Verification: exp(0.5*{true_fixed_point:.6f}) - {true_fixed_point:.6f} = {verification_error:.2e}")
        else:
            st.write(f"⚠️ Warning: Fixed point verification failed with error {verification_error:.2e}")
    
    # Sample the function and test with neural network
    xs = np.linspace(a_int, b_int, test_n_points)
    features = np.array([volterra_test_function(x) for x in xs]).reshape(1, test_n_points)
    
    # Neural network prediction
    if scaler is not None:
        features_scaled = scaler.transform(features)
    else:
        features_scaled = features
    
    features_tensor = torch.from_numpy(features_scaled.reshape(1, test_n_points, 1)).to(device).float()
    
    with torch.no_grad():
        predicted_fixed_point = model(features_tensor).cpu().item()
        # Inverse transform the prediction if scalers are provided
        if y_scaler is not None:
            predicted_fixed_point = y_scaler.inverse_transform([[predicted_fixed_point]])[0][0]
    
    # Calculate error
    final_error = abs(predicted_fixed_point - true_fixed_point) if true_fixed_point is not None else float('inf')
    
    # Display results
    st.write("### Neural Network Fixed Point Prediction")
    model_predictions = {"Neural Network": predicted_fixed_point}
    
    # Create visualization
    tables, figures = create_test_results_visualization(
        model_predictions, true_fixed_point, volterra_test_function, interval
    )
    
    if tables and figures:
        results_df, verification_df = tables
        fig1, fig2 = figures
        
        results_col, viz_col = st.columns([1, 1])
        
        with results_col:
            st.table(results_df)
            st.write("### Function Verification")
            st.table(verification_df)
        
        with viz_col:
            st.pyplot(fig1)
            if fig2 is not None:
                st.pyplot(fig2)
    
    return predicted_fixed_point, true_fixed_point, final_error

def test_logistic_system_with_model(model, device, test_n_points=31, scaler=None, y_scaler=None):
    """
    Test the model on the discrete logistic map variant used in the paper:
    f(x) = x + 0.1 * x * (1 - x)
    Fixed points (solutions of f(x)=x) are near x=0 (unstable) and x=1 (stable).
    The implementation uses the common helper `logistic_system_function` from
    `utils.math_utils` to ensure consistency with saved configs and tests.
    """
    st.write("## Logistic Dynamical System Test")
    st.write("Testing map: f(x) = x + 0.1 * x * (1 - x)")
    st.write("Expected fixed points: x=0 (unstable), x=1 (stable)")
    
    model.eval()
    domain = (-0.5, 1.5)  # Search around expected fixed points
    
    # Find both fixed points using the shared logistic_system_function
    fixed_points_found = []

    # Search for fixed point near 0
    interval_0 = progressive_find_interval(logistic_system_function, (-0.2, 0.2))
    if interval_0:
        fp_0 = fsolve_fixed_point(logistic_system_function, interval_0[0], interval_0[1])
        if fp_0 is not None:
            fixed_points_found.append(("Near x=0", fp_0))
    
    # Search for fixed point near 1
    interval_1 = progressive_find_interval(logistic_system_function, (0.8, 1.2))
    if interval_1:
        fp_1 = fsolve_fixed_point(logistic_system_function, interval_1[0], interval_1[1])
        if fp_1 is not None:
            fixed_points_found.append(("Near x=1", fp_1))
    
    # Test neural network on the stable fixed point (near 1)
    if fixed_points_found:
        # Use the fixed point closest to 1 for neural network testing
        test_fp_name, true_fixed_point = max(fixed_points_found, key=lambda x: x[1])

        # Sample function around this fixed point
        margin = 0.3
        xs = np.linspace(max(0, true_fixed_point - margin), 
                         min(1.5, true_fixed_point + margin), test_n_points)
        features = np.array([logistic_system_function(x) for x in xs]).reshape(1, test_n_points)

        # Neural network prediction
        if scaler is not None:
            features_scaled = scaler.transform(features)
        else:
            features_scaled = features

        features_tensor = torch.from_numpy(features_scaled.reshape(1, test_n_points, 1)).to(device).float()

        with torch.no_grad():
            predicted_fixed_point = model(features_tensor).cpu().item()
            if y_scaler is not None:
                predicted_fixed_point = y_scaler.inverse_transform([[predicted_fixed_point]])[0][0]

        # Calculate error
        final_error = abs(predicted_fixed_point - true_fixed_point)

        st.write(f"### Testing Fixed Point {test_fp_name}")
        st.write(f"True fixed point: {true_fixed_point:.6f}")
        st.write(f"Neural network prediction: {predicted_fixed_point:.6f}")
        st.write(f"Error: {final_error:.6f}")

        # Show all found fixed points
        st.write("### All Fixed Points Found:")
        for name, fp in fixed_points_found:
            verification = abs(logistic_system_function(fp) - fp)
            st.write(f"{name}: x = {fp:.6f} (verification error: {verification:.2e})")

        return predicted_fixed_point, true_fixed_point, final_error
    else:
        st.write("No fixed points found for the logistic system test.")
        return None, None, None

def run_comprehensive_fixed_point_tests(model, device, test_n_points=31, scaler=None, y_scaler=None):
    """
    Run all fixed point tests: original complex function, Volterra equation, and dynamical system.
    """
    st.write("# Comprehensive Fixed Point Testing Suite")
    st.write("Testing neural network on multiple types of fixed point problems")
    
    results = {}
    
    # Test 1: Original complex function
    st.write("---")
    try:
        from utils.math_utils import complex_function
        pred1, true1, err1 = test_complex_function_enhanced(
            model, device, complex_function, test_n_points=test_n_points, 
            scaler=scaler, y_scaler=y_scaler
        )
        results["Complex Function"] = {"predicted": pred1, "true": true1, "error": err1}
    except Exception as e:
        st.error(f"Error in complex function test: {str(e)}")
    
    # Test 2: Volterra integral equation
    st.write("---")
    try:
        pred2, true2, err2 = test_volterra_equation_with_model(
            model, device, test_n_points=test_n_points, 
            scaler=scaler, y_scaler=y_scaler
        )
        results["Volterra Equation"] = {"predicted": pred2, "true": true2, "error": err2}
    except Exception as e:
        st.error(f"Error in Volterra equation test: {str(e)}")
    
    # Test 3: Logistic dynamical system
    st.write("---")
    try:
        pred3, true3, err3 = test_logistic_system_with_model(
            model, device, test_n_points=test_n_points, 
            scaler=scaler, y_scaler=y_scaler
        )
        results["Logistic System"] = {"predicted": pred3, "true": true3, "error": err3}
    except Exception as e:
        st.error(f"Error in logistic system test: {str(e)}")
    
    # Summary of results
    st.write("---")
    st.write("## Test Summary")
    
    summary_data = []
    for test_name, result in results.items():
        if result["predicted"] is not None and result["true"] is not None:
            summary_data.append({
                "Test": test_name,
                "True Fixed Point": f"{result['true']:.6f}",
                "Predicted": f"{result['predicted']:.6f}",
                "Absolute Error": f"{result['error']:.6f}",
                "Relative Error": f"{result['error']/abs(result['true'])*100:.2f}%" if result['true'] != 0 else "N/A"
            })
    
    if summary_data:
        import pandas as pd
        summary_df = pd.DataFrame(summary_data)
        st.table(summary_df)
        
        # Calculate overall performance
        errors = [r["error"] for r in results.values() if r["error"] is not None]
        if errors:
            avg_error = np.mean(errors)
            max_error = np.max(errors)
            min_error = np.min(errors)
            
            st.write("### Overall Performance")
            st.write(f"Average error: {avg_error:.6f}")
            st.write(f"Maximum error: {max_error:.6f}")
            st.write(f"Minimum error: {min_error:.6f}")
            st.write(f"Number of successful tests: {len([r for r in results.values() if r['error'] is not None])}/3")
    
    return results