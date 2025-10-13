import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import matplotlib.pyplot as plt  # Add missing matplotlib import

def create_metrics_plot(train_losses, val_losses, train_mse_values, val_mse_values, r2_scores):
    """Create a dual-axis plot showing R², MSE, and Loss metrics using Plotly."""
    if not r2_scores or not val_mse_values:
        return None
    
    # Create a combined DataFrame for all metrics
    metrics_df = pd.DataFrame({
        "R² Score": r2_scores,
        "Validation MSE": val_mse_values,
        "Training MSE": train_mse_values,
        "Validation Loss": val_losses,
        "Training Loss": train_losses
    })
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add R2 score trace
    fig.add_trace(
        go.Scatter(
            x=metrics_df.index,
            y=metrics_df['R² Score'],
            name="R²",
            line=dict(color='blue', width=2),
            mode='lines+markers',
            marker=dict(size=6)
        ),
        secondary_y=False
    )
    
    # Add MSE traces
    fig.add_trace(
        go.Scatter(
            x=metrics_df.index,
            y=metrics_df['Validation MSE'],
            name="Val MSE",
            line=dict(color='red', width=2),
            mode='lines+markers',
            marker=dict(size=6)
        ),
        secondary_y=True
    )
    
    fig.add_trace(
        go.Scatter(
            x=metrics_df.index,
            y=metrics_df['Training MSE'],
            name="Train MSE",
            line=dict(color='orange', width=2, dash='dash'),
            mode='lines+markers',
            marker=dict(size=6)
        ),
        secondary_y=True
    )
    
    # Add Loss traces
    fig.add_trace(
        go.Scatter(
            x=metrics_df.index,
            y=metrics_df['Validation Loss'],
            name="Val Loss",
            line=dict(color='green', width=2),
            mode='lines+markers',
            marker=dict(size=6)
        ),
        secondary_y=True
    )
    
    fig.add_trace(
        go.Scatter(
            x=metrics_df.index,
            y=metrics_df['Training Loss'],
            name="Train Loss",
            line=dict(color='purple', width=2, dash='dash'),
            mode='lines+markers',
            marker=dict(size=6)
        ),
        secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        title='Training Metrics: R², MSE, and Loss',
        xaxis_title='Epoch',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template='plotly_white'
    )
    
    # Update y-axes labels
    fig.update_yaxes(title_text="R² Score", secondary_y=False, range=[-0.1, 1.1])
    fig.update_yaxes(title_text="MSE / Loss", secondary_y=True)
    
    return fig

def create_prediction_plot(targets, predictions, r2_val=None, mse_val=None, epoch=None):
    """Create a scatter plot of predictions vs true values using Plotly."""
    if len(targets) == 0 or len(predictions) == 0:
        return None
    
    try:
        # Convert to numpy for plotting if they're tensors
        if isinstance(targets, torch.Tensor):
            targets_np = targets.cpu().numpy().flatten()
        else:
            targets_np = np.array(targets).flatten()
            
        if isinstance(predictions, torch.Tensor):
            preds_np = predictions.cpu().numpy().flatten()
        else:
            preds_np = np.array(predictions).flatten()
        
        # Create figure
        fig = go.Figure()
        
        # Add scatter plot
        fig.add_trace(
            go.Scatter(
                x=targets_np,
                y=preds_np,
                mode='markers',
                name='Predictions',
                marker=dict(
                    size=8,
                    color='blue',
                    opacity=0.6
                )
            )
        )
        
        # Determine plot limits
        min_val = min(targets_np.min(), preds_np.min())
        max_val = max(targets_np.max(), preds_np.max())
        margin = (max_val - min_val) * 0.1
        lims = [min_val - margin, max_val + margin]
        
        # Add y=x line
        fig.add_trace(
            go.Scatter(
                x=lims,
                y=lims,
                mode='lines',
                name='y=x',
                line=dict(
                    color='red',
                    width=2,
                    dash='dash'
                )
            )
        )
        
        # Update layout
        title = 'Prediction vs True Values'
        if epoch is not None:
            title += f' (Epoch {epoch})'
        if r2_val is not None:
            title += f'<br>R² = {r2_val:.4f}'
        
        fig.update_layout(
            title=title,
            xaxis_title='True Values',
            yaxis_title='Predicted Values',
            showlegend=True,
            template='plotly_white',
            hovermode='closest'
        )
        
        # Set equal aspect ratio
        fig.update_xaxes(range=lims)
        fig.update_yaxes(range=lims)
        
        # Add annotations for metrics if provided
        if r2_val is not None and mse_val is not None:
            fig.add_annotation(
                x=lims[1],
                y=lims[0],
                text=f"R² = {r2_val:.4f}<br>MSE = {mse_val:.6f}",
                showarrow=False,
                xanchor='right',
                yanchor='bottom',
                bgcolor='white',
                bordercolor='black',
                borderwidth=1,
                borderpad=4
            )
        
        return fig
    except Exception as e:
        st.write(f"Error plotting predictions: {str(e)}")
        return None

def plot_learning_rate(learning_rates):
    """Create a plot of learning rates over epochs using Plotly."""
    if not learning_rates:
        return None
    
    # Create figure
    fig = go.Figure()
    
    # Add learning rate trace
    fig.add_trace(
        go.Scatter(
            x=list(range(len(learning_rates))),
            y=learning_rates,
            mode='lines+markers',
            name='Learning Rate',
            line=dict(width=2),
            marker=dict(size=6)
        )
    )
    
    # Update layout
    fig.update_layout(
        title='Learning Rate Schedule',
        xaxis_title='Epoch',
        yaxis_title='Learning Rate',
        template='plotly_white',
        hovermode='x unified',
        showlegend=False
    )
    
    # Set y-axis to log scale
    fig.update_yaxes(type='log')
    
    return fig

def create_test_results_visualization(model_predictions, true_fixed_point, complex_function, interval):
    """Create visualization for test results showing fixed point predictions."""
    if interval is None:
        return None, None
    
    a_int, b_int = interval
    
    # Create a two-column layout with tables and plots
    results_data = {
        "Model": ["True Value"],
        "Fixed Point": [f"{true_fixed_point:.6f}" if true_fixed_point is not None else "Not found"],
        "Error": ["0.000000"]
    }
    
    verification_data = {
        "Model": ["True Value"],
        "f(x)": [f"{complex_function(true_fixed_point):.6f}" if true_fixed_point is not None else "N/A"],
        "|f(x) - x|": [f"{abs(complex_function(true_fixed_point) - true_fixed_point):.6f}" if true_fixed_point is not None else "N/A"]
    }
    
    # Add model predictions to the tables
    for name, prediction in model_predictions.items():
        if prediction is not None:
            results_data["Model"].append(name)
            results_data["Fixed Point"].append(f"{prediction:.6f}")
            
            if true_fixed_point is not None:
                error = abs(prediction - true_fixed_point)
                results_data["Error"].append(f"{error:.6f}")
            else:
                results_data["Error"].append("N/A")
            
            verification_data["Model"].append(name)
            verification_data["f(x)"].append(f"{complex_function(prediction):.6f}")
            verification_data["|f(x) - x|"].append(f"{abs(complex_function(prediction) - prediction):.6f}")
    
    # Convert to DataFrames
    results_df = pd.DataFrame(results_data)
    verification_df = pd.DataFrame(verification_data)
    
    # Create function plot
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    
    # Plot the function and y=x line
    x_vals = np.linspace(a_int, b_int, 200)
    f_vals = [complex_function(x) for x in x_vals]
    ax1.plot(x_vals, f_vals, label="f(x) = complex_function(x)", linewidth=3, color='#1f77b4')
    ax1.plot(x_vals, x_vals, 'k--', label="y = x", linewidth=2)
    
    # Color map for different models
    colors = ['green', 'red', 'blue', 'purple', 'orange', 'cyan']
    markers = ['o', 's', '^', '*', 'D', 'P']
    
    # Plot the fixed points
    if true_fixed_point is not None:
        ax1.scatter(true_fixed_point, complex_function(true_fixed_point), color=colors[0], s=200, 
                  label=f"True Fixed Point ({true_fixed_point:.4f})", zorder=5, edgecolor='black')
    
    # Plot model predictions
    idx = 1
    for name, prediction in model_predictions.items():
        if prediction is not None and idx < len(colors):
            ax1.scatter(prediction, complex_function(prediction), color=colors[idx], s=150, 
                      label=f"{name} ({prediction:.4f})", zorder=4, marker=markers[idx], alpha=0.8,
                      edgecolor='black')
            idx += 1
    
    # Beautify the plot
    ax1.set_xlabel("x", fontsize=14)
    ax1.set_ylabel("f(x)", fontsize=14)
    ax1.set_title("Complex Function Fixed Point Test", fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12, loc='best')
    ax1.tick_params(labelsize=12)
    ax1.set_facecolor('#f8f9fa')
    
    # Create a zoomed-in view if there's a true fixed point
    fig2 = None
    if true_fixed_point is not None:
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        
        # Determine zoom range - focus on area around true fixed point
        margin = 0.2 * abs(true_fixed_point)
        x_zoom = np.linspace(true_fixed_point - margin, true_fixed_point + margin, 200)
        f_zoom = [complex_function(x) for x in x_zoom]
        
        # Plot function and y=x in the zoomed area
        ax2.plot(x_zoom, f_zoom, label="f(x)", linewidth=3, color='#1f77b4')
        ax2.plot(x_zoom, x_zoom, 'k--', label="y = x", linewidth=2)
        
        # Plot the fixed points in the zoomed view
        if true_fixed_point is not None:
            ax2.scatter(true_fixed_point, complex_function(true_fixed_point), color=colors[0], s=200, 
                      label=f"True ({true_fixed_point:.4f})", zorder=5, edgecolor='black')
        
        # Plot model predictions
        idx = 1
        for name, prediction in model_predictions.items():
            if prediction is not None and idx < len(colors):
                ax2.scatter(prediction, complex_function(prediction), color=colors[idx], s=150, 
                          label=f"{name} ({prediction:.4f})", zorder=4, marker=markers[idx], alpha=0.8,
                          edgecolor='black')
                idx += 1
        
        # Beautify the zoomed plot
        ax2.set_xlabel("x", fontsize=14)
        ax2.set_ylabel("f(x)", fontsize=14)
        ax2.set_title("Zoomed View of Fixed Point Region", fontsize=16)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10, loc='best')
        ax2.set_facecolor('#f8f9fa')
    
    return (results_df, verification_df), (fig1, fig2)