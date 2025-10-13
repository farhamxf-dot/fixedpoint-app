import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create a traditional ML model ensemble for fixed point prediction
def create_ml_ensemble(X_train, y_train, X_val, y_val):
    st.write("Training ML model ensemble...")
    
    # Prepare a pipeline with StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train multiple models for ensemble
    models = []
    
    # Model 1: Gradient Boosting Regressor
    gb = GradientBoostingRegressor(
        n_estimators=200, 
        learning_rate=0.05,
        max_depth=4, 
        subsample=0.8,
        random_state=42
    )
    gb.fit(X_train_scaled, y_train)
    models.append(('GB', gb))
    
    # Model 2: Random Forest Regressor
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    rf.fit(X_train_scaled, y_train)
    models.append(('RF', rf))
    
    # Model 3: SVR with RBF kernel
    svr = SVR(
        kernel='rbf',
        C=10.0,
        gamma='scale',
        epsilon=0.01
    )
    svr.fit(X_train_scaled, y_train)
    models.append(('SVR', svr))
    
    # Evaluate individual models
    for name, model in models:
        y_pred = model.predict(X_val_scaled)
        r2 = r2_score(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        st.write(f"{name} - R²: {r2:.6f}, MSE: {mse:.6f}")
    
    # Create ensemble prediction function
    def ensemble_predict(X):
        X_scaled = scaler.transform(X)
        predictions = []
        
        for name, model in models:
            pred = model.predict(X_scaled)
            predictions.append(pred)
        
        # Ensemble prediction (weighted average)
        ensemble_pred = 0.5 * predictions[0] + 0.3 * predictions[1] + 0.2 * predictions[2]
        return ensemble_pred
    
    # Evaluate ensemble
    ensemble_pred = ensemble_predict(X_val)
    ensemble_r2 = r2_score(y_val, ensemble_pred)
    ensemble_mse = mean_squared_error(y_val, ensemble_pred)
    st.write(f"Ensemble - R²: {ensemble_r2:.6f}, MSE: {ensemble_mse:.6f}")
    
    # Visualize ensemble predictions
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_val, ensemble_pred, alpha=0.6, label='Predictions')
    
    # Plot y=x line
    min_val = min(min(y_val), min(ensemble_pred))
    max_val = max(max(y_val), max(ensemble_pred))
    margin = (max_val - min_val) * 0.1
    ax.plot([min_val-margin, max_val+margin], [min_val-margin, max_val+margin], 'r--', label='y=x')
    
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'ML Ensemble Predictions (R² = {ensemble_r2:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)
    
    return ensemble_predict, scaler, ensemble_r2
