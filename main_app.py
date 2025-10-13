import os
import torch
import streamlit as st
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
# Import modules
from models import CombinedModel, EnhancedCombinedModel
from data import generate_data_csv, FixedPointDataset
from training import train_model_enhanced, create_ml_ensemble, test_complex_function_enhanced
from utils import complex_function, OVERRIDE_BIFURCATION_PARAMS
from training.test_functions import (test_volterra_equation_with_model, 
                                   test_logistic_system_with_model, 
                                   run_comprehensive_fixed_point_tests)

def get_model_parameters():
    """Get model parameters from UI controls."""
    st.sidebar.header("Model Parameters")
    
    # LSTM Parameters
    st.sidebar.subheader("LSTM Architecture")
    lstm_hidden_size = st.sidebar.slider("LSTM Hidden Size", 32, 256, 128, 16, key="lstm_hidden_size")
    lstm_num_layers = st.sidebar.slider("LSTM Layers", 1, 4, 2, 1, key="lstm_num_layers")
    lstm_dropout = st.sidebar.slider("LSTM Dropout", 0.0, 0.5, 0.2, 0.1, key="lstm_dropout")

    # Attention Parameters
    st.sidebar.subheader("Attention Layer")
    attention_heads = st.sidebar.slider("Attention Heads", 1, 8, 4, 1, key="attention_heads")
    attention_dropout = st.sidebar.slider("Attention Dropout", 0.0, 0.5, 0.1, 0.1, key="attention_dropout")

    # Chaotic Oscillator Parameters
    st.sidebar.subheader("Chaotic Oscillator")
    oscillator_units = st.sidebar.slider("Oscillator Units", 16, 64, 41, 1, key="oscillator_units")
    oscillator_count = st.sidebar.slider("Number of Oscillators", 4, 16, 12, 1, key="oscillator_count")
    k = st.sidebar.slider("Bifurcation Parameter (k)", 0.1, 4.0, 1.0, 0.1, key="bifurcation_k")
    s = st.sidebar.slider("Scale Parameter (s)", 0.1, 2.0, 1.0, 0.1, key="scale_s")

    # Convolutional Branch Parameters
    st.sidebar.subheader("Convolutional Architecture")
    conv_channels = [
        st.sidebar.slider(f"Conv Layer {i+1} Channels", 16, 256, [32, 64, 128][i], 16, key=f"conv_layer_{i}")
        for i in range(3)
    ]
    kernel_size = st.sidebar.slider("Kernel Size", 3, 7, 3, 2, key="kernel_size")

    # Transformer Parameters (for EnhancedCombinedModel)
    st.sidebar.subheader("Transformer Architecture")
    transformer_dim = st.sidebar.slider("Embedding Dimension", 32, 256, 128, 16, key="transformer_dim")
    transformer_heads = st.sidebar.slider("Transformer Heads", 1, 8, 4, 1, key="transformer_heads")
    transformer_blocks = st.sidebar.slider("Transformer Blocks", 1, 4, 2, 1, key="transformer_blocks")

    # Dense Layer Parameters
    st.sidebar.subheader("Dense Layers")
    dense_hidden_size = st.sidebar.slider("Dense Hidden Size", 64, 512, 256, 32, key="dense_hidden_size")
    dense_dropout = st.sidebar.slider("Dense Dropout", 0.0, 0.5, 0.2, 0.1, key="dense_dropout")

    # Training Parameters
    st.sidebar.subheader("Training Parameters")
    learning_rate = st.sidebar.slider("Learning Rate", 1e-5, 1e-2, 1e-3, step=1e-5, format="%.5f", key="learning_rate")
    batch_size = st.sidebar.slider("Batch Size", 16, 256, 64, 16, key="batch_size")
    epochs = st.sidebar.slider("Epochs", 10, 200, 100, 10, key="epochs")

    return {
        "lstm_params": {
            "hidden_size": lstm_hidden_size,
            "num_layers": lstm_num_layers,
            "dropout": lstm_dropout
        },
        "attention_params": {
            "heads": attention_heads,
            "dropout": attention_dropout
        },
        "oscillator_params": {
            "units": oscillator_units,
            "count": oscillator_count,
            "k": k,
            "s": s
        },
        "conv_params": {
            "channels": conv_channels,
            "kernel_size": kernel_size
        },
        "transformer_params": {
            "dim": transformer_dim,
            "heads": transformer_heads,
            "blocks": transformer_blocks
        },
        "dense_params": {
            "hidden_size": dense_hidden_size,
            "dropout": dense_dropout
        },
        "training_params": {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs
        }
    }

def main():
    """Main application function."""
    st.title("Advanced Fixed Point Predictor App")
    st.write("Configure training parameters using the sidebar and run the training process.")
    # Create model parameter widgets once and keep the returned params
    params = get_model_parameters()
    
    st.sidebar.header("Training Configuration")
    n_rows = st.sidebar.selectbox("Number of rows for data generation", 
                                  [500, 1000, 2000, 3000, 5000, 10000], index=2,
                                  key="n_rows")
    n_points = st.sidebar.selectbox("Number of points per function", 
                                    [11, 21, 31, 41, 51, 61, 81], index=2,
                                    key="n_points")
    num_epochs = st.sidebar.selectbox("Number of training epochs", 
                                      [50, 100, 200, 300, 500, 700], index=2,
                                      key="num_epochs_select")
    poly_degree = st.sidebar.slider("Polynomial Degree (1 = no transform)", 1, 3, 1,
                                  key="poly_degree")
    
    k_slider = st.sidebar.slider("Oscillator k", 0.001, 0.5, 0.05, key="k_slider_main")
    s_slider = st.sidebar.slider("Sigmoid scale s", 0.1, 5.0, 1.0, key="s_slider_main")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Model Architecture",
        ["Original Combined Model", "Enhanced Model", "Ensemble (Neural Net + ML)"]
    )
    
    # Data augmentation option
    use_augmentation = st.sidebar.checkbox("Use Data Augmentation", value=True)
    
    # Advanced training options
    st.sidebar.subheader("Advanced Training Options")
    early_stop_patience = st.sidebar.slider("Early Stopping Patience", 5, 50, 20, key="early_stop_patience")
    
    # SWA options
    use_swa = st.sidebar.checkbox("Use Stochastic Weight Averaging", value=True, key="use_swa")
    if use_swa:
        swa_start = st.sidebar.slider("SWA Start Epoch", 20, 100, 50, key="swa_start")
    else:
        swa_start = num_epochs + 1  # Disable SWA
    
    override_bif = st.sidebar.checkbox("Override bifurcation parameters", value=False)
    if override_bif:
        bif_a1 = st.sidebar.slider("a1", -10.0, 10.0, 0.0, key="bif_a1")
        bif_a2 = st.sidebar.slider("a2", -10.0, 10.0, 5.0, key="bif_a2")
        bif_a3 = st.sidebar.slider("a3", -10.0, 10.0, 5.0, key="bif_a3")
        bif_b1 = st.sidebar.slider("b1", -10.0, 10.0, 0.0, key="bif_b1")
        bif_b2 = st.sidebar.slider("b2", -10.0, 10.0, -1.0, key="bif_b2")
        bif_b3 = st.sidebar.slider("b3", -10.0, 10.0, 1.0, key="bif_b3")
        global OVERRIDE_BIFURCATION_PARAMS
        OVERRIDE_BIFURCATION_PARAMS = (bif_a1, bif_a2, bif_a3, bif_b1, bif_b2, bif_b3)
    else:
        OVERRIDE_BIFURCATION_PARAMS = None

    model_option = st.sidebar.selectbox("Model Option", ["Train new model", "Load saved model"])
    
    # Add scaler selection
    scaler_type = st.sidebar.selectbox(
        "Select Scaler Type",
        ["StandardScaler", "MinMaxScaler"],
        help="StandardScaler normalizes to mean=0, std=1. MinMaxScaler scales to a fixed range [0, 1]"
    )
    
    # NEW: Testing options
    st.sidebar.subheader("Testing Options")
    test_mode = st.sidebar.selectbox(
        "Select Test Mode",
        ["Original Complex Function", "Comprehensive Testing Suite", 
         "Volterra Equation Only", "Dynamical System Only"],
        help="Choose which type of fixed point problem to test the model on"
    )
    
    if model_option == "Train new model":
        # Display current model parameters in the main area
        st.header("Current Model Parameters")
        
        # Display parameters in an organized way
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("LSTM Parameters")
            st.write(f"Hidden Size: {params['lstm_params']['hidden_size']}")
            st.write(f"Layers: {params['lstm_params']['num_layers']}")
            st.write(f"Dropout: {params['lstm_params']['dropout']}")
            
            st.subheader("Attention Parameters")
            st.write(f"Heads: {params['attention_params']['heads']}")
            st.write(f"Dropout: {params['attention_params']['dropout']}")
            
            st.subheader("Chaotic Oscillator")
            st.write(f"Units: {params['oscillator_params']['units']}")
            st.write(f"Count: {params['oscillator_params']['count']}")
            st.write(f"k: {params['oscillator_params']['k']}")
            st.write(f"s: {params['oscillator_params']['s']}")

        with col2:
            st.subheader("Convolutional Parameters")
            st.write(f"Channels: {params['conv_params']['channels']}")
            st.write(f"Kernel Size: {params['conv_params']['kernel_size']}")
            
            st.subheader("Transformer Parameters")
            st.write(f"Dimension: {params['transformer_params']['dim']}")
            st.write(f"Heads: {params['transformer_params']['heads']}")
            st.write(f"Blocks: {params['transformer_params']['blocks']}")
            
            st.subheader("Training Parameters")
            st.write(f"Learning Rate: {params['training_params']['learning_rate']}")
            st.write(f"Batch Size: {params['training_params']['batch_size']}")
            st.write(f"Epochs: {params['training_params']['epochs']}")
        
        st.markdown("---")
        
        run_button = st.sidebar.button("Run Training")
        if run_button:
            # Generate or load data
            csv_file = "functions_fixed_points_progressive.csv"
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                if df.shape[1] - 1 != n_points:
                    st.write("CSV file does not match selected n_points. Regenerating data...")
                    generate_data_csv(n_rows=n_rows, n_points=n_points, output_file=csv_file)
                else:
                    st.write(f"Using existing data file: {csv_file}")
            else:
                generate_data_csv(n_rows=n_rows, n_points=n_points, output_file=csv_file)
            
            # Create dataset and data loaders
            dataset = FixedPointDataset(csv_file, poly_degree=poly_degree, normalize=True, 
                                      augment=use_augmentation, scaler_type=scaler_type)
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            
            # Select device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            st.write(f"Using device: {device}")
            
            # Initialize ML ensemble if needed
            ensemble_predict = None
            ensemble_scaler = None
            ensemble_r2 = None
            
            # model parameters were already retrieved and displayed above
            
            # Initialize model based on selected type
            model = None
            
            if model_type == "Original Combined Model":
                model = CombinedModel(
                    n_points=dataset.X.shape[1],
                    k=params["oscillator_params"]["k"],
                    s=params["oscillator_params"]["s"],
                    lstm_hidden_size=params["lstm_params"]["hidden_size"],
                    lstm_num_layers=params["lstm_params"]["num_layers"],
                    lstm_dropout=params["lstm_params"]["dropout"],
                    attention_heads=params["attention_params"]["heads"],
                    attention_dropout=params["attention_params"]["dropout"],
                    conv_channels=params["conv_params"]["channels"],
                    kernel_size=params["conv_params"]["kernel_size"],
                    dense_hidden_size=params["dense_params"]["hidden_size"],
                    dense_dropout=params["dense_params"]["dropout"],
                    oscillator_units=params["oscillator_params"]["units"],
                    oscillator_count=params["oscillator_params"]["count"]
                ).to(device)
                st.write("Using original CombinedModel architecture")
            elif model_type == "Enhanced Model":
                model = EnhancedCombinedModel(
                    n_points=dataset.X.shape[1],
                    k=params["oscillator_params"]["k"],
                    s=params["oscillator_params"]["s"],
                    lstm_hidden_size=params["lstm_params"]["hidden_size"],
                    lstm_num_layers=params["lstm_params"]["num_layers"],
                    lstm_dropout=params["lstm_params"]["dropout"],
                    attention_heads=params["attention_params"]["heads"],
                    attention_dropout=params["attention_params"]["dropout"],
                    conv_channels=params["conv_params"]["channels"],
                    kernel_size=params["conv_params"]["kernel_size"],
                    transformer_dim=params["transformer_params"]["dim"],
                    transformer_heads=params["transformer_params"]["heads"],
                    transformer_blocks=params["transformer_params"]["blocks"],
                    dense_hidden_size=params["dense_params"]["hidden_size"],
                    dense_dropout=params["dense_params"]["dropout"],
                    oscillator_units=params["oscillator_params"]["units"],
                    oscillator_count=params["oscillator_params"]["count"]
                ).to(device)
                st.write("Using EnhancedCombinedModel architecture with improved components")
            elif model_type == "Ensemble (Neural Net + ML)":
                model = EnhancedCombinedModel(
                    n_points=dataset.X.shape[1],
                    k=params["oscillator_params"]["k"],
                    s=params["oscillator_params"]["s"],
                    lstm_hidden_size=params["lstm_params"]["hidden_size"],
                    lstm_num_layers=params["lstm_params"]["num_layers"],
                    lstm_dropout=params["lstm_params"]["dropout"],
                    attention_heads=params["attention_params"]["heads"],
                    attention_dropout=params["attention_params"]["dropout"],
                    conv_channels=params["conv_params"]["channels"],
                    kernel_size=params["conv_params"]["kernel_size"],
                    transformer_dim=params["transformer_params"]["dim"],
                    transformer_heads=params["transformer_params"]["heads"],
                    transformer_blocks=params["transformer_params"]["blocks"],
                    dense_hidden_size=params["dense_params"]["hidden_size"],
                    dense_dropout=params["dense_params"]["dropout"],
                    oscillator_units=params["oscillator_params"]["units"],
                    oscillator_count=params["oscillator_params"]["count"]
                ).to(device)
                st.write("Using hybrid ensemble approach (Neural Network + ML)")
                
                # Train the ML ensemble component
                st.write("Training ML ensemble component...")
                # Prepare data for ML models
                X_flat = dataset.X.reshape(dataset.X.shape[0], -1)
                y = dataset.y
                
                # Split into train/val
                X_train = X_flat[:train_size]
                y_train = y[:train_size]
                X_val = X_flat[train_size:]
                y_val = y[train_size:]
                
                # Train the ML ensemble
                ensemble_predict, ensemble_scaler, ensemble_r2 = create_ml_ensemble(X_train, y_train, X_val, y_val)
            
            if model is None:
                st.error("Failed to initialize model. Please check model type selection.")
                return
            
            st.write("Starting neural network training...")
            model = train_model_enhanced(
                model, 
                train_loader, 
                val_loader, 
                device, 
                params=params,
                num_epochs=num_epochs,
                patience=early_stop_patience,
                swa_start=swa_start
            )
            
            # Save model (store state_dict + model config so we can reload exact architecture)
            model_save_path = f"fixed_point_{model_type.lower().replace(' ', '_').replace('(', '').replace(')', '')}.pth"
            checkpoint = {
                'state_dict': model.state_dict(),
                'config': params,
                'model_type': model_type
            }
            torch.save(checkpoint, model_save_path)
            st.write(f"Model saved to {model_save_path}")
            
            # Enhanced testing section based on selected test mode
            st.write("## Testing Trained Model")
            
            if test_mode == "Original Complex Function":
                # Original test
                test_complex_function_enhanced(
                    model, 
                    device, 
                    complex_function,
                    ensemble_predict=ensemble_predict,
                    ensemble_scaler=ensemble_scaler,
                    test_n_points=n_points, 
                    scaler=dataset.scaler, 
                    y_scaler=dataset.y_scaler
                )
                
            elif test_mode == "Comprehensive Testing Suite":
                # New comprehensive testing
                run_comprehensive_fixed_point_tests(
                    model,
                    device,
                    test_n_points=n_points,
                    scaler=dataset.scaler,
                    y_scaler=dataset.y_scaler
                )
                
            elif test_mode == "Volterra Equation Only":
                # Volterra equation test only
                test_volterra_equation_with_model(
                    model,
                    device,
                    test_n_points=n_points,
                    scaler=dataset.scaler,
                    y_scaler=dataset.y_scaler
                )
                
            elif test_mode == "Dynamical System Only":
                # Dynamical system test only
                test_logistic_system_with_model(
                    model,
                    device,
                    test_n_points=n_points,
                    scaler=dataset.scaler,
                    y_scaler=dataset.y_scaler
                )
    else:
        # Load saved model option
        saved_models = [f for f in os.listdir('.') if f.endswith('.pth')]
        if not saved_models:
            st.write("No saved model available. Please train a model first.")
        else:
            selected_model = st.sidebar.selectbox("Select saved model", saved_models)
            load_button = st.sidebar.button("Load Model")
            if load_button:
                csv_file = "functions_fixed_points_progressive.csv"
                if os.path.exists(csv_file):
                    df = pd.read_csv(csv_file)
                    if df.shape[1] - 1 != n_points:
                        st.write("CSV file does not match selected n_points. Regenerating data...")
                        generate_data_csv(n_rows=n_rows, n_points=n_points, output_file=csv_file)
                    else:
                        st.write(f"Using existing data file: {csv_file}")
                else:
                    generate_data_csv(n_rows=n_rows, n_points=n_points, output_file=csv_file)
                
                dataset = FixedPointDataset(csv_file, poly_degree=poly_degree, normalize=True, scaler_type=scaler_type)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                # Try to load checkpoint first to get saved config if available
                try:
                    chk = torch.load(selected_model, map_location=device)
                    if isinstance(chk, dict) and 'state_dict' in chk and 'config' in chk:
                        # Use saved configuration
                        cfg = chk['config']
                        saved_model_type = chk.get('model_type', '')
                        
                        # Show the user what configuration we're loading
                        st.write("Loading model with saved configuration:")
                        st.write(f"- Model type: {saved_model_type}")
                        st.write(f"- Oscillator units: {cfg['oscillator_params']['units']}")
                        st.write(f"- Oscillator count: {cfg['oscillator_params']['count']}")
                        st.write(f"- Dense hidden size: {cfg['dense_params']['hidden_size']}")
                        
                        # Create model with exact saved configuration
                        if 'Enhanced' in saved_model_type:
                            model = EnhancedCombinedModel(
                                n_points=dataset.X.shape[1],
                                k=cfg['oscillator_params']['k'],
                                s=cfg['oscillator_params']['s'],
                                lstm_hidden_size=cfg['lstm_params']['hidden_size'],
                                lstm_num_layers=cfg['lstm_params']['num_layers'],
                                lstm_dropout=cfg['lstm_params']['dropout'],
                                attention_heads=cfg['attention_params']['heads'],
                                attention_dropout=cfg['attention_params']['dropout'],
                                conv_channels=cfg['conv_params']['channels'],
                                kernel_size=cfg['conv_params']['kernel_size'],
                                transformer_dim=cfg['transformer_params'].get('dim', 128),
                                transformer_heads=cfg['transformer_params'].get('heads', 4),
                                transformer_blocks=cfg['transformer_params'].get('blocks', 2),
                                dense_hidden_size=cfg['dense_params']['hidden_size'],
                                dense_dropout=cfg['dense_params']['dropout'],
                                oscillator_units=cfg['oscillator_params']['units'],
                                oscillator_count=cfg['oscillator_params']['count']
                            ).to(device)
                            st.write("Using EnhancedCombinedModel architecture with saved config")
                        else:
                            model = CombinedModel(
                                n_points=dataset.X.shape[1],
                                k=cfg['oscillator_params']['k'],
                                s=cfg['oscillator_params']['s'],
                                lstm_hidden_size=cfg['lstm_params']['hidden_size'],
                                lstm_num_layers=cfg['lstm_params']['num_layers'],
                                lstm_dropout=cfg['lstm_params']['dropout'],
                                attention_heads=cfg['attention_params']['heads'],
                                attention_dropout=cfg['attention_params']['dropout'],
                                conv_channels=cfg['conv_params']['channels'],
                                kernel_size=cfg['conv_params']['kernel_size'],
                                dense_hidden_size=cfg['dense_params']['hidden_size'],
                                dense_dropout=cfg['dense_params']['dropout'],
                                oscillator_units=cfg['oscillator_params']['units'],
                                oscillator_count=cfg['oscillator_params']['count']
                            ).to(device)
                            st.write("Using CombinedModel architecture with saved config")
                    else:
                        # No saved config, fall back to default initialization
                        if "enhanced" in selected_model.lower():
                            model = EnhancedCombinedModel(n_points=dataset.X.shape[1], k=k_slider, s=s_slider).to(device)
                            st.write("Using EnhancedCombinedModel architecture with current settings")
                        else:
                            model = CombinedModel(n_points=dataset.X.shape[1], k=k_slider, s=s_slider).to(device)
                            st.write("Using CombinedModel architecture with current settings")
                        st.warning("Loading model with current UI settings - sizes may not match saved model")
                        
                    # Force initialization with a dummy input batch
                    dummy_input = torch.zeros(2, dataset.X.shape[1], 1).to(device)
                    _ = model(dummy_input)
                    
                except Exception as e:
                    st.error(f"Error loading checkpoint: {str(e)}")
                    st.warning("Falling back to default model configuration")
                    # Fall back to default initialization
                    if "enhanced" in selected_model.lower():
                        model = EnhancedCombinedModel(n_points=dataset.X.shape[1], k=k_slider, s=s_slider).to(device)
                        st.write("Using EnhancedCombinedModel architecture with current settings")
                    else:
                        model = CombinedModel(n_points=dataset.X.shape[1], k=k_slider, s=s_slider).to(device)
                        st.write("Using CombinedModel architecture with current settings")
                    
                    # Force initialization with a dummy input batch
                    dummy_input = torch.zeros(2, dataset.X.shape[1], 1).to(device)
                    _ = model(dummy_input)
                
                # Now try to load the weights
                try:
                    # Load the state dict or checkpoint
                    if isinstance(chk, dict) and 'state_dict' in chk:
                        state_dict = chk['state_dict']
                    else:
                        state_dict = chk
                        
                    # Try to load the state dict with strict=False to allow partial loads
                    load_result = model.load_state_dict(state_dict, strict=False)
                    st.write("Loaded checkpoint:", selected_model)
                    
                    # Report any loading issues
                    if load_result.missing_keys:
                        st.warning("Some model parameters were not found in checkpoint:")
                        st.write(load_result.missing_keys)
                    if load_result.unexpected_keys:
                        st.warning("Checkpoint contained extra parameters not used by model:")
                        st.write(load_result.unexpected_keys)
                        
                    if load_result.missing_keys or load_result.unexpected_keys:
                        st.info("ℹ️ Parameter mismatches detected. This usually means the model was trained with different architecture settings. Consider:")
                        st.write("1. Training a new model with current settings")
                        st.write("2. Adjusting model parameters to match the saved checkpoint")
                        st.write("3. Using the 'Show saved model config' option to see original settings")
                    
                    # Enhanced testing for loaded model based on selected test mode
                    st.write("## Testing Loaded Model")
                    
                    if test_mode == "Original Complex Function":
                        test_complex_function_enhanced(
                            model, 
                            device, 
                            complex_function,
                            test_n_points=n_points, 
                            scaler=dataset.scaler, 
                            y_scaler=dataset.y_scaler
                        )
                        
                    elif test_mode == "Comprehensive Testing Suite":
                        run_comprehensive_fixed_point_tests(
                            model,
                            device,
                            test_n_points=n_points,
                            scaler=dataset.scaler,
                            y_scaler=dataset.y_scaler
                        )
                        
                    elif test_mode == "Volterra Equation Only":
                        test_volterra_equation_with_model(
                            model,
                            device,
                            test_n_points=n_points,
                            scaler=dataset.scaler,
                            y_scaler=dataset.y_scaler
                        )
                        
                    elif test_mode == "Dynamical System Only":
                        test_logistic_system_with_model(
                            model,
                            device,
                            test_n_points=n_points,
                            scaler=dataset.scaler,
                            y_scaler=dataset.y_scaler
                        )
                        
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
                    st.write("Please check if the model file is compatible with the selected architecture.")
                    # If checkpoint contains config, offer to display it
                    try:
                        chk_preview = torch.load(selected_model, map_location='cpu')
                        if isinstance(chk_preview, dict) and 'config' in chk_preview:
                            if st.checkbox("Show saved model config (from checkpoint)"):
                                st.json(chk_preview['config'])
                    except Exception:
                        pass

if __name__ == '__main__':
    main()