import torch
import torch.nn as nn
import torch.optim as optim
import streamlit as st
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.math_utils import r2_score_torch
from utils.visualization import create_metrics_plot, create_prediction_plot, plot_learning_rate

# Enhanced training function for better performance and monitoring
def train_model_enhanced(model, train_loader, val_loader, device, params, num_epochs=100, patience=20, 
                        swa_start=50, cycle_len=10):
    # Loss functions with weights
    mse_criterion = nn.MSELoss()
    smooth_l1_criterion = nn.SmoothL1Loss(beta=0.01)
    huber_criterion = nn.HuberLoss(delta=0.1)
    
    # Get learning rate from params
    learning_rate = params["training_params"]["learning_rate"]
    
    # Optimizer with AdamW
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=cycle_len,
        T_mult=2,
        eta_min=learning_rate * 0.01
    )
    
    # SWA setup
    swa_model = None
    swa_scheduler = None
    if swa_start < num_epochs:
        swa_model = torch.optim.swa_utils.AveragedModel(model)
        swa_scheduler = torch.optim.swa_utils.SWALR(
            optimizer, 
            swa_lr=0.0001,
            anneal_epochs=5,
            anneal_strategy='cos'
        )
    
    # Training setup
    max_grad_norm = 1.0
    best_val_loss = float('inf')
    best_r2 = -float('inf')
    patience_counter = 0
    best_model_state = None
    best_swa_model_state = None
    
    # Initialize metrics storage
    train_losses, val_losses = [], []
    train_mse_values, val_mse_values = [], []
    r2_scores = []
    
    # Create visualization layout
    st.write("## Training Progress")
    
    # Create columns for charts
    metrics_col, prediction_col = st.columns(2)
    
    with metrics_col:
        st.write("### Performance Metrics")
        metrics_chart = st.empty()
    
    with prediction_col:
        st.write("### Prediction vs True Values")
        prediction_chart = st.empty()
    
    # Status indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Learning rate chart
    st.write("### Learning Rate")
    learning_rate_chart = st.empty()

    # Training loop
    learning_rates = []
    
    for epoch in range(num_epochs):
        # Track learning rate
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # Training phase
        model.train()
        train_loss_epoch = 0.0
        train_mse_epoch = 0.0
        valid_batches = 0
        
        for batch_x, batch_y in train_loader:
            try:
                # Ensure correct input shape
                batch_x = batch_x.to(device).float()
                if len(batch_x.shape) == 2:
                    batch_x = batch_x.unsqueeze(-1)
                batch_y = batch_y.to(device).float().unsqueeze(1)
                
                optimizer.zero_grad()
                
                # Forward pass - directly pass the input to the model
                outputs = model(batch_x)
                
                # Handle outputs
                if isinstance(outputs, tuple):
                    main_output, aux1_output, aux2_output = outputs
                else:
                    main_output = outputs
                    aux1_output = aux2_output = None
                
                # Calculate losses
                mse_loss = mse_criterion(main_output, batch_y)
                smooth_l1_loss = smooth_l1_criterion(main_output, batch_y)
                huber_loss = huber_criterion(main_output, batch_y)
                
                # Combined loss
                loss = 0.5 * mse_loss + 0.3 * smooth_l1_loss + 0.2 * huber_loss
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                
                # Update metrics
                train_loss_epoch += loss.item()
                train_mse_epoch += mse_loss.item()
                valid_batches += 1
                
            except Exception as e:
                st.warning(f"Error in training batch: {str(e)}")
                continue
        
        # Calculate average training metrics
        if valid_batches > 0:
            train_loss_epoch /= valid_batches
            train_mse_epoch /= valid_batches
            train_losses.append(train_loss_epoch)
            train_mse_values.append(train_mse_epoch)
        
        # Validation phase
        model.eval()
        val_loss_epoch = 0.0
        val_mse_epoch = 0.0
        val_predictions = []
        val_targets = []
        valid_val_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                try:
                    # Ensure correct input shape
                    batch_x = batch_x.to(device).float()
                    if len(batch_x.shape) == 2:
                        batch_x = batch_x.unsqueeze(-1)
                    batch_y = batch_y.to(device).float().unsqueeze(1)
                    
                    # Forward pass - directly pass the input to the model
                    outputs = model(batch_x)
                    
                    # Handle outputs
                    if isinstance(outputs, tuple):
                        main_output, aux1_output, aux2_output = outputs
                    else:
                        main_output = outputs
                        aux1_output = aux2_output = None
                    
                    # Calculate losses
                    mse_loss = mse_criterion(main_output, batch_y)
                    smooth_l1_loss = smooth_l1_criterion(main_output, batch_y)
                    huber_loss = huber_criterion(main_output, batch_y)
                    
                    # Combined loss
                    loss = 0.5 * mse_loss + 0.3 * smooth_l1_loss + 0.2 * huber_loss
                    
                    # Update metrics
                    val_loss_epoch += loss.item()
                    val_mse_epoch += mse_loss.item()
                    val_predictions.extend(main_output.cpu().numpy())
                    val_targets.extend(batch_y.cpu().numpy())
                    valid_val_batches += 1
                    
                except Exception as e:
                    st.warning(f"Error in validation batch: {str(e)}")
                    continue
        
        # Calculate average validation metrics
        if valid_val_batches > 0:
            val_loss_epoch /= valid_val_batches
            val_mse_epoch /= valid_val_batches
            val_losses.append(val_loss_epoch)
            val_mse_values.append(val_mse_epoch)
            
            # Calculate R2 score
            r2 = r2_score_torch(torch.tensor(val_predictions), torch.tensor(val_targets))
            r2_scores.append(r2)
            
            # Update visualizations
            metrics_chart.plotly_chart(create_metrics_plot(train_losses, val_losses, train_mse_values, val_mse_values, r2_scores))
            prediction_chart.plotly_chart(create_prediction_plot(val_targets, val_predictions, r2, val_mse_epoch, epoch))
            learning_rate_chart.plotly_chart(plot_learning_rate(learning_rates))
            
            # Update SWA model if enabled
            if swa_start <= epoch and swa_model is not None and swa_scheduler is not None:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            
            # Update learning rate
            scheduler.step()
            
            # Early stopping check
            if val_loss_epoch < best_val_loss:
                best_val_loss = val_loss_epoch
                best_r2 = r2
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                if swa_model is not None:
                    best_swa_model_state = swa_model.state_dict().copy()
            else:
                patience_counter += 1
            
            # Update progress
            progress = (epoch + 1) / num_epochs
            progress_bar.progress(progress)
            
            # Update status text
            status_text.text(f"Epoch {epoch+1}/{num_epochs} - "
                           f"Train Loss: {train_loss_epoch:.4f} - "
                           f"Val Loss: {val_loss_epoch:.4f} - "
                           f"R2: {r2:.4f}")
            
            # Early stopping
            if patience_counter >= patience:
                st.write(f"Early stopping triggered after {epoch + 1} epochs")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if swa_model is not None and best_swa_model_state is not None:
            swa_model.load_state_dict(best_swa_model_state)
            return swa_model
        return model
    
    return model