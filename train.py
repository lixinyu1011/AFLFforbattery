import numpy as np
np.set_printoptions(threshold=np.inf)
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
import os
import joblib
import matplotlib.pyplot as plt
import torch.nn.functional as F

def calculate_metrics(real_data, predicted_data):
    """
    Compute and return evaluation metrics: R2, MAE, MSE, MAPE, RMSE, and raw errors.
    """
    r2 = r2_score(real_data, predicted_data)
    mae = mean_absolute_error(real_data, predicted_data)
    mape = mean_absolute_percentage_error(real_data, predicted_data)
    mse = mean_squared_error(real_data, predicted_data)
    rmse = np.sqrt(mse)
    errors = real_data - predicted_data
    return r2, mae, mse, mape, rmse, errors

def train(args, logger, scaler, model, X_loaded, Y_loaded):
    # Create model history directory if it doesn't exist
    model_history_dir = './model_history'
    if not os.path.exists(model_history_dir):
        os.makedirs(model_history_dir)

    batch_size = args.batch_size
    train_dataset = TensorDataset(X_loaded, Y_loaded)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Store training information
    info = []

    for epoch in range(args.train_epoch):
        model.train()
        train_predictions = []
        train_true_values = []
        Train_loss = 0.0

        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(args.device), Y_batch.to(args.device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, Y_batch)
            Train_loss += loss.item()
            loss.backward()
            optimizer.step()

            # Collect predictions and ground truth for training evaluation
            train_predictions.append(output.detach().cpu())
            train_true_values.append(Y_batch.detach().cpu())

        # Compute average training loss
        Train_loss /= len(train_loader)

        # Concatenate predictions and targets
        train_predictions = torch.cat(train_predictions).numpy()
        train_true_values = torch.cat(train_true_values).numpy()

        # Extract the first feature for evaluation
        train_predictions_first_feature = train_predictions[:, :, 0]
        train_true_values_first_feature = train_true_values[:, :, 0]

        # Compute training metrics
        Train_r2, Train_MAE, Train_MSE, Train_MAPE, Train_RMSE, _ = calculate_metrics(
            train_true_values_first_feature.flatten(),
            train_predictions_first_feature.flatten()
        )

        # Save training metrics for the current epoch
        epoch_info = {
            'epoch': epoch + 1,
            'Train_LOSS': Train_loss,
            'Train_R2': Train_r2,
            'Train_MAE': Train_MAE,
            'Train_MSE': Train_MSE,
            'Train_MAPE': Train_MAPE,
            'Train_RMSE': Train_RMSE
        }
        info.append(epoch_info)

        # Save model checkpoint
        torch.save(model.state_dict(), os.path.join(model_history_dir, f'model_epoch_{epoch + 1}.pth'))

        logger.info(f'Epoch: {epoch + 1}, Train_LOSS: {Train_loss:.4f}, Train_R2: {Train_r2:.3f}, '
                    f'Train_MAE: {Train_MAE:.4f}, Train_RMSE: {Train_RMSE:.4f}')

    # Save training information to CSV
    base_columns = ['Epoch', 'Train_LOSS', 'Train_R2', 'Train_MAE', 'Train_MSE', 'Train_MAPE', 'Train_RMSE']
    df = pd.DataFrame(columns=base_columns)

    for epoch_info in info:
        row = {
            'Epoch': epoch_info['epoch'],
            'Train_LOSS': epoch_info['Train_LOSS'],
            'Train_R2': epoch_info['Train_R2'],
            'Train_MAE': epoch_info['Train_MAE'],
            'Train_MSE': epoch_info['Train_MSE'],
            'Train_MAPE': epoch_info['Train_MAPE'],
            'Train_RMSE': epoch_info['Train_RMSE']
        }
        df.loc[len(df)] = row

    csv_filename = f"training_info_{args.val_data_path}_{args.start}.csv"
    df.to_csv(csv_filename, index=False)

    return info
