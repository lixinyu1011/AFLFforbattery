import numpy as np
np.set_printoptions(threshold=np.inf)
import torch
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy.stats import sem
import os

def calculate_metrics(real_data, predicted_data):
    """
    MAE, MSE, MAPE, R2。
    """
    r2 = r2_score(real_data, predicted_data)
    mae = mean_absolute_error(real_data, predicted_data)
    mape = mean_absolute_percentage_error(real_data, predicted_data)
    mse = mean_squared_error(real_data, predicted_data)
    rmse = np.sqrt(mse)
    errors = real_data - predicted_data
    return r2, mae, mse, mape, rmse, errors


def val(args, best_model, scaler):

    Val_data_read = pd.read_csv(f'../data_processed/{args.val_data_path}_LOESS.csv')
    val_data_array = Val_data_read.iloc[:, args.train_columns].to_numpy().astype(np.float32)
    Val_data_raw = torch.tensor(val_data_array[args.start - args.seq:, :])
    Val_real_data = val_data_array[args.start:, :]
    # print(len(Val_real_data))


    Val_data_normalized_np = scaler.transform(Val_data_raw.numpy())

    # best_model.load_state_dict(torch.load(f'./{args.val_data_path}_10best_model.pth'))
    best_model.load_state_dict(torch.load(f'./model_epoch_100.pth'))
    best_model.eval()
    best_model.to(args.device)

    point_list = Val_data_normalized_np[:args.seq].tolist()
    y_pred = []
    confidence_intervals = []
    while len(point_list) < len(Val_data_normalized_np):
        x = point_list[-args.seq:] 
        x = torch.tensor([x], dtype=torch.float32).to(args.device) 
        pred = best_model(x)
        next_point = pred.detach().cpu().numpy()


        for i in range(next_point.shape[1]):
            point_list.append(next_point[0, i, :].tolist())

    y_pred = np.array(point_list[args.seq:]) 
    y_pred_pre = y_pred[:len(Val_real_data)]
    y_pred_pre = scaler.inverse_transform(y_pred_pre)




    y_pred_padded = np.full(val_data_array.shape, np.nan, dtype=np.float32)
    y_pred_padded[args.start:] = y_pred_pre


    pred_std = 0.05 
    lower_bound = (1+pred_std) * y_pred_padded
    upper_bound = (1-pred_std) * y_pred_padded

    # print(lower_bound.shape,upper_bound.shape)

    r2, mae, mse, mape, rmse, errors = calculate_metrics(y_pred_pre[:, 0], Val_real_data[:, 0])


    # plt.figure(figsize=(10, 5))
    plt.plot(range(len(y_pred_padded[:, 0])), y_pred_padded[:, 0], label='Predicted Data')
    plt.plot(range(len(val_data_array[:, 0])), val_data_array[:, 0], label='Real Data', alpha=0.5)
    plt.fill_between(range(len(y_pred_padded[:, 0])), lower_bound[:, 0], upper_bound[:, 0], color='gray', alpha=0.3,
                     label='95% confidence interval')

    # plt.title(f'{args.exp}')
    plt.xlabel('Cycles')
    plt.ylabel('Capacity (Ah)')
    plt.legend()
    plt.tight_layout()
    plt.show()


    df = pd.DataFrame({
        'Cycle': range(1, len(y_pred_padded)+1),
        'Predicted_Data': y_pred_padded[:, 0],
        'Real_Data': val_data_array[:, 0],
        'Lower_Bound': lower_bound[:, 0],
        'Upper_Bound': upper_bound[:, 0]
    })


    metrics_data = {
        'Cycle': ['MAE', 'MAPE', 'RMSE', 'R2'],
        'Predicted_Data': [mae, mape, rmse, r2],
        'Real_Data': [np.nan] * 4,
        'Lower_Bound': [np.nan] * 4,
        'Upper_Bound': [np.nan] * 4
    }
    metrics_df = pd.DataFrame(metrics_data)


    df = pd.concat([df, metrics_df], ignore_index=True)


    different_dir = './3train_1val'
    if not os.path.exists(different_dir):
        os.makedirs(different_dir)


    output_csv_path = f'./3train_1val/{args.val_data_path}_start_{args.start}.csv'
    # df.to_csv(output_csv_path, index=False)


    return Val_real_data, y_pred, y_pred_pre[:, 0]


def train_val(args, best_model, scaler):


    Val_data_read = pd.read_csv(args.val_train_data_path)
    val_data_array = torch.tensor(Val_data_read.iloc[:, args.train_columns].to_numpy().astype(np.float32))

    Val_data_raw = val_data_array[:args.seq, :]
    Val_real_data = val_data_array[args.seq:args.start, :]


    Val_data_normalized_np = scaler.transform(Val_data_raw.numpy())

    Val_data_normalized = torch.tensor(Val_data_normalized_np).float()
    Val_data_normalized = Val_data_normalized.to(args.device)

    best_model.load_state_dict(torch.load(args.best_model_pth))
    best_model.eval()

    current_data = Val_data_normalized.clone().detach().unsqueeze(0)  # current_data[1,5,20]

    predictions = []
    for _ in range(args.start-args.seq):
        with torch.no_grad():
            prediction = best_model(current_data)  # prediction[1, 20]
            predictions.append(prediction)
            current_data = torch.cat((current_data[:, 1:, :], prediction.unsqueeze(0)), 1)


    predicted_data = torch.cat(predictions, 0)

    predicted_data_cpu = predicted_data.cpu().numpy() 
    predicted_data_unnorm_np = scaler.inverse_transform(predicted_data_cpu)  


    predicted_df = pd.DataFrame(predicted_data_unnorm_np,
                                columns=[f'Feature_{i}' for i in range(predicted_data_unnorm_np.shape[1])])


    predicted_df.to_csv(f'predicted_data{args.exp}.csv', index=False)

    real_data_feature = Val_real_data[:, int(args.feature_num)]
    predicted_data_feature = predicted_data_unnorm_np[:, int(args.feature_num)]
    r2, mae, mse, mape, rmse, errors = calculate_metrics(real_data_feature, predicted_data_feature)

    val_feature = {
        'target_R2': r2,
        'target_MAE': mae,
        'target_MAPE': mape,
        'target_MSE': mse,
        'target_RMSE': rmse,
        'target_errors': errors,
    }


    return  Val_real_data, predicted_data_unnorm_np, val_feature


