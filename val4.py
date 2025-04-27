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
    计算和返回所有需要的指标：MAE, MSE, MAPE, R2。
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

    # 对Val_data进行标准化处理
    Val_data_normalized_np = scaler.transform(Val_data_raw.numpy())
    # 加载最好的模型
    # best_model.load_state_dict(torch.load(f'./{args.val_data_path}_10best_model.pth'))
    best_model.load_state_dict(torch.load(f'./model_epoch_100.pth'))
    best_model.eval()
    best_model.to(args.device)

    point_list = Val_data_normalized_np[:args.seq].tolist()
    y_pred = []
    confidence_intervals = []
    while len(point_list) < len(Val_data_normalized_np):
        x = point_list[-args.seq:]  # 获取最后 args.seq 个点
        x = torch.tensor([x], dtype=torch.float32).to(args.device)  # 确保 x 是二维的，并移到设备上
        pred = best_model(x)
        next_point = pred.detach().cpu().numpy()


        # 将预测出的数据按行拆分并添加到 point_list 中
        for i in range(next_point.shape[1]):
            point_list.append(next_point[0, i, :].tolist())

    y_pred = np.array(point_list[args.seq:])  # 从序列中切割预测的部分
    y_pred_pre = y_pred[:len(Val_real_data)]
    y_pred_pre = scaler.inverse_transform(y_pred_pre)



    # 填充预测数据
    y_pred_padded = np.full(val_data_array.shape, np.nan, dtype=np.float32)
    y_pred_padded[args.start:] = y_pred_pre

    # 设置信区间是预测值的 ±5%
    pred_std = 0.05 # 5% 假设标准差
    lower_bound = (1+pred_std) * y_pred_padded
    upper_bound = (1-pred_std) * y_pred_padded

    # print(lower_bound.shape,upper_bound.shape)
    # 计算误差指标
    r2, mae, mse, mape, rmse, errors = calculate_metrics(y_pred_pre[:, 0], Val_real_data[:, 0])

    # 绘制预测数据、真实数据和置信区间
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

    # 创建DataFrame保存数据
    df = pd.DataFrame({
        'Cycle': range(1, len(y_pred_padded)+1),
        'Predicted_Data': y_pred_padded[:, 0],
        'Real_Data': val_data_array[:, 0],
        'Lower_Bound': lower_bound[:, 0],
        'Upper_Bound': upper_bound[:, 0]
    })

    # 添加误差指标到DataFrame的最后几行
    metrics_data = {
        'Cycle': ['MAE', 'MAPE', 'RMSE', 'R2'],
        'Predicted_Data': [mae, mape, rmse, r2],
        'Real_Data': [np.nan] * 4,
        'Lower_Bound': [np.nan] * 4,
        'Upper_Bound': [np.nan] * 4
    }
    metrics_df = pd.DataFrame(metrics_data)

    # 合并数据和误差指标
    df = pd.concat([df, metrics_df], ignore_index=True)
    # 训练前的准备：检查模型历史目录是否存在，如果不存在则创建

    different_dir = './3train_1val'
    if not os.path.exists(different_dir):
        os.makedirs(different_dir)

    # 保存数据到CSV
    output_csv_path = f'./3train_1val/{args.val_data_path}_start_{args.start}.csv'
    # df.to_csv(output_csv_path, index=False)


    return Val_real_data, y_pred, y_pred_pre[:, 0]


def train_val(args, best_model, scaler):

    # 加载验证数据
    Val_data_read = pd.read_csv(args.val_train_data_path)
    val_data_array = torch.tensor(Val_data_read.iloc[:, args.train_columns].to_numpy().astype(np.float32))
    # val_data_array 已经是 float32 类型的张量
    Val_data_raw = val_data_array[:args.seq, :]
    Val_real_data = val_data_array[args.seq:args.start, :]

    # 对Val_data进行标准化处理
    Val_data_normalized_np = scaler.transform(Val_data_raw.numpy())
    # 将标准化后的 NumPy 数组转换回 PyTorch 张量
    Val_data_normalized = torch.tensor(Val_data_normalized_np).float()
    Val_data_normalized = Val_data_normalized.to(args.device)
    # 加载最佳模型
    best_model.load_state_dict(torch.load(args.best_model_pth))
    best_model.eval()

    current_data = Val_data_normalized.clone().detach().unsqueeze(0)  # current_data[1,5,20]

    predictions = []
    for _ in range(args.start-args.seq):
        with torch.no_grad():
            prediction = best_model(current_data)  # prediction[1, 20]
            predictions.append(prediction)
            current_data = torch.cat((current_data[:, 1:, :], prediction.unsqueeze(0)), 1)

    # 将预测结果合并成一个tensor
    predicted_data = torch.cat(predictions, 0)

    # 对预测数据进行反标准化
    predicted_data_cpu = predicted_data.cpu().numpy()  # 先将张量移至CPU
    predicted_data_unnorm_np = scaler.inverse_transform(predicted_data_cpu)  # 然后进行反归一化操作

    # print(predicted_data_unnorm_np)
    # 将反标准化后的数据转换为 pandas DataFrame
    predicted_df = pd.DataFrame(predicted_data_unnorm_np,
                                columns=[f'Feature_{i}' for i in range(predicted_data_unnorm_np.shape[1])])

    # 将DataFrame保存为CSV文件
    predicted_df.to_csv(f'predicted_data{args.exp}.csv', index=False)

    # 计算容量特征的误差
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


