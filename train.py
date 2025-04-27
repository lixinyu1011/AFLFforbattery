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
    计算和返回所有需要的指标：MAE, MSE, MAPE, R2。
    """
    r2 = r2_score(real_data, predicted_data)
    mae = mean_absolute_error(real_data, predicted_data)
    mape = mean_absolute_percentage_error(real_data, predicted_data)
    mse = mean_squared_error(real_data, predicted_data)
    rmse = np.sqrt(mse)
    errors = real_data - predicted_data
    return r2, mae, mse, mape, rmse, errors


def train(args, logger, scaler, model, X_loaded, Y_loaded):

    # 训练前的准备：检查模型历史目录是否存在，如果不存在则创建
    model_history_dir = './model_history'
    if not os.path.exists(model_history_dir):
        os.makedirs(model_history_dir)

    batch_size = args.batch_size
    train_dataset = TensorDataset(X_loaded, Y_loaded)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)  # 可以根据需要调整batch_size和shuffle

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 初始化用于存储最佳模型的变量
    best_rmse = float('inf')  # 使用无穷作为起始值
    best_epoch = None
    best_Train_LOSS = None
    best_Test_MAE_ls = float('inf')
    # 存储训练信息
    info = []

    for epoch in range(args.train_epoch):
        model.train()
        train_predictions = []
        train_true_values = []
        Train_loss = 0.0  # 初始化损失累积变量

        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(args.device), Y_batch.to(args.device)
            optimizer.zero_grad()
            output = model(X_batch)  # output=[batchsize, 5, 8]
            # print(output.shape, Y_batch.shape)
            loss = criterion(output, Y_batch)
            Train_loss += loss.item()  # 累计损失
            loss.backward()
            optimizer.step()

            # 收集预测值和真实值用于训练评估
            train_predictions.append(output.detach().cpu())
            train_true_values.append(Y_batch.detach().cpu())

        # 计算训练过程中的平均损失值
        Train_loss /= len(train_loader)

        # 将收集到的预测值和真实值转换为适合评估函数的格式
        train_predictions = torch.cat(train_predictions).numpy()
        train_true_values = torch.cat(train_true_values).numpy()

        # 提取第一个特征
        train_predictions_first_feature = train_predictions[:, :, 0]  # 可以用args.feature_num代替
        train_true_values_first_feature = train_true_values[:, :, 0]

        # 计算训练集上的评估指标
        Train_r2, Train_MAE, Train_MSE, Train_MAPE, Train_RMSE, _ = calculate_metrics(train_true_values_first_feature.flatten(), train_predictions_first_feature.flatten())


        # print(Train_MAE)

        # 验证集评估
        model.eval()
        with torch.no_grad():
            Val_data_read = pd.read_csv(f'../data_processed/{args.val_data_path}_LOESS.csv')
            val_data_array = Val_data_read.iloc[:, args.train_columns].to_numpy().astype(np.float32)
            Val_data_raw = torch.tensor(val_data_array[args.start - args.seq:, :])
            Val_real_data = val_data_array[args.start:, :]

            # 对Val_data进行标准化处理
            Val_data_normalized_np = scaler.transform(Val_data_raw.numpy())
            point_list = Val_data_normalized_np[:args.seq].tolist()
            y_pred = []
            while len(point_list) < len(Val_data_normalized_np):
                x = point_list[-args.seq:]  # 获取最后 args.seq 个点
                x = torch.tensor([x], dtype=torch.float32).to(args.device)  # 确保 x 是二维的，并移到设备上
                pred = model(x)
                next_point = pred.detach().cpu().numpy()

                # 将预测出的数据按行拆分并添加到 point_list 中
                for i in range(next_point.shape[1]):
                    point_list.append(next_point[0, i, :].tolist())

            y_pred = np.array(point_list[args.seq:])  # 从序列中切割预测的部分
            y_pred_pre = scaler.inverse_transform(y_pred)
            y_pred_pre = y_pred_pre[:len(Val_real_data[:, 0])]

            test_r2, test_mae, test_mse, test_mape, test_rmse, test_errors = calculate_metrics(y_pred_pre[:, 0], Val_real_data[:, 0])

            # 保存训练的特征每一代的信息
            epoch_info = {
                'epoch': epoch + 1,
                'Train_LOSS': Train_loss,
                'Train_R2': Train_r2,
                'Train_MAE': Train_MAE,
                'Train_MSE': Train_MSE,
                'Train_MAPE': Train_MAPE,
                'Train_RMSE': Train_RMSE,
                'Test_R2': test_r2,
                'Test_MAE': test_mae,
                'Test_MSE': test_mse,
                'Test_MAPE': test_mape,
                'Test_RMSE': test_rmse,
                'Test_ERROR': test_errors
            }
            info.append(epoch_info)


            if test_mae < best_Test_MAE_ls:
                best_epoch = epoch + 1  # 记录达到最佳 RMSE 的代数
                best_Train_LOSS = Train_loss
                best_Test_r2 = test_r2
                best_Test_MAE = test_mae
                best_Test_MSE = test_mse
                best_Test_MAPE = test_mape
                best_Test_RMSE = test_rmse
                torch.save(model.state_dict(), f'./{args.val_data_path}_{args.start}best_model.pth')
                best_Test_MAE_ls = test_mae

        # 在每个epoch结束后保存模型参数
        torch.save(model.state_dict(), os.path.join(model_history_dir, f'model_epoch_{epoch+1}.pth'))

        logger.info(f'Epoch: {epoch + 1}, Train_LOSS: {Train_loss:.4f}, Train_R2: {Train_r2:.3f},Train_MAE: {Train_MAE:4f},'
                    f'Test_MAE: {test_mae:.4f},Test_R2: {test_r2:.3f}, '
                    f'best_epoch: {best_epoch}, best_Test_MAE: {best_Test_MAE:.4f}, best_Test_MAPE: {best_Test_MAPE:.4f}, ')

    info.append({'best_epoch': best_epoch, 'best_Train_LOSS': best_Train_LOSS, 'best_Test_r2': best_Test_r2, 'best_Test_MAE': best_Test_MAE, 'best_Test_MSE': best_Test_MSE, 'best_Test_MAPE': best_Test_MAPE, 'best_Test_RMSE': best_Test_RMSE})

    #写入info信息
    base_columns = ['Epoch', 'Train_LOSS', 'Train_R2', 'Train_MAE', 'Train_MSE', 'Train_MAPE', 'Train_RMSE',
               'Test_R2', 'Test_MAE', 'Test_MSE', 'Test_MAPE', 'Test_RMSE', 'Test_ERROR']

    df = pd.DataFrame(columns=base_columns)

    for epoch_info in info[:-1]:  # 排除最后一个添加的总结信息
        row = {
            'Epoch': epoch_info['epoch'],
            'Train_LOSS': epoch_info['Train_LOSS'],
            'Train_R2': epoch_info['Train_R2'],
            'Train_MAE': epoch_info['Train_MAE'],
            'Train_MSE': epoch_info['Train_MSE'],
            'Train_MAPE': epoch_info['Train_MAPE'],
            'Train_RMSE': epoch_info['Train_RMSE'],
            'Test_R2': epoch_info['Test_R2'],
            'Test_MAE': epoch_info['Test_MAE'],
            'Test_MSE': epoch_info['Test_MSE'],
            'Test_MAPE': epoch_info['Test_MAPE'],
            'Test_RMSE': epoch_info['Test_RMSE'],
            'Test_ERROR': epoch_info['Test_ERROR']
        }

        df.loc[len(df)] = row

    csv_filename = f"training_info_{args.val_data_path}_{args.start}.csv"
    df.to_csv(csv_filename, index=False)

    logger.info(info[-1:])

    return info

