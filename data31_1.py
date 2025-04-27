import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
import joblib

def data(args):

    # 读取每个CSV文件并存储到列表中
    dataframes = [pd.read_csv(f'../data_processed/{file}_LOESS.csv') for file in args.train_files]
    # print(dataframes)
    # 提取特征列
    dataframes = [df.iloc[:, args.train_columns] for df in dataframes]

    # print(dataframes)

    # 按行拼接所有DataFrame
    concatenated_df = pd.concat(dataframes, ignore_index=True).to_numpy()

    data = torch.tensor(concatenated_df[:, :].astype(np.float32))
    # print(data)

    # 实例化MinMaxScaler
    scaler = MinMaxScaler()

    # 拟合数据并转换
    scaler.fit_transform(data)
    # 保存归一化对象到文件
    joblib.dump(scaler, 'scaler.joblib')

    X_all = []
    Y_all = []
    for file in args.train_files:
        df = pd.read_csv(f'../data_processed/{file}_LOESS.csv')

        if file == args.val_data_path:
            # 从指定的起点划分训练集
            data_file = torch.tensor(df.iloc[:args.start, args.train_columns].to_numpy().astype(np.float32))
        else:
            # 使用整个文件划分训练集
            data_file = torch.tensor(df.iloc[:, args.train_columns].to_numpy().astype(np.float32))

        # 标准化数据
        normalized_data = scaler.transform(data_file.numpy())

        collect_X = []
        collect_Y = []
        for i in range(len(normalized_data) - args.seq - args.seq_out + 1):
            X_block = normalized_data[i:(i + args.seq), :]
            Y_block = normalized_data[(i + args.seq):(i + args.seq + args.seq_out), :]
            collect_X.append(torch.tensor(X_block, dtype=torch.float32))  # 确保是Tensor
            collect_Y.append(torch.tensor(Y_block, dtype=torch.float32))  # 确保是Tensor

        X_all.append(torch.stack(collect_X))
        Y_all.append(torch.stack(collect_Y))

    # 将列表转换为tensor
    X = torch.cat(X_all)  # 使用cat而不是stack，因为要一个大的Tensor
    Y = torch.cat(Y_all)

    # 保存X和Y
    torch.save(X, 'X_10.pt')
    torch.save(Y, 'Y_10.pt')

    # 打印X和Y的形状
    # print(f"X shape: {X.shape}, Y shape: {Y.shape}")

    # 加载X和Y
    X_loaded = torch.load('X_10.pt')
    Y_loaded = torch.load('Y_10.pt')
    return scaler, X_loaded, Y_loaded


