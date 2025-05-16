import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
import joblib

def data(args):
    # Read each CSV file and store them in a list
    dataframes = [pd.read_csv(f'../data_processed/{file}_LOESS.csv') for file in args.train_files]

    # Extract specified feature columns
    dataframes = [df.iloc[:, args.train_columns] for df in dataframes]

    # Concatenate all dataframes row-wise
    concatenated_df = pd.concat(dataframes, ignore_index=True).to_numpy()
    data = torch.tensor(concatenated_df[:, :].astype(np.float32))

    # Initialize MinMaxScaler
    scaler = MinMaxScaler()

    # Fit the scaler and transform the data
    scaler.fit_transform(data)

    # Save the fitted scaler to a file
    joblib.dump(scaler, 'scaler.joblib')

    X_all = []
    Y_all = []
    for file in args.train_files:
        df = pd.read_csv(f'../data_processed/{file}_LOESS.csv')

        if file == args.val_data_path:
            # Use data before the start index for training
            data_file = torch.tensor(df.iloc[:args.start, args.train_columns].to_numpy().astype(np.float32))
        else:
            # Use the entire file for training
            data_file = torch.tensor(df.iloc[:, args.train_columns].to_numpy().astype(np.float32))

        # Normalize the data
        normalized_data = scaler.transform(data_file.numpy())

        collect_X = []
        collect_Y = []
        for i in range(len(normalized_data) - args.seq - args.seq_out + 1):
            X_block = normalized_data[i:(i + args.seq), :]
            Y_block = normalized_data[(i + args.seq):(i + args.seq + args.seq_out), :]
            collect_X.append(torch.tensor(X_block, dtype=torch.float32))
            collect_Y.append(torch.tensor(Y_block, dtype=torch.float32))

        X_all.append(torch.stack(collect_X))
        Y_all.append(torch.stack(collect_Y))

    # Concatenate into final tensors
    X = torch.cat(X_all)
    Y = torch.cat(Y_all)

    # Save training X and Y
    torch.save(X, 'X_10.pt')
    torch.save(Y, 'Y_10.pt')

    # Load training X and Y
    X_loaded = torch.load('X_10.pt')
    Y_loaded = torch.load('Y_10.pt')

    # Process validation feature set
    X_all_val_feat = []
    Y_all_val_feat = []
    for file in args.val_feature:
        df = pd.read_csv(f'../data_processed/{file}_LOESS.csv')

        # Use data from the start index onward for validation
        data_file = torch.tensor(df.iloc[args.start:, args.train_columns].to_numpy().astype(np.float32))

        # Normalize the data
        normalized_data = scaler.transform(data_file.numpy())

        collect_X_val_feat = []
        collect_Y_val_feat = []
        for i in range(len(normalized_data) - args.seq - args.seq_out + 1):
            X_block = normalized_data[i:(i + args.seq), :]
            Y_block = normalized_data[(i + args.seq):(i + args.seq + args.seq_out), :]
            collect_X_val_feat.append(torch.tensor(X_block, dtype=torch.float32))
            collect_Y_val_feat.append(torch.tensor(Y_block, dtype=torch.float32))

        X_all_val_feat.append(torch.stack(collect_X_val_feat))
        Y_all_val_feat.append(torch.stack(collect_Y_val_feat))

    # Concatenate validation feature tensors
    X = torch.cat(X_all_val_feat)
    Y = torch.cat(Y_all_val_feat)

    # Save validation X and Y
    torch.save(X, f'X_{args.val_data_path}_val_feat.pt')
    torch.save(Y, f'Y_{args.val_data_path}_val_feat.pt')

    return scaler, X_loaded, Y_loaded
