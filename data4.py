import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
import joblib

def data(args):
    # Read each CSV file and store them in a list
    dataframes = [pd.read_csv(file) for file in args.train_files]

    # Extract specified feature columns
    dataframes = [df.iloc[:, args.train_columns] for df in dataframes]

    # Concatenate all DataFrames row-wise
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
        df = pd.read_csv(file)
        data_file = torch.tensor(df.iloc[:args.start, args.train_columns].to_numpy().astype(np.float32))

        # Normalize data
        normalized_data = scaler.transform(data_file)

        collect_X = []
        collect_Y = []
        for i in range(len(normalized_data) - args.seq - args.seq_out + 1):
            X_block = normalized_data[i:(i + args.seq), :]
            Y_block = normalized_data[(i + args.seq):(i + args.seq + args.seq_out), :]
            collect_X.append(torch.tensor(X_block, dtype=torch.float32))
            collect_Y.append(torch.tensor(Y_block, dtype=torch.float32))

        X_all.append(torch.stack(collect_X))
        Y_all.append(torch.stack(collect_Y))

    # Concatenate lists into tensors
    X = torch.cat(X_all)
    Y = torch.cat(Y_all)

    # Save X and Y tensors
    torch.save(X, 'X_10.pt')
    torch.save(Y, 'Y_10.pt')

    # Print shapes of X and Y
    print(X.size(), Y.size())

    # Load X and Y tensors
    X_loaded = torch.load('X_10.pt')
    Y_loaded = torch.load('Y_10.pt')
    return scaler, X_loaded, Y_loaded
