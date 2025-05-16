import numpy as np
np.set_printoptions(threshold=np.inf)
import logging
from datetime import datetime
import argparse

from model import TransformerBiLSTMGATTCrossAttModel
from data31_2 import data
# from ten_battery_predict_1battery import data_10
from train import train
from val4 import val, train_val
from plot import plot

import torch
import random

torch.manual_seed(42)

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    # set_seed()

    # Create argument parser
    parser = argparse.ArgumentParser(description='AFLF for LFP RUL Prediction')
    parser.add_argument('--exp', type=str, default='AFLF', help='Experiment name')

    # General settings
    parser.add_argument('--input_size', type=int, default=9, help='Input feature dimension')
    parser.add_argument('--output_size', type=int, default=9, help='Output feature dimension')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--best_model_pth', type=str, default='./best_model.pth', help='Path to best model')
    parser.add_argument('--device', type=str, default='cuda', help='Training device')

    parser.add_argument('--train_columns', type=int, nargs='*', default=[1,2,3,4,5,6,7,8,9], help='Feature column indices')

    parser.add_argument('--train_files', type=str, nargs='+', default=['B0005', 'B0007', 'B0006'], help='Training file names')
    parser.add_argument('--feature_num', type=str, default='0', help='Index of main feature')
    parser.add_argument('--train_epoch', type=int, default=1, help='Number of training epochs')

    # Validation
    parser.add_argument('--val_data_path', type=str, default='B0018', help='Validation data filename')
    parser.add_argument('--val_feature', type=str, default=['B0018'], help='Validation feature label')
    parser.add_argument('--start', type=int, default=10, help='Start index for validation cycles')

    # Model type
    parser.add_argument('--model', type=str, default='TF', help='Model type: TF (for AFLF)')

    # BiLSTM/GRU base layer settings
    parser.add_argument('--num_hidden', type=int, default=100, help='Number of hidden units in LSTM')
    parser.add_argument('--L_layer', type=int, default=2, help='Number of LSTM layers')

    # AFLF-specific settings
    parser.add_argument('--f', type=int, default=48, help='Number of AFLF feature fusion channels')
    parser.add_argument('--layer1', type=int, default=5, help='Number of layers in AFLF feature encoder block')
    parser.add_argument('--num_channels', type=int, nargs='*', default=[256, 256, 256, 256, 256, 256, 256],
                        help='Number of channels per layer in AFLF feature encoder')
    parser.add_argument('--layer2', type=int, default=1, help='Number of layers in AFLF temporal attention block')
    parser.add_argument('--head', type=int, default=8, help='Number of attention heads in AFLF')
    parser.add_argument('--kernel_size', type=int, default=3, help='Kernel size used in AFLF encoder')

    # Transformer-BiLSTM-GATT-CrossAttention model settings (core of AFLF)
    parser.add_argument('--hidden_dim', type=int, default=128, help='Transformer attention dimension')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of encoder layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--hidden_layer_sizes', type=int, nargs='*', default=[32, 64], help='BiLSTM layer sizes')
    parser.add_argument('--attention_dim', type=int, nargs='*', default=64,
                        help='Global attention dimension (default: last BiLSTM output)')

    parser.add_argument('--seq', type=int, default=10, help='Input sequence length')
    parser.add_argument('--seq_out', type=int, default=1, help='Output sequence length')

    # Parse arguments
    args = parser.parse_args()

    # Setup logging
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger = logging.getLogger('custom')
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(f"lxy_{args.exp}.log")
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.debug('This is a debug message')
    logger.info('This is an info message')
    logger.warning('This is a warning message')

    # Load training data
    scaler, X_loaded, Y_loaded = data(args)
    # mean, std, X_loaded, Y_loaded = data_10(args)

    # Initialize AFLF model
    if args.model == 'TF':
        model = TransformerBiLSTMGATTCrossAttModel(
            args.batch_size, args.input_size, args.output_size,
            args.num_layers, args.num_heads, args.hidden_dim,
            args.hidden_layer_sizes, args.hidden_layer_sizes[-1],
            args.seq, args.seq_out, args.dropout
        )
    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    model = model.to(args.device)

    # Train model using AFLF
    info_train = train(args, logger, scaler, model, X_loaded, Y_loaded)

    # Evaluate model
    Val_real_data, predicted_data_unnorm_np, val_feature = val(args, model, scaler)

    # Optional: train set evaluation
    # T_Val_real_data, T_predicted_data_unnorm_np, T_val_feature = train_val(args, model, scaler)

    # Optional: plot result
    # plot(args, info_train, predicted_data_unnorm_np, val_feature,
    #      T_predicted_data_unnorm_np, T_val_feature)

    logger.info("Program completed successfully.")
