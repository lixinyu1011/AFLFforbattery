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
import numpy as np
import random

torch.manual_seed(42)
def set_seed(seed=42):
    """设置随机种子以确保可重复性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':

    # set_seed()

    # 创建ArgumentParser对象
    parser = argparse.ArgumentParser(description='TCN-Attention for LFP RUL Prediction')
    parser.add_argument('--exp', type=str, default='TCNA', help='name')
    # 添加参数
    parser.add_argument('--input_size', type=int, default=9, help='Input size feature')
    parser.add_argument('--output_size', type=int, default=9, help='Output size feature')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of training batch size')
    # calculate set
    parser.add_argument('--best_model_pth', type=str, default='./best_model.pth', help='best model path')
    parser.add_argument('--device', type=str, default='cuda', help='training device')

    parser.add_argument('--train_columns', type=int, nargs='*', default=[1,2,3,4,5,6,7,8,9], help='1,2,5,6,7,8Indices of train_columns to remove . Default is tempreture.')

    # parser.add_argument('--train_files', type=str, nargs='+', default=['B0005', 'B0006', 'B0007', 'B0018'], help='List of file paths_match_data_10')
    parser.add_argument('--train_files', type=str, nargs='+', default=['B0005', 'B0007', 'B0006'], help='List of file paths_match_data_10')


    # parser.add_argument('--val_train_data_path', type=str, default='B0006', help='evaluate train_data path')

    parser.add_argument('--feature_num', type=str, default='0', help='seq for training data length')
    parser.add_argument('--train_epoch', type=int, default=1, help='Number of training epochs')# B5:27, B6:333, B7:450, B18:24
    # parser.add_argument('--predict_seq', type=int, default=100, help='seq for evaluation predict_length')

    parser.add_argument('--val_data_path', type=str, default='B0018', help='evaluate data path')
    parser.add_argument('--val_feature', type=str, default=['B0018'], help='evaluate data path')
    parser.add_argument('--start', type=int, default=10, help='seq for start cycle')


    parser.add_argument('--model', type=str, default='TF', help='select model TCN, LSTM, GRU,TF')
    # LSTM, GRU
    parser.add_argument('--num_hidden', type=int, default=100, help='LSTM hidden layer')
    parser.add_argument('--L_layer', type=int, default=2, help='Number of LSTM layers')
    #TCNA
    parser.add_argument('--f', type=int, default=48, help='Number of channels')
    parser.add_argument('--layer1', type=int, default=5, help='Number of layers in the first TCN block')
    parser.add_argument('--num_channels', type=int, nargs='*', default=[256, 256, 256, 256, 256, 256, 256],
                        help='Indices of train_columns to remove . Default is tempreture.')
    parser.add_argument('--layer2', type=int, default=1, help='Number of layers in Attention block')
    parser.add_argument('--head', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--kernel_size', type=int, default=3, help='Kernel size for TCN')
    # Transformer
    """
     # Transformer参数
    hidden_dim = 128  # 注意力维度
    num_layers = 2  # 编码器层数
    num_heads = 2  # 多头注意力头数
    # BiGRU 层数和维度数
    hidden_layer_sizes = [32, 64]
    # 全局注意力维度数
    attention_dim = hidden_layer_sizes[-1]  # 注意力层维度 默认为 BiGRU输出层维度

    """
    parser.add_argument('--hidden_dim', type=int, default=128, help='注意力维度')
    parser.add_argument('--num_layers', type=int, default=3, help='编码器层数')
    parser.add_argument('--num_heads', type=int, default=8, help='多头注意力头数')
    # BiLSTM 层数和维度数
    parser.add_argument('--hidden_layer_sizes', type=int, nargs='*', default=[32, 64], help='BiLSTM 层数和维度数')
    # 全局注意力维度数
    parser.add_argument('--attention_dim', type=int, nargs='*', default=64,
                        help=' attention_dim = hidden_layer_sizes[-1],注意力层维度 默认为 BiGRU输出层维度')

    parser.add_argument('--seq', type=int, default=10, help='seq for training data length')
    parser.add_argument('--seq_out', type=int, default=1, help='输出序列长度')

    # 解析参数
    args = parser.parse_args()

    # 设置日志文件
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




    # 加载训练数据
    scaler, X_loaded, Y_loaded = data(args)
    # mean, std, X_loaded, Y_loaded = data_10(args)

    # 创建模型
    if args.model == 'TF':
        model = TransformerBiLSTMGATTCrossAttModel(args.batch_size, args.input_size, args.output_size, args.num_layers, args.num_heads, args.hidden_dim,  args.hidden_layer_sizes, args.hidden_layer_sizes[-1], args.seq, args.seq_out, args.dropout)
        # model = TransformerBiGRUGATTCrossAttModel(batch_size, input_dim, output_dim, num_layers,
        #                                       num_heads, hidden_dim,
        #                                       hidden_layer_sizes, attention_dim
    else:
        raise ValueError(f"Unsupported model type: {args.model}")
    model = model.to(args.device)
    # 训练模型
    info_train = train(args, logger, scaler, model, X_loaded, Y_loaded)
    # info_train = []

    # 评价模型
    Val_real_data, predicted_data_unnorm_np, val_feature = val(args, model, scaler)

    # 以下代码未修改
    # T_Val_real_data, T_predicted_data_unnorm_np, T_val_feature = train_val(args, model, scaler)

    # 画图
    # plot(args, info_train, predicted_data_unnorm_np, val_feature,
    #      T_predicted_data_unnorm_np, T_val_feature)


    logger.info("program terminal normal")


