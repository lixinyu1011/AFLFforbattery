import torch
# 导入Matplotlib库
import matplotlib.pyplot as plt
from model import TransformerBiLSTMGATTCrossAttModel
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import joblib

torch.manual_seed(42)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 20
plt.rcParams['axes.labelweight'] = 'bold'  # 设置轴标签加粗
plt.rcParams['axes.labelsize'] = 22  # 设置轴说明签字号为20
plt.rcParams['axes.linewidth'] = 2  # 设置图表中轴线的宽度为2。这包括了图表的外框线以及可能的内部分隔线。
plt.rcParams['xtick.major.width'] = 2  # 设置x轴主刻度线的宽度为2。这只影响主刻度线的粗细，而非刻度标签或次刻度线。
plt.rcParams['ytick.major.width'] = 2  # 设置y轴主刻度线的宽度为2。这同样仅影响主刻度线的粗细。
plt.rcParams['lines.linewidth'] = 3  # 设置所有线图（如plot命令生成的线）的线宽为3。这会让图中的线条看起来更粗。
plt.rcParams['legend.frameon'] = False  # 设置图例背景框的显示状态。这里设置为False，意味着图例将没有背景框，使得图例更加简洁。
plt.rcParams['xtick.labelsize'] = 20  # 设置x轴刻度数字字号为10
plt.rcParams['ytick.labelsize'] = 20  # 设置y轴刻度数字字号为10

batterys = ['B0006']  # 示例电池列表

for item in batterys:
    # 加载模型
    model = torch.load(f'{item}_10best_model.pth')

    # 定义函数计算所有测试集样本的特征重要性并返回平均重要性
    def feature_importance_all_samples(model, test_loader, input_dim, output_dim):
        model.eval()
        feature_importance_sum = torch.zeros(input_dim)  # 初始化特征重要性总和

        with torch.no_grad():
            for data, label in test_loader:
                # 检查数据维度
                assert data.size(2) == input_dim, f"Data dimension mismatch: expected {input_dim}, got {data.size(2)}"

                # 前向传播获取原始输出
                original_output = model(data)

                # 针对每个特征逐个进行置零操作并重新计算输出
                for i in range(input_dim):  # 遍历特征维度
                    # 复制输入数据
                    modified_data = data.clone()
                    # 将当前特征置零
                    modified_data[:, :, i] = 0
                    # 通过模型进行前向传播
                    modified_output = model(modified_data)
                    # 计算输出变化
                    output_change = torch.abs(original_output - modified_output)
                    # 对所有输出特征计算平均变化
                    output_change_mean = torch.mean(output_change, dim=1)
                    # 计算特征重要性（输出变化越大，特征重要性越高）
                    # 累加特征重要性
                    feature_importance_sum[i] += torch.mean(output_change_mean)

        # 计算平均特征重要性
        feature_importance_avg = feature_importance_sum / len(test_loader)

        return feature_importance_avg

    batch_size = 32
    input_size = 9
    output_size = 9
    num_layers = 3
    num_heads = 8
    hidden_dim = 128
    hidden_layer_sizes = [32, 64]
    seq = 10
    seq_out = 1
    dropout = 0.1

    model = TransformerBiLSTMGATTCrossAttModel(batch_size, input_size, output_size, num_layers, num_heads, hidden_dim, hidden_layer_sizes, hidden_layer_sizes[-1], seq, seq_out, dropout)
    # 计算所有样本的特征重要性
    X_loaded = torch.load(f'X_{item}_val_feat.pt')
    Y_loaded = torch.load(f'Y_{item}_val_feat.pt')
    test_dataset = TensorDataset(X_loaded, Y_loaded)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    importance_all_samples = feature_importance_all_samples(model, test_loader, input_size, output_size)

    # 计算特征重要性的总和
    total_importance = sum(importance_all_samples)

    # 对特征重要性进行归一化处理
    normalized_importance = [imp.item() / total_importance.item() for imp in importance_all_samples]

    # 生成示例颜色列表
    colors = ['#D83B52', '#E96558', '#F37044', '#F2EE97', '#9AD68E', '#48B275', '#4598A8', '#509BC9', '#556EB4']


    # 特征名称列表
    # feature_names = ['capacity', 'xvc_time_1', 'dqcdvc_max_1', 'dqcdvc_area_1', 'tcmax_time_1', 'time_minv_vd', 'dqcdvd_min', 'dqcdvd_area', 'tdmax_time']
    feature_names = ['capacity', 'CCT', 'HDCC', 'ADCC', 'MTC', 'TD', 'LDCD', 'ADCD', 'MTD']

    # 对特征重要性进行排序
    sorted_indices = torch.argsort(torch.tensor(normalized_importance), descending=True)
    sorted_importance = [normalized_importance[i] for i in sorted_indices]
    sorted_feature_names = [feature_names[i] for i in sorted_indices]
    sorted_colors = [colors[i] for i in sorted_indices]

    # 创建棒棒糖图
    plt.figure(figsize=(13, 10), dpi=300)

    # 绘制线条
    for i, (importance, color) in enumerate(zip(sorted_importance, sorted_colors)):
        plt.plot([0, importance], [i, i], color=color, linewidth=3)  # 将linewidth设置为2

    # 绘制点的外圈（略大的圆环）
    plt.scatter(sorted_importance, range(len(sorted_importance)), color='none', edgecolor=sorted_colors, s=600, linewidth=4)  # 将s和linewidth分别设置为200和4

    # 绘制点（扩大一倍）
    plt.scatter(sorted_importance, range(len(sorted_importance)), color=sorted_colors, s=600)  # 将s设置为150

    # 绘制略大的圆环
    for i, importance in enumerate(sorted_importance):
        plt.scatter(importance, i, color='none', edgecolor=sorted_colors[i], s=1500, linewidth=2, alpha=0.6)  # 将s设置为300，linewidth设置为1，增加alpha

    # 添加数值标签
    for i, importance in enumerate(sorted_importance):
        plt.text(importance + 0.01, i, str(round(importance, 2)), ha='left', va='center', fontsize=24)

    # 设置y轴刻度和标签
    plt.yticks(range(len(sorted_importance)), sorted_feature_names)

    # 设置x轴范围
    plt.xlim(0, max(sorted_importance) * 1.1)
    # 自定义x轴显示的坐标和值
    xticks_values = [0, 0.05, 0.10, 0.15, 0.2]  # 自定义坐标
    xticks_labels = ['0', '0.05', '0.10', '0.15', '0.20']  # 自定义标签
    plt.xticks(xticks_values, xticks_labels)
    # 设置轴线和边框
    # plt.gca().spines['top'].set_visible(True)
    # plt.gca().spines['right'].set_visible(True)
    # plt.gca().spines['left'].set_linewidth(1)
    # plt.gca().spines['bottom'].set_linewidth(1)
    #
    # # 添加上轴和右轴
    # plt.gca().spines['top'].set_linewidth(1)
    # plt.gca().spines['right'].set_linewidth(1)

    # 去除网格线
    plt.grid(False)

    # 设置标签
    plt.xlabel('Feature contribution')
    # plt.ylabel('')

    # 移除图例
    # plt.legend().set_visible(False)

    # 保存图像
    # plt.savefig('lollipop_chart.pdf', bbox_inches='tight')
    plt.tight_layout()
    # 显示图像
    plt.show()
