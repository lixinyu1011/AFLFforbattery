import torch
import matplotlib.pyplot as plt
from model import TransformerBiLSTMGATTCrossAttModel
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 20
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['legend.frameon'] = False
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20

batterys = ['B0006']

for item in batterys:
    model = torch.load(f'{item}_10best_model.pth')

    def feature_importance_all_samples(model, test_loader, input_dim, output_dim):
        model.eval()
        feature_importance_sum = torch.zeros(input_dim)

        with torch.no_grad():
            for data, label in test_loader:
                assert data.size(2) == input_dim

                original_output = model(data)

                for i in range(input_dim):
                    modified_data = data.clone()
                    modified_data[:, :, i] = 0
                    modified_output = model(modified_data)
                    output_change = torch.abs(original_output - modified_output)
                    output_change_mean = torch.mean(output_change, dim=1)
                    feature_importance_sum[i] += torch.mean(output_change_mean)

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

    X_loaded = torch.load(f'X_{item}_val_feat.pt')
    Y_loaded = torch.load(f'Y_{item}_val_feat.pt')
    test_dataset = TensorDataset(X_loaded, Y_loaded)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    importance_all_samples = feature_importance_all_samples(model, test_loader, input_size, output_size)

    total_importance = sum(importance_all_samples)
    normalized_importance = [imp.item() / total_importance.item() for imp in importance_all_samples]

    colors = ['#D83B52', '#E96558', '#F37044', '#F2EE97', '#9AD68E', '#48B275', '#4598A8', '#509BC9', '#556EB4']
    feature_names = ['capacity', 'CCT', 'HDCC', 'ADCC', 'MTC', 'TD', 'LDCD', 'ADCD', 'MTD']

    sorted_indices = torch.argsort(torch.tensor(normalized_importance), descending=True)
    sorted_importance = [normalized_importance[i] for i in sorted_indices]
    sorted_feature_names = [feature_names[i] for i in sorted_indices]
    sorted_colors = [colors[i] for i in sorted_indices]

    plt.figure(figsize=(13, 10), dpi=300)

    for i, (importance, color) in enumerate(zip(sorted_importance, sorted_colors)):
        plt.plot([0, importance], [i, i], color=color, linewidth=3)

    plt.scatter(sorted_importance, range(len(sorted_importance)), color='none', edgecolor=sorted_colors, s=600, linewidth=4)
    plt.scatter(sorted_importance, range(len(sorted_importance)), color=sorted_colors, s=600)

    for i, importance in enumerate(sorted_importance):
        plt.scatter(importance, i, color='none', edgecolor=sorted_colors[i], s=1500, linewidth=2, alpha=0.6)

    for i, importance in enumerate(sorted_importance):
        plt.text(importance + 0.01, i, str(round(importance, 2)), ha='left', va='center', fontsize=24)

    plt.yticks(range(len(sorted_importance)), sorted_feature_names)
    plt.xlim(0, max(sorted_importance) * 1.1)
    xticks_values = [0, 0.05, 0.10, 0.15, 0.2]
    xticks_labels = ['0', '0.05', '0.10', '0.15', '0.20']
    plt.xticks(xticks_values, xticks_labels)
    plt.grid(False)
    plt.xlabel('Feature contribution')
    plt.tight_layout()
    plt.show()
