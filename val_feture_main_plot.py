import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 22
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['legend.frameon'] = False
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20

batterys = ['B0005', 'B0006', 'B0007', 'B0018']
all_zeroing_methods = []
all_labels = None

for battery in batterys:
    df = pd.read_csv(f'{battery}_feature_importance.csv')
    if all_labels is None:
        all_labels = df['Feature'].values
    zeroing_method = df['Spearman'].values
    zeroing_method = np.append(zeroing_method, zeroing_method[0])
    all_zeroing_methods.append(zeroing_method)

angles = np.linspace(0, 2 * np.pi, len(all_labels), endpoint=False).tolist() + [0]
colors = ['#F68356', '#3FB4BD', '#9ACD32', '#1E90FF']
gray = '#C0C0C0'

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

for zeroing_method, color, battery in zip(all_zeroing_methods, colors, batterys):
    ax.fill(angles, zeroing_method, color=color, alpha=0.25, label=battery)
    ax.plot(angles, zeroing_method, color=color, linewidth=2, linestyle='solid')
    ax.scatter(angles[:-1], zeroing_method[:-1], color=color, s=100, zorder=5)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(all_labels, fontsize=24, fontweight='bold', fontfamily='Times New Roman')
ax.set_rticks([0.1, 0.15, 0.2, 0.4])
ax.set_yticklabels(['', '', '', ''])
ax.set_rlabel_position(0)

for angle in angles[:-1]:
    ax.plot([angle, angle], [0, 0.25], color=gray, linestyle='--', linewidth=3, alpha=0.5)

for r in [0.05, 0.1, 0.15, 0.2, 0.25]:
    ax.plot(angles, [r] * len(angles), color=gray, linestyle='--', linewidth=3, alpha=0.5)

ax.spines['polar'].set_visible(False)
ax.yaxis.grid(False)
ax.xaxis.grid(False)

for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
    x, y = label.get_position()
    label.set_position((x, y - 0.08))

legend_elements = [Patch(facecolor=color, edgecolor=color, alpha=0.25, label=battery) for color, battery in zip(colors, batterys)]
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.8, 1.15))

plt.tight_layout()
plt.show()
