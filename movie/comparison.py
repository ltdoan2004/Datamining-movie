import matplotlib.pyplot as plt
import numpy as np

# Model names
models = [
    'RandomForest', 'LinearReg', 'M5PTree', 'REPTree',
    'DecStump', 'kNN', 'GaussProc', 'MLP', 'AdaBoost',
    'Ensemble1', 'Ensemble2'
]

# Note on model names:
# AdaBoost: Adaptive Boosting
# RandomForest: Random Forest Regressor
# LinearReg: Linear Regression
# M5PTree: M5P Decision Tree
# REPTree: Reduced Error Pruning Tree
# DecStump: Decision Stump
# kNN: k-Nearest Neighbors
# GaussProc: Gaussian Processes
# MLP: Multilayer Perceptron
# Ensemble1: Ensemble (AdaBoost + RandomForest + GradientBoosting + ExtraTrees)
# Ensemble2: Ensemble (AdaBoost + RandomForest)
# AdaBoost: Adaptive Boosting included as part of the ensemble methods

# Evaluation metrics
mae = [0.0636, 0.2321, 0.0647, 0.0653, 0.0822, 0.2192, 0.0252, 0.2441, 0.0304, 0.0145, 0.0149]
rmse = [0.1186, 0.2808, 0.1209, 0.1216, 0.1393, 0.2920, 0.0350, 0.2977, 0.1290, 0.1204, 0.1221]
correlation = [0.9261, 0.4486, 0.9230, 0.9221, 0.8963, 0.4287, 0.9004, 0.3600, 0.8900, 0.9400, 0.9300]
runtime = [3303.37, 21.27, 456.91, 108.24, 23.45, 14110.54, 10391.40, 4277.82, 5.27, 195.38, 135.16]  # in seconds

x = np.arange(len(models))  # the label locations
width = 0.25  # the width of the bars

# Plotting MAE, RMSE, and Correlation
fig, ax1 = plt.subplots(figsize=(14, 8))
rects1 = ax1.bar(x - width, mae, width, label='MAE (lower is better)', color='#1f77b4')  # Blue
rects2 = ax1.bar(x, rmse, width, label='RMSE (lower is better)', color='#2ca02c')  # Green
rects3 = ax1.bar(x + width, correlation, width, label='Correlation (higher is better)', color='#ff7f0e')  # Orange

# Labels, title and custom x-axis tick labels
ax1.set_xlabel('Models')
ax1.set_ylabel('Error Metrics / Correlation')
ax1.set_title('Model Evaluation Metrics Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=45, ha='right')

# Adding legends
ax1.legend(loc='upper left')

# Adding data labels to bars
def add_labels(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2.,
            1.01 * height,
            f'{height:.2f}',
            ha='center', va='bottom'
        )

add_labels(rects1, ax1)
add_labels(rects2, ax1)
add_labels(rects3, ax1)

fig.tight_layout()
plt.savefig('model_evaluation_metrics.png')
plt.show()

# Plotting Runtime separately using a horizontal bar chart
fig, ax2 = plt.subplots(figsize=(14, 8))
rects4 = ax2.barh(x, runtime, height=0.5, label='Runtime (s) (lower is better)', color='#d62728')  # Red

# Labels, title and custom x-axis tick labels
ax2.set_ylabel('Models')
ax2.set_xlabel('Runtime (seconds)')
ax2.set_title('Model Runtime Comparison')
ax2.set_yticks(x)
ax2.set_yticklabels(models, ha='right')

# Adding legend
ax2.legend(loc='upper right')

# Adding data labels to bars
for rect in rects4:
    width = rect.get_width()
    ax2.text(
        width + 50,  # Positioning text slightly to the right of the bar
        rect.get_y() + rect.get_height() / 2.,
        f'{width:.2f}',
        ha='center', va='center'
    )

fig.tight_layout()
plt.savefig('model_runtime_comparison.png')
plt.show()