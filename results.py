import matplotlib.pyplot as plt
import numpy as np

# Final model metrics
models = ['MLP', 'XGBoost', 'Random Forest', 'LSTM']
accuracy = [0.9930, 0.9893, 0.9890, 0.9819]    # Accuracy in percentage
precision = [0.993, 0.996, 0.996, 0.982]
recall = [0.993, 0.982, 0.982, 0.982]
f1_score = [0.993, 0.989, 0.989, 0.982]

# Setup the figure
x = np.arange(len(models))  # the label locations
width = 0.2  # width of the bars

fig, ax = plt.subplots(figsize=(12, 7))

# Create grouped bars
rects1 = ax.bar(x - 1.5*width, accuracy, width, label='Accuracy')
rects2 = ax.bar(x - 0.5*width, precision, width, label='Precision')
rects3 = ax.bar(x + 0.5*width, recall, width, label='Recall')
rects4 = ax.bar(x + 1.5*width, f1_score, width, label='F1 Score')

# Labels and title
ax.set_ylabel('Score')
ax.set_title('Comparison of Models Across Metrics')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0.95, 1.01)  # Focused zoom to highlight small differences
ax.legend()

# Add text labels above bars
def autolabel(rects):
    """Attach a text label above each bar."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

# Apply labels
autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

# Add grid
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Tight layout
plt.tight_layout()

# Save the figure (optional)
# plt.savefig('model_metric_comparison.png', dpi=300)

# Show the plot
plt.show()
