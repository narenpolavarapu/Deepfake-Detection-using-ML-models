import matplotlib.pyplot as plt

# Define data
epochs = [1, 2, 3, 4, 5]
train_loss = [0.0786, 0.0057, 0.0060, 0.0075, 0.0073]
train_accuracy = [0.9695, 0.9990, 0.9992, 0.9985, 0.9987]
validation_loss = [0.8167, 1.8344, 1.1476, 1.3990, 0.6701]
validation_accuracy = [0.8349, 0.7177, 0.7754, 0.7793, 0.8664]

bar_width = 0.2
opacity = 0.8

plt.figure(figsize=(12, 8))

# Plot train loss
plt.bar([e - 1.5*bar_width for e in epochs], train_loss, bar_width, alpha=opacity, color='c', label='Train Loss')
# Plot train accuracy
plt.bar([e - 0.5*bar_width for e in epochs], train_accuracy, bar_width, alpha=opacity, color='m', label='Train Accuracy')
# Plot validation loss
plt.bar([e + 0.5*bar_width for e in epochs], validation_loss, bar_width, alpha=opacity, color='y', label='Validation Loss')
# Plot validation accuracy
plt.bar([e + 1.5*bar_width for e in epochs], validation_accuracy, bar_width, alpha=opacity, color='k', label='Validation Accuracy')

plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Metrics', fontsize=14)
plt.title('Metrics by Epoch', fontsize=16)
plt.xticks(epochs, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)

plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Data
methods = ['Existing Method', 'Proposed Method']
test_loss = [0.7313, 0.0017]
test_accuracy = [0.7941, 0.9998]

x = np.arange(len(methods))  # the label locations
width = 0.2  # the width of the bars (adjusted)

# Create double bar graph
fig, ax = plt.subplots(figsize=(8, 6))
bars1 = ax.bar(x - width/2, test_loss, width, label='Test Loss', color='#7fcdbb')
bars2 = ax.bar(x + width/2, test_accuracy, width, label='Test Accuracy', color='#fc9272')

# Add labels, title, and legend
ax.set_xlabel('Method')
ax.set_ylabel('Scores')
ax.set_title('Comparison of Test Loss and Test Accuracy between Methods')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend()

# Add data labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

plt.show()
