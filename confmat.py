import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('darkgrid')
plt.rcParams['font.size'] = 30
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['font.family'] = 'lato'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['mathtext.default'] = 'rm'
plt.rcParams['mathtext.fontset'] = 'cm'


def mean(x):
    return sum(x) / len(x)


true_cls = np.load('data/true.npy')
pred_cls = np.load('data/pred.npy')

n_class = 3
conf_mat = np.zeros((n_class, n_class))
total = true_cls.shape[0]
count = 0

for i in range(total):
    m = int(true_cls[i])
    n = int(pred_cls[i])
    conf_mat[m, n] += 1

    if m == n:
        count += 1

np.save('data/conf_mat.npy', conf_mat)
print(conf_mat)
print(np.sum(conf_mat))
tp = conf_mat.diagonal()
fn = conf_mat.sum(1) - tp
fp = conf_mat.sum(0) - tp

accuracy = count / total * 100
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1score = (2 * precision * recall) / (precision + recall)

print(f'Accuracy: {accuracy:.2f}%')
print(f'Precision: {mean(precision) * 100:.2f}%')
print(f'Recall: {mean(recall) * 100:.2f}%')
print(f'F1 score: {mean(f1score) * 100:.2f}%')

classes = ['Neutral', 'Positive', 'Negative']
plt.figure(figsize=[16, 12])
plot = sns.heatmap(conf_mat / total, vmin=0, vmax=1, annot=True, fmt=".2g", linewidths=2, cmap='Reds',
                   xticklabels=classes, yticklabels=classes)

plt.title('')
plt.savefig('data/conf_mat.pdf', bbox_inches='tight', pad_inches=0)
plt.show()
