import csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('darkgrid')

plt.figure(figsize=[12, 7])
plt.rcParams['font.size'] = 26
plt.rcParams['font.family'] = 'lato'
plt.rcParams['axes.linewidth'] = 2.0
plt.rcParams['mathtext.default'] = 'rm'
plt.rcParams['mathtext.fontset'] = 'cm'

red = {'color': '#FF2500', 'marker': 's', 'markersize': 13}
orange = {'color': '#fc5a03', 'marker': 'p', 'markersize': 14}
yellow = {'color': '#FBBC05', 'marker': 'D', 'markersize': 13}
green = {'color': '#34A853', 'marker': 'o', 'markersize': 14}
cyan = {'color': '#00DAFF', 'marker': 'H', 'markersize': 14}
blue = {'color': '#4285F4', 'marker': '*', 'markersize': 16}
violet = {'color': '#8142ff', 'marker': 'X', 'markersize': 14}

with open('data/train.csv') as fp:
    reader = csv.reader(fp, delimiter=',')
    lines = [row for row in reader]

header = lines[0]
print(header)
sentiment, time, age, country = [], [], [], []

for line in lines[1:]:
    sentiment.append(line[3])
    time.append(line[4])
    age.append(line[5])
    country.append(line[6])


def process(smt, dat):
    key = list(set(dat))
    key.sort()
    item = {}
    for k in key:
        item[k] = {'neutral': 0, 'positive': 0, 'negative': 0}
    for s, d in zip(smt, dat):
        item[d][s] += 1
    return item


itm = process(sentiment, country)
print(len(itm))
itm = dict(sorted(itm.items(), key=lambda item: -list(item[1].values())[2]))
# keep first 10
item = {}
for i, (k, v) in enumerate(itm.items()):
    item[k] = v
    if i == 4:
        break
itm = item
print(itm)
neutral = []
positive = []
negative = []
for _, v in itm.items():
    val = list(v.values())
    neutral.append(val[0])
    positive.append(val[1])
    negative.append(val[2])

print(neutral, positive, negative)

x = np.arange(len(itm))
print(x)
width = 0.1
plt.bar(x - 0.2, neutral, width, color=yellow['color'], label='Neutral', edgecolor='k')
plt.bar(x, positive, width, color=green['color'], label='Positive', edgecolor='k')
plt.bar(x + 0.2, negative, width, color=red['color'], label='Negative', edgecolor='k')
plt.xticks(x, list(itm.keys()))
plt.legend()
plt.title('Sentiment distribution based on country.')
plt.tight_layout()
plt.subplots_adjust(left=0.07, bottom=0.08, right=0.98, top=0.92, wspace=0, hspace=0)
plt.show()
