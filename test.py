import numpy as np
import torch
from network import Network
from dataset import load_dataset
from utils import mean, evaluate

batch_size = 64
vocab_size = 28996
max_seq_len = 20
features = 256
heads = 8
layers = 6
output_size = 3
drop_rate = 0.1

print('Loading data & network ...')
_, test_loader = load_dataset(batch_size=batch_size, max_seq_len=max_seq_len, num_workers=0)

network = Network(vocab_size=vocab_size,
                  max_seq_len=max_seq_len,
                  features=features,
                  heads=heads,
                  n_layer=layers,
                  output_size=output_size,
                  dropout_rate=drop_rate).cuda()

network.load_state_dict(torch.load(f'./weights/model_balanced.h5'))

# test
true, pred = [], []

network.eval()
test_acc = []
with torch.no_grad():
    for x_true, y_true in test_loader:
        x_true, y_true = x_true.cuda(), y_true.cuda()

        # forward
        y_pred = network(x_true)

        # evaluate
        accuracy = evaluate(true=y_true, pred=y_pred)
        test_acc.append(accuracy.item())

        true.append(y_true.detach().cpu().numpy())
        pred.append(torch.argmax(y_pred, dim=-1).detach().cpu().numpy())

test_acc = mean(test_acc) * 100

history = f'test acc: {test_acc:.2f}%'
print(history)

true = np.concatenate(true, axis=0)
pred = np.concatenate(pred, axis=0)

print(true.shape)
print(pred.shape)

np.save('data/true.npy', true)
np.save('data/pred.npy', pred)

print('All Done!')
