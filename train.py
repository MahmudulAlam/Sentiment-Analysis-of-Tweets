import time
import torch
from network import Network
from dataset import load_dataset
from utils import mean, evaluate, save_history

batch_size = 64
vocab_size = 28996
max_seq_len = 20
features = 256
heads = 8
layers = 6
output_size = 3
drop_rate = 0.1

print('Loading data & network ...')
train_loader, test_loader = load_dataset(batch_size=batch_size, max_seq_len=max_seq_len, num_workers=0)

network = Network(vocab_size=vocab_size,
                  max_seq_len=max_seq_len,
                  features=features,
                  heads=heads,
                  n_layer=layers,
                  output_size=output_size,
                  dropout_rate=drop_rate).cuda()

# network.load_state_dict(torch.load('./weights/model.h5'))


epochs = 20
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(network.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
print('Start training ...')
history = []

for epoch in range(1, epochs + 1):
    tic = time.time()
    train_loss, train_acc = [], []

    # train
    network.train()
    for x_true, y_true in train_loader:
        x_true, y_true = x_true.cuda(), y_true.cuda()
        optimizer.zero_grad()

        # forward + loss + backward + optimize
        y_pred = network(x_true)
        loss = loss_function(y_pred, y_true)

        loss.backward()
        optimizer.step()

        # evaluate
        accuracy = evaluate(true=y_true, pred=y_pred)
        train_loss.append(loss.item())
        train_acc.append(accuracy.item())

    train_loss = mean(train_loss)
    train_acc = mean(train_acc) * 100

    # test
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

    test_acc = mean(test_acc) * 100
    toc = time.time()

    history.append(f'Epoch: {epoch}/{epochs}, train loss: {train_loss:>6.4f}, train acc: {train_acc:.2f}%, '
                   f'test acc: {test_acc:.2f}%, eta: {toc - tic:.2f}s')
    print(history[-1])

    # torch.save(network.state_dict(), f'./weights/model_{epoch}.h5')
    scheduler.step()

torch.save(network.state_dict(), f'./weights/model.h5')
save_history('./weights/history.csv', history, mode='w')
print('All Done!')
