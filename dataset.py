import csv
import torch
from transformers import AutoTokenizer


class Dataset(torch.utils.data.Dataset):
    def __init__(self, split, max_seq_len, balanced=False):
        super(Dataset, self).__init__()

        with open(f'data/{split}.csv') as fp:
            reader = csv.reader(fp, delimiter=',')
            lines = [row for row in reader]

        data = {'text': [], 'sentiment': [], 'time': [], 'age': [], 'country': []}
        neutral, positive, negative = 0, 0, 0

        if split == 'train':
            for line in lines[1:]:
                if balanced:
                    if line[3] == 'neutral':
                        if neutral == 6000:
                            continue
                        else:
                            neutral += 1

                    if line[3] == 'positive':
                        if positive == 6000:
                            continue
                        else:
                            positive += 1

                    if line[3] == 'negative':
                        if negative == 6000:
                            continue
                        else:
                            negative += 1

                data['text'].append(line[1])
                data['sentiment'].append(line[3])
                data['time'].append(line[4])
                data['age'].append(line[5])
                data['country'].append(line[6])
        else:
            for line in lines[1:]:
                if balanced:
                    if line[2] == 'neutral':
                        if neutral == 1000:
                            continue
                        else:
                            neutral += 1

                    if line[2] == 'positive':
                        if positive == 1000:
                            continue
                        else:
                            positive += 1

                    if line[2] == 'negative':
                        if negative == 1000:
                            continue
                        else:
                            negative += 1

                if line[2] == '':
                    continue

                # print(neutral, positive, negative)
                data['text'].append(line[1])
                data['sentiment'].append(line[2])
                data['time'].append(line[3])
                data['age'].append(line[4])
                data['country'].append(line[5])

        self.data = data
        print(len(self.data['text']))
        self.max_seq_len = max_seq_len
        self.label = {'neutral': 0, 'positive': 1, 'negative': 2}
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased', model_max_length=max_seq_len)

    def __len__(self):
        return len(self.data['text'])

    def __getitem__(self, idx):
        x = self.data['text'][idx]
        y = self.data['sentiment'][idx]
        # todo: better to use tokenizer inside a collator. all sentences can be tokenized parallely.
        tokens = self.tokenizer(x, padding='max_length', max_length=self.max_seq_len, truncation=True,
                                return_tensors='pt', return_attention_mask=False, return_token_type_ids=False)
        x = torch.squeeze(tokens['input_ids'])
        y = torch.tensor(self.label.get(y, 0))
        return x, y


def load_dataset(batch_size, max_seq_len, num_workers=0):
    train_set = Dataset(split='train', max_seq_len=max_seq_len)
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers,
                                               drop_last=False,
                                               pin_memory=True,
                                               collate_fn=None)

    test_set = Dataset(split='test', max_seq_len=max_seq_len)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=num_workers,
                                              drop_last=False,
                                              pin_memory=True,
                                              collate_fn=None)

    return train_loader, test_loader


if __name__ == '__main__':
    train, test = load_dataset(batch_size=32, max_seq_len=20)

    total = 0
    for x_, y_ in test:
        total += x_.shape[0]

    print(total)
