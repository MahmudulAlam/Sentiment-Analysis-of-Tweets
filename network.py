import math
import torch
import torch.nn as nn


class WordEmbedding(nn.Module):
    def __init__(self, vocab_size: int, features: int):
        super(WordEmbedding, self).__init__()
        self.we = nn.Embedding(num_embeddings=vocab_size, embedding_dim=features)

    def forward(self, x):
        return self.we(x)


class PositionalEmbeddingLearned(nn.Module):
    def __init__(self, max_seq_len: int, features: int):
        super(PositionalEmbeddingLearned, self).__init__()
        self.positions = torch.arange(0, max_seq_len).cuda()
        self.pe = nn.Embedding(num_embeddings=max_seq_len, embedding_dim=features)

    def forward(self, x):
        return x + self.pe(self.positions)[None, 0:x.size(1)]


class PositionalEmbeddingFixed(nn.Module):
    def __init__(self, max_seq_len: int, features: int):
        super(PositionalEmbeddingFixed, self).__init__()
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, features, 2) * (-math.log(10000.0) / features))
        pe = torch.zeros(1, max_seq_len, features)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, 0:x.size(1)]


class MHAttention(nn.Module):
    def __init__(self, features: int, heads: int, dropout_rate: float):
        super(MHAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=features, num_heads=heads, dropout=dropout_rate, batch_first=True)

    def forward(self, query, key, value, mask, look_ahead_mask):
        x, w = self.mha(query=query, key=key, value=value, key_padding_mask=mask, attn_mask=look_ahead_mask)
        return x


class MLP(nn.Module):
    def __init__(self, features: int, dropout_rate: float):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(features, features * 4)
        self.linear2 = nn.Linear(features * 4, features)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class Encoder(nn.Module):
    def __init__(self, features, heads, dropout_rate):
        super(Encoder, self).__init__()
        self.ln = nn.LayerNorm(features, eps=1e-6)
        self.attn = MHAttention(features=features, heads=heads, dropout_rate=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.mlp = MLP(features=features, dropout_rate=dropout_rate)

    def forward(self, x, mask):
        lnx = self.ln(x)
        attention = self.attn(query=lnx, key=lnx, value=lnx, mask=mask, look_ahead_mask=None)
        attention = self.dropout(attention)
        x = x + attention

        lnx = self.ln(x)
        feed_forward = self.mlp(lnx)
        return x + feed_forward


class Network(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int, features: int, heads: int, n_layer: int,
                 output_size: int, dropout_rate: float = 0.1):
        super(Network, self).__init__()
        self.en_we = WordEmbedding(vocab_size=vocab_size, features=features)
        self.en_pe = PositionalEmbeddingFixed(max_seq_len=max_seq_len, features=features)

        self.drop = nn.Dropout(dropout_rate)

        self.encoder = nn.ModuleList([
            Encoder(features=features, heads=heads, dropout_rate=dropout_rate) for _ in range(n_layer)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(features, eps=1e-6),
            nn.Linear(features, output_size)
        )

    def forward(self, en_input):
        encoder_mask = torch.where(en_input > 0, False, True)

        en_input = self.en_we(en_input)
        en_input = self.en_pe(en_input)
        en_input = self.drop(en_input)

        x = en_input
        for layer in self.encoder:
            x = layer(x, encoder_mask)

        x = torch.mean(x, dim=1)
        x = self.head(x)
        return x


if __name__ == '__main__':
    encoder_inputs_ = torch.tensor([[1, 5, 6, 4, 3, 2], [1, 3, 7, 4, 2, 0]]).cuda()
    decoder_inputs_ = torch.tensor([[1, 3, 4, 6, 5], [1, 4, 7, 3, 2]]).cuda()

    network = Network(vocab_size=10,
                      max_seq_len=6,
                      features=32,
                      heads=4,
                      n_layer=2,
                      output_size=10,
                      dropout_rate=0.1).cuda()
    out = network(encoder_inputs_)
    print(out.shape)
