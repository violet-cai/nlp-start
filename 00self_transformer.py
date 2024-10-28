import math
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def clones(module, n):
    return nn.ModuleList([deepcopy(module) for _ in range(n)])


def sequence_mask(size):
    # 对序列进行mask
    shape = (1, size, size)
    mask = np.triu(np.ones(shape), k=1).astype('uint8')
    return (torch.from_numpy(mask) == 0).to(device)


class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(WordEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):  # max_len最长长度
        super(PositionEmbedding, self).__init__()
        self.d_model = d_model
        super(PositionEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # 初始化pe max_len * d_model
        position = torch.arange(0, max_len).unsqueeze(1)  # 构建pos，为句子的长度 max_len * 1
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # pe 1 * max_len * d_model
        self.register_buffer('pe', pe)  # 为pe注册缓存区，在模型保存和加载时被保存，不会被优化器更新

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # d_k 每个头的特征大小

        self.key = nn.Linear(d_model, self.d_k * num_heads, bias=False)
        self.query = nn.Linear(d_model, self.d_k * num_heads, bias=False)
        self.value = nn.Linear(d_model, self.d_k * num_heads, bias=False)
        self.fc_out = nn.Linear(num_heads * self.d_k, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        # query:batch_size, seq_len, d_model
        batch_size = query.size(0)
        query = self.query(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # transpose(1,2)作用是交换维度1和维度2的维度 query变为(batch_size,num_heads,seq_len,d_k)
        key = self.key(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.value(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
            # 将scores在对应mask的位置替换成很大的负数，经过softmax后值趋近于0
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        context = torch.matmul(attention, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        # context重新变为(batch_size,seq_len,d_model)
        return self.fc_out(context)


class FeedForward(nn.Module):
    # feedforward层
    def __init__(self, d_model, d_ff, dropout=0.1):  # d_ff 中间维度
        super(FeedForward, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout1(self.relu(self.fc1(self.layer_norm(x))))
        return self.dropout2(self.fc2(x))


class SublayerConnection(nn.Module):
    # 残差连接，同时进行层归一化
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feedforward = FeedForward(d_model, d_ff, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, query, key, value, mask=None):
        attention = self.attention(query, key, value, mask)
        x = self.dropout1(self.layer_norm1(attention + query))
        forward = self.feedforward(x)
        return self.dropout2(self.layer_norm2(forward + x))


class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.linear = nn.Linear(d_model, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.sublayer = SublayerConnection(d_model, num_heads, d_ff, dropout)

    def forward(self, x, mask):
        x = self.sublayer(x, x, x, mask)
        return x


class Encoder(nn.Module):
    def __init__(self, n, encoder_layer):
        super(Encoder, self).__init__()
        self.layers = clones(encoder_layer, n)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.sublayer_self = SublayerConnection(d_model, num_heads, d_ff, dropout)
        self.sublayer_cross = SublayerConnection(d_model, num_heads, d_ff, dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.sublayer_self(x, x, x, tgt_mask)
        x = self.sublayer_cross(x, memory, memory, src_mask)
        return x


class Decoder(nn.Module):
    def __init__(self, n, decoder_layer):
        super(Decoder, self).__init__()
        self.layers = clones(decoder_layer, n)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return x


class SelfTransformer(nn.Module):
    def __init__(self, vocab, d_model, d_ff, num_heads, dropout=0.1):
        super(SelfTransformer, self).__init__()
        self.vocab = vocab
        self.d_model = d_model
        self.d_ff = d_ff
        self.word_embed = WordEmbedding(len(vocab), d_model)
        self.position_embed = PositionEmbedding(d_model)
        self.encoder = Encoder(6, EncoderLayer(d_model, num_heads, d_ff, dropout))
        self.decoder = Decoder(6, DecoderLayer(d_model, num_heads, d_ff, dropout))
        self.generator = Generator(d_model, len(vocab))

    def forward(self, src, tgt, src_mask, tgt_mask):
        src_embed = self.word_embed(src) * math.sqrt(self.d_model)
        src_embed = src_embed + self.position_embed(src_embed)
        tgt_embed = self.word_embed(tgt) * math.sqrt(self.d_model)
        tgt_embed = tgt_embed + self.position_embed(tgt_embed)
        memory = self.encoder(src_embed, src_mask)
        output = self.decoder(tgt_embed, memory, src_mask, tgt_mask)
        return self.generator(output)


if __name__ == '__main__':
    vocab = ['<pad>', 'hello', 'world', 'goodbye']
    model = SelfTransformer(vocab, d_model=512, d_ff=2048, num_heads=8).to(device)

    # 创建一些示例输入
    src = torch.tensor([[1, 2], [2, 1]]).to(device)  # (batch_size=2, source_seq_len=2)
    tgt = torch.tensor([[1, 2], [2, 1]]).to(device)  # (batch_size=2, target_seq_len=2)
    src_mask = sequence_mask(2)  # (1, source_seq_len, source_seq_len)
    tgt_mask = sequence_mask(2)  # (1, target_seq_len, target_seq_len)

    output = model(src, tgt, src_mask, tgt_mask)
    print(output.shape)  # (batch_size, seq_len, vocab_size)
