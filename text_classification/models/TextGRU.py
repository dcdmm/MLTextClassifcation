import torch
import torch.nn as nn
from abc import ABC


class TextGRU(nn.Module, ABC):
    """
    TextRNN模型的pytorch实现(具体任务对应修改)

    Parameters
    ---------
    num_class : int
       类别数
    vocab_size : int
        单词表的单词数目
    embedding_size : int
        输出词向量的维度大小
    hidden_size : int
        隐含变量的维度大小(权重矩阵W_{ih}、W_{hh}中h的大小)
    num_layers : int
        循环神经网络层数
    bidirectional : bool
        是否为设置为双向循环神经网络
    dropout_ratio : float
        元素归零的概率
    """

    def __init__(self, num_class, vocab_size, embedding_size, hidden_size, num_layers, bidirectional, dropout_ratio):
        super(TextGRU, self).__init__()
        self.bidirectional = bidirectional
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.GRU(input_size=embedding_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          bidirectional=self.bidirectional,
                          dropout=dropout_ratio, batch_first=True)
        if self.bidirectional:
            mul = 2
        else:
            mul = 1
        self.linear = nn.Linear(hidden_size * mul, num_class)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, text):
        # text.shape=[batch_size, sent len]

        # embedded.sahpe=[batch_size, sen len, embedding_size]
        embedded = self.dropout(self.embed(text))
        out_normal, hidden = self.rnn(embedded)

        if self.bidirectional:  # 双向时
            # hidden = [batch_size, hid dim * num directions]
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))  # 利用前向后后向的信息
        else:
            hidden = self.dropout(hidden[-1, :, :])
        result = self.linear(hidden)  # result.shape=[batch_size, out_size]
        return result
