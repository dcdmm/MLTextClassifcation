import torch
import torch.nn as nn
import torch.nn.functional as F


class TextRNN_att(nn.Module):
    """
    TextRNN_att模型的pytorch实现(具体任务对应修改)

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

    def __init__(self, num_class, vocab_size, embedding_size, hidden_size, num_layers, bidirectional, dropout_ratio=0.3):
        super(TextRNN_att, self).__init__()
        self.bidirectional = bidirectional
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.GRU(input_size=embedding_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          bidirectional=self.bidirectional,
                          dropout=dropout_ratio,
                          batch_first=True)  # batch_size为第一个维度

        if self.bidirectional:
            mul = 2
        else:
            mul = 1
        self.linear = nn.Linear(hidden_size * mul, num_class)
        self.dropout = nn.Dropout(dropout_ratio)

        self.W_w_b_w = nn.Linear(hidden_size * mul, hidden_size * mul)  # 可学习参数:W_w, b_w
        self.u_w = nn.Linear(hidden_size * mul, 1, bias=False)  # 可学习参数:u_w

    def forward(self, text):
        # text.shape=[batch_size, sent len]

        # embedded.shape=[batch_size, sen len, embedding_size]
        embedded = self.dropout(self.embed(text))
        # out.shape=[batch_size, sen len, hidden_size * num directions]  # 即h_{it}
        out, hidden = self.rnn(embedded)

        # *************************Attention过程*************************
        # 对每个序列的输出计算注意力权重
        # Q,K,V都是out(类似加性注意力)
        # u.shape=[batch_size, sen len, hidden_size * num directions]
        u = torch.tanh(self.W_w_b_w(out))
        # att.shape=[batch_size, sen len, 1]
        att = self.u_w(u)
        att_score = F.softmax(att, dim=1)
        # score_out.shape=[batch_size, sen len, hidden_size * num directions]  # 广播机制
        score_out = out * att_score
        # *************************Attention过程*************************

        # feat.shape=[batch_size, hidden_size * num directions]
        feat = torch.sum(score_out, dim=1)
        # result.shape=[batch_size, num_class]
        result = self.linear(feat)
        return result
