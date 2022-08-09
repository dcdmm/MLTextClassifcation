import torch.nn as nn
import torch


class TextRNN_MeanMaxPool(nn.Module):
    """
    TextRNN + [MeanPool, MaxPool]模型的pytorch实现(具体任务对应修改)

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

    def __init__(self, num_class, vocab_size, embedding_size, hidden_size, num_layers, bidirectional, dropout_ratio=0.2):
        super(TextRNN_MeanMaxPool, self).__init__()
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
        self.linear1 = nn.Linear(hidden_size * mul * 2, 1024)
        self.linear2 = nn.Linear(1024, num_class)
        self.norm = nn.LayerNorm(normalized_shape=1024)
        self.dropout = nn.Dropout(dropout_ratio)
        self.relu = nn.ReLU()

    def forward(self, text):
        # text.shape=[batch_size, sent len]

        # embedded.shape=[batch_size, sen len, embedding_size]
        embedded = self.dropout(self.embed(text))
        # out.shape=[batch_size, sen len, hidden_size * num directions]  # 即h_{it}
        out, hidden = self.rnn(embedded)
        
        # feat_mean.shape=[batch_size, hidden_size * num directions]
        feat_mean = torch.mean(out, dim=1)  # 聚合操作
        # feat_max.shape=[batch_size, hidden_size * num directions]
        feat_max = torch.max(out, dim=1).values  # 聚合操作
        # result.shape=[batch_size, hidden_size * num directions * 2]
        result = torch.cat((feat_mean, feat_max), 1)
        
        # result.shape=[batch_size, 1024]
        result = self.linear1(result)
        result = self.norm(result)  # 归一化层一般防止全连接/卷积层之后
        result = self.relu(result)
        result = self.dropout(result)
        # result.shape=[batch_size, num_class]
        result = self.linear2(result)
        return result