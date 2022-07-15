import torch.nn as nn
import torch


class BertRNN(nn.Module):
    """
    BertRNN(类似TextRNN)模型的pytorch实现(具体任务对应修改)(transformer实现训练过程)

    Parameters
    ---------
    num_class : int
        类别数
    pretrained_model : torch.nn.modules
        预训练模型(从transformer记载)
    criterion : torch.nn.modules.loss
        损失函数
    hidden_size : int
        隐含变量的维度大小(权重矩阵W_{ih}、W_{hh}中h的大小)
    num_layers : int
        循环神经网络层数
    bidirectional : bool
        是否为设置为双向循环神经网络
    dropout_ratio : float
        元素归零的概率
    """

    def __init__(self, pretrained_model, num_class, criterion, hidden_size, num_layers, bidirectional, dropout_ratio=0.3):
        super(BertRNN, self).__init__()
        self.pretrained = pretrained_model
        self.criterion = criterion
        self.bidirectional = bidirectional
        self.rnn = nn.GRU(input_size=pretrained_model.config.hidden_size,
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

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        out = self.pretrained(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids)

        # last_hidden_state.shape=[batch_size, sequence_length, pretrained_model.config.hidden_size]
        last_hidden_state = out.last_hidden_state
        # hidden.shape=[num_layers * num directions, batch_size, hidden_size]
        out, hidden = self.rnn(last_hidden_state)

        if self.bidirectional:  # 双向时
            # hidden = [batch_size, hidden_size * num directions]
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))  # 利用前向后后向的信息
        else:
            hidden = self.dropout(hidden[-1, :, :])
        out = self.linear(hidden)  # result.shape=[batch_size, num_class]
        out = out.softmax(dim=1)

        loss = None
        if labels is not None:  # 若包含标签
            loss = self.criterion(out, labels)

        # 训练与评估阶段
        # ★★★★★
        # 返回值为一个元组
        # 元组的第一个元素必须为该批次数据的损失值
        # 元组的第二个元素为该批次数据的预测值(可选)
        # * 验证数据集评估函数指标的计算
        # * predict方法预测结果(predictions)与评估结果(metrics)(结合输入labels)的计算
        if loss is not None:
            return loss, out
        # 预测阶段
        # ★★★★★
        # 返回值为模型的预测结果
        else:
            return out
