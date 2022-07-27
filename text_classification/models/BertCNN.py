import torch.nn as nn
import torch


class BertCNN(nn.Module):
    """
    BertCNN(类似TextCNN)模型的pytorch实现(具体任务对应修改)(transformer实现训练过程)

    Parameters
    ---------
    num_class : int
        类别数
    pretrained_model : torch.nn.modules
        预训练模型(从transformer记载)
    criterion : torch.nn.modules.loss
        损失函数
    kernel_sizes : tuple
        一般来说:不同大小卷积核的组合通常优于同样大小的卷积核
        不同卷积层卷积核的宽度;如:kernel_sizes=(3, 4, 5)
    num_channels : tuple
        不同卷积层输出通道数;如:num_channels=(100, 100, 100)
    dropput_ratio : float
        dropout层p值(建议在0.2到0.5之间,0.2比较建议)
    """

    def __init__(self, pretrained_model, num_class, criterion, kernel_sizes, num_channels, dropout_ratio=0.2):
        super(BertCNN, self).__init__()
        self.pretrained = pretrained_model
        self.criterion = criterion
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.decoder = nn.Linear(sum(num_channels), num_class)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.relu = nn.ReLU()
        # 通过nn.ModuleList()创建多个⼀维卷积层
        self.convs = nn.ModuleList()
        pre_hidden_size = pretrained_model.config.hidden_size
        for out_channels, kernel_sizes in zip(num_channels, kernel_sizes):
            self.convs.append(
                nn.Conv1d(in_channels=pre_hidden_size, out_channels=out_channels, kernel_size=kernel_sizes))

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        out = self.pretrained(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids)

        # last_hidden_state.shape=[batch_size, sequence_length, pretrained_model.config.hidden_size]
        last_hidden_state = out.last_hidden_state

        # 根据⼀维卷积层的输⼊格式,重新排列张量,以便通道作为第2维
        # last_hidden_state.shape=[batch_size, pretrained_model.config.hidden_size, sequence_length]
        last_hidden_state = last_hidden_state.permute(0, 2, 1)

        # conv(last_hidden_state).shape=[batch_size, out_channels, L_out];其中out_channelsh表示输出通道数,L_out表示每个输出通道的宽度
        # self.pool(conv(last_hidden_state)).shape=[batch_size, output_channels, 1]
        # torch.squeeze(self.relu(self.pool(conv(last_hidden_state))), dim=-1).shape=[batch_size, output_channels]
        # encoding.shape=[N, output_channels1 + output_channels2 + output_channels3 + .......]
        encoding = torch.cat(
            [torch.squeeze(self.relu(self.pool(conv(last_hidden_state))), dim=-1) for conv in self.convs],
            dim=1)

        # outputs.shape=[N, num_class]
        out = self.decoder(self.dropout(encoding))
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
