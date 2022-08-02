import torch.nn as nn
import torch


class BertLastFour_MeanMaxPool(torch.nn.Module):
    """Bert最后四层隐藏层的连接 + [MeanPool, MaxPool](transformer实现训练过程)"""

    def __init__(self, pretrained_model, num_class, criterion, dropout_ratio=0.2):
        super().__init__()
        self.pretrained = pretrained_model
        self.hidden_size = pretrained_model.config.hidden_size
        self.linear1 = torch.nn.Linear(self.hidden_size * 8, 1024)
        self.norm = nn.LayerNorm(normalized_shape=1024)
        self.dropout = nn.Dropout(dropout_ratio)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(1024, num_class)
        self.criterion = criterion

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        model_output = self.pretrained(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids)
        # all_hidden_states.shape=[pretrained_model.config.num_hidden_layers + 1, batch_size, sequence_length, self.hidden_size]
        all_hidden_states = torch.stack(model_output.hidden_states)
        # concatenate_pooling.shape=[batch_size, sequence_length, self.hidden_size * 4]
        concatenate_pooling = torch.cat(
            (all_hidden_states[-1], all_hidden_states[-2], all_hidden_states[-3], all_hidden_states[-4]),
            -1)  # 最后四层隐藏层的连接
        
        # mean_pooling.shape=[batch_size, self.hidden_size * 4]
        mean_pooling = torch.mean(concatenate_pooling, dim=1)  # 聚合操作
        # max_pooling.shape=[batch_size, self.hidden_size * 4]
        max_pooling = torch.max(concatenate_pooling, dim=1).values  # 聚合操作
        # result.shape=[batch_size, self.hidden_size * 8]
        result = torch.cat((mean_pooling, max_pooling), 1)
        
        # result.shape=[batch_size, 1024]
        result = self.linear1(result)
        result = self.norm(result)
        result = self.relu(result)
        result = self.dropout(result)
        # result.shape=[batch_size, num_class]
        result = self.linear2(result)
        result = result.softmax(dim=1)
        
        loss = None
        if labels is not None:  # 若包含标签
            loss = self.criterion(result, labels)

        # 训练与评估阶段
        # ★★★★★
        # 返回值为一个元组
        # 元组的第一个元素必须为该批次数据的损失值
        # 元组的第二个元素为该批次数据的预测值(可选)
        # * 验证数据集评估函数指标的计算
        # * predict方法预测结果(predictions)与评估结果(metrics)(结合输入labels)的计算
        if loss is not None:
            return loss, result
        # 预测阶段
        # ★★★★★
        # 返回值为模型的预测结果
        else:
            return result
