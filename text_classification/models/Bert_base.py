import torch
import torch.nn as nn


class Bert_base(torch.nn.Module):
    """Bert + Linear基础模型(transformer实现训练过程)"""

    def __init__(self, pretrained_model, num_class, criterion, dropout_ratio=0.3):
        super().__init__()
        self.pretrained = pretrained_model
        self.hidden_size = pretrained_model.config.hidden_size
        self.fc = torch.nn.Linear(self.h_size, num_class)
        self.criterion = criterion
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        model_output = self.pretrained(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids)

        # model_output.pooler_output.shape=[batch_size, self.hidden_size]
        out = self.fc(self.dropout(model_output.pooler_output))
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
