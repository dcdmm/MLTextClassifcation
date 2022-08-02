import torch
import torch.nn as nn


class AttentionHead(nn.Module):
    """注意力权重"""

    def __init__(self, h_size):
        super().__init__()
        self.W_w_b_w = nn.Linear(h_size, h_size)  # 可学习参数:W_w, b_w
        self.u_w = nn.Linear(h_size, 1)  # 可学习参数:u_w

    def forward(self, features):
        # Q, K, V都是features(类似加性注意力)

        # features.shape=[batch_size, sen len, h_size)
        # u.shape=[batch_size, sen len, h_size]
        u = torch.tanh(self.W_w_b_w(features))
        # att.shape=[batch_size, sen len, 1]
        att = self.u_w(u)
        att_score = torch.softmax(att, dim=1)
        # score_out.shape=[batch_size, sen len, h_size]  # 广播机制
        score_out = att_score * features
        # feat=[batch_size, h_size]
        feat = torch.sum(score_out, dim=1)

        return feat


class BertLastFour_att(torch.nn.Module):
    """Bert最后四层隐藏层的连接 + [最后一个序列, attention](transformer实现训练过程)"""

    def __init__(self, pretrained_model, num_class, criterion, dropout_ratio=0.2):
        super().__init__()
        self.pretrained = pretrained_model
        self.hidden_size = pretrained_model.config.hidden_size
        self.fc = torch.nn.Linear(self.hidden_size * 8, num_class)
        self.head = AttentionHead(self.hidden_size * 4)
        self.criterion = criterion
        self.dropout = nn.Dropout(dropout_ratio)

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
        # cls_pooling.shape=[batch_size, self.hidden_size * 4]
        cls_pooling = concatenate_pooling[:, 0]  # 最后一个序列的信息
        # head_logits.shape=[batch_size, self.hidden_size * 4]
        head_logits = self.head(concatenate_pooling)  # 最后4层的注意力权重(可选)
        # out.shape=[batch_size, self.hidden_size *8]
        out = torch.cat([head_logits, cls_pooling], -1)

        # out.shape=[batch_size, num_class]
        out = self.fc(self.dropout(out))  # 拼接
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
