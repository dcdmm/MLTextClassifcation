import torch.nn as nn
import torch


class Base(nn.Module):
    """Embedding + Linear基础模型"""

    def __init__(self, pre_trained_embed, num_class):
        super(Base, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(pre_trained_embed)
        self.fc = nn.Linear(pre_trained_embed.shape[1], num_class)

    def forward(self, text):
        # text.shape=(N, L);其中L表示序列长度
        # embedded.shape=(N, L, C);其中C表示输出词向量的维度大小(即nn.Embedding类参数embedding_dim)
        embedded = self.embedding(text)
        embedded_mean = torch.mean(embedded, dim=1)  # 聚合操作
        return self.fc(embedded_mean)
