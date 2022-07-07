import torch.nn as nn
import torch


class BaseModel(nn.Module):
    def __init__(self, pre_trained_embed, num_class):
        super(BaseModel, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(pre_trained_embed)
        self.fc = nn.Linear(pre_trained_embed.shape[1], num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text):
        embedded = self.embedding(text)
        embedded_mean = torch.mean(embedded, dim=1)
        return self.fc(embedded_mean)
