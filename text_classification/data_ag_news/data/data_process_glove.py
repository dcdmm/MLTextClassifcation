import pandas as pd
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torchtext
import torch
from torch.utils.data import DataLoader


class DataProcess:
    def __init__(self, train_path, test_path, device, batch_size=16):
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = get_tokenizer(tokenizer='basic_english')
        self.vocab = self.build_vocab()

    def build_vocab(self):
        r"""构建词表"""

        def yield_tokens(data_iter):
            for _, ser in data_iter:
                yield self.tokenizer(ser['text'])  # 分词

        vocab = build_vocab_from_iterator(yield_tokens(self.train_df.iterrows()))
        vocab.insert_token("<unk>", 0)
        vocab.insert_token("<pad>", 1)
        vocab.set_default_index(0)
        return vocab

    def get_pre_trained(self, name, cache):
        r"""获取模型预训练词向量矩阵"""
        vec1 = torchtext.vocab.Vectors(name=name,
                                       cache=cache)
        pre_trained = vec1.get_vecs_by_tokens(self.vocab.get_itos())
        return pre_trained

    def get_dataLoader(self,
                       text_max_len):  # 句子最大长度(超过该长度将被截断)
        r"""获取DataLoade"""
        label_pipeline = lambda label: int(label) - 1  # 使分类标签从0开始
        text_pipeline = lambda line: [self.vocab([i])[0] for i in self.tokenizer(line)]

        def collate_batch(batch):
            label_list = []  # 分类标签
            text_list = []
            for (_label, _text) in batch:
                label_list.append(label_pipeline(_label))
                processed_text = torch.tensor(
                    DataProcess.truncate_pad(text_pipeline(_text), text_max_len, self.vocab['<pad>']),
                    dtype=torch.int64)
                text_list.append(processed_text)
            label_list = torch.tensor(label_list, dtype=torch.int64)
            text_list = torch.stack(text_list)
            return label_list.to(self.device), text_list.to(self.device)

        test_map_data = DataProcess.to_map_style_dataset(self.test_df)
        train_map_data = DataProcess.to_map_style_dataset(self.train_df)
        test_dataloader = DataLoader(test_map_data, batch_size=self.batch_size, shuffle=False, collate_fn=collate_batch)
        train_dataloader = DataLoader(train_map_data, batch_size=self.batch_size, shuffle=True,
                                      collate_fn=collate_batch)
        return train_dataloader, test_dataloader

    @staticmethod
    def to_map_style_dataset(df):
        r"""Convert DataFrame to map-style dataset."""

        class _MapStyleDataset(torch.utils.data.Dataset):

            def __init__(self, dataframe):
                self._data = dataframe.values

            def __len__(self):
                return self._data.shape[0]

            def __getitem__(self, idx):
                return self._data[idx]

        return _MapStyleDataset(df)

    @staticmethod
    def truncate_pad(line, text_max_len, padding_token):
        r"""截断或填充文本序列"""
        if len(line) > text_max_len:
            return line[:text_max_len]  # 句子截断
        return line + [padding_token] * (text_max_len - len(line))  # 句子填充


if __name__ == '__main__':
    cpu_or_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dp = DataProcess('../datasets/train.csv', '../datasets/test.csv', cpu_or_gpu)
    print(dp.vocab)
    train_loader, test_loader = dp.get_dataLoader(141)
    print(train_loader)
    pre_vector = dp.get_pre_trained("glove.6B.50d.txt", '../../extra/glove_vector/')
    print(pre_vector.shape)
