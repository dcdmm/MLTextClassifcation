import pandas as pd
import re
import collections
import torch
from torch.utils.data import DataLoader
import jsonlines
from tqdm.notebook import tqdm


def tokenizer(line):
    """基础英文分词器"""
    _patterns = [r"\'", r"\"", r"\.", r"<br \/>", r",", r"\(", r"\)", r"\!", r"\?", r"\;", r"\:", r"\s+"]
    _replacements = [" '  ", "", " . ", " ", " , ", " ( ", " ) ", " ! ", " ? ", " ", " ", " "]
    _patterns_dict = list((re.compile(p), r) for p, r in zip(_patterns, _replacements))
    line = line.lower()
    for pattern_re, replaced_str in _patterns_dict:
        line = pattern_re.sub(replaced_str, line)
    return line.split()


class Vocab:
    """
    Vocabulary for text
    """

    def __init__(self, tokens=None, min_freq=2, reserved_tokens=None):
        # tokens: 单词tokens
        # min_freq: The minimum frequency needed to include a token in the vocabulary.
        # reserved_tokens: 自定义tokens
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        counter = collections.Counter(tokens)
        # Sort according to frequencies
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # The index for the unknown token is 0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)  # 未在字典中则返回'<unk>'
        return [self.__getitem__(token) for token in tokens]  # 递归

    def to_tokens(self, indices):
        """第indices位置处的token"""
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):
        """Index for the unknown token"""
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs


class Vectors:
    def __init__(self, name, max_vectors=None) -> None:
        self.vectors = None
        self.name = name
        self.max_vectors = max_vectors
        self.itos = None
        self.stoi = None
        self.cache()

    def cache(self):
        with open(self.name, "r", encoding='utf-8') as f:
            read_value = f.readlines()

        all_value, itos = [], []
        for i in tqdm(range(len(read_value))):
            l_split = read_value[i].split(' ')
            itos.append(l_split[0])
            all_value.append([float(i.strip()) for i in l_split[1: ]])
        all_value = torch.tensor(all_value)
        self.vectors = all_value
        self.itos = itos
        num_lines = len(self.vectors)
        if not self.max_vectors or self.max_vectors > num_lines:
            self.max_vectors = num_lines
        self.vectors = self.vectors[:self.max_vectors, :]
        self.itos = self.itos[:self.max_vectors]
        self.stoi = {word: i for i, word in enumerate(self.itos)}

    def __len__(self):
        return len(self.vectors)
    
    def __getitem__(self, token):
        if token in self.stoi:
            return self.vectors[self.stoi[token]]
        else:
            dim = self.vectors.shape[1]
            return torch.Tensor.zero_(torch.Tensor(dim))
        
    def get_vecs_by_tokens(self, tokens):
        indices = [self[token] for token in tokens]
        vecs = torch.stack(indices)
        return vecs


class DataProcess:
    def __init__(self, train_path, test_path, device, batch_size=16):
        train_lst = []
        with open(train_path, encoding='utf-8') as fp:
            for item in jsonlines.Reader(fp):
                train_lst.append(item)
        self.train_df = pd.DataFrame(train_lst)
        test_lst = []
        with open(test_path, encoding='utf-8') as fp:
            for item in jsonlines.Reader(fp):
                test_lst.append(item)
        self.test_df = pd.DataFrame(test_lst)
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.vocab = self.build_vocab()

    def build_vocab(self):
        r"""构建词表"""
        split_list = []
        for _, ser in self.train_df.iterrows():
            split_list.extend(tokenizer(ser['text']))
        
        vocab = Vocab(split_list, min_freq=1, reserved_tokens=['<pad>'])
        return vocab
        
    def get_pre_trained(self, name):
        r"""获取模型预训练词向量矩阵"""
        # 加载预训练词向量文件
        vec1 = Vectors(name=name)
        pre_trained = vec1.get_vecs_by_tokens(self.vocab.idx_to_token)
        return pre_trained

    def get_dataLoader(self,
                       text_max_len):  # 句子最大长度(超过该长度将被截断)
        r"""获取DataLoade"""
        text_pipeline = lambda line: [self.vocab[token] for token in tokenizer(line)]

        def collate_batch(batch):
            label_list = []  # 分类标签
            text_list = []
            for _text, _label, _ in batch:
                label_list.append(_label)
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
            return line[:text_max_len]  # 句子截断(句子长度>text_max_len时)
        return line + [padding_token] * (text_max_len - len(line))  # 句子填充(句子长度<text_max_len时)


if __name__ == '__main__':
    cpu_or_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dp = DataProcess('train.jsonl', 'test.jsonl', cpu_or_gpu)
    print(dp.vocab)
    train_loader, test_loader = dp.get_dataLoader(141)
    for i in train_loader:
        print(i)
        break
    pre_vector = dp.get_pre_trained("glove.6B.50d.txt")
    print(pre_vector.shape)
    print(pre_vector)
