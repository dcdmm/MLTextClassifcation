from torchtext.datasets import AG_NEWS
from torchtext.vocab import build_vocab_from_iterator
from nltk.tokenize import word_tokenize
import nltk
import torch

nltk.download('punkt')

train_iter = AG_NEWS(split='train')


def yield_tokens(data_iter):
    for _, text in data_iter:
        yield word_tokenize(text)


vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])


text_pipeline = lambda x: vocab(word_tokenize(x))
label_pipeline = lambda x: int(x) - 1


def collate_batch(batch, device):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)
