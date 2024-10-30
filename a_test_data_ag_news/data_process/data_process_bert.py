import torch.utils.data as Data
import pandas as pd
import torch
from transformers import BertTokenizer
import jsonlines


class Dataset(Data.Dataset):
    """定义数据集"""

    def __init__(self, file_path):
        lst = []
        with open(file_path, encoding='utf-8') as fp:
            for item in jsonlines.Reader(fp):
                lst.append(item)
        self.dataset = pd.DataFrame(lst)

    # 必须实现__len__魔法方法
    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, i):
        """定义索引方式"""
        text = self.dataset.iloc[i, :]['text']
        label = self.dataset.iloc[i, :]['class']
        return text, label


def get_collate_fn(tokenizer, max_len=512):
    """返回collate_fun函数(通过闭包函数引入形参)"""

    def collate_fn(data):
        model_input_names = tokenizer.model_input_names
        sents = [i[0] for i in data]

        # 批量编码句子
        text_token = tokenizer(text=sents,
                               truncation=True,
                               padding='max_length',
                               max_length=max_len,
                               return_token_type_ids=True,
                               return_attention_mask=True,
                               return_tensors='pt')
        result = {}
        for name in model_input_names:
            result[name] = text_token[name]

        labels = [i[1] for i in data]
        labels = torch.LongTensor(labels)
        result['labels'] = labels  # ★★★★对应模型forward方法labels参数
        return result

    return collate_fn


if __name__ == '__main__':
    token = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset_test = Dataset('../datasets/test.jsonl')
    dataLoader = Data.DataLoader(dataset=dataset_test, batch_size=4, collate_fn=get_collate_fn(token))
    for i in dataLoader:
        print(i)
        break
