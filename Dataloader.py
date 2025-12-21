import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda")
import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_data = f.read() # 读取文本数据
# print(len(raw_data))
# print(raw_data)

class GPTDataloaderV1(Dataset):
    def __init__(self, data, tokenizer, max_length, stride):
        self.input_ids = []
        self.output_ids = []
        token_ids = tokenizer.encode(data)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i: i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.output_ids.append(torch.tensor(target_chunk))


    def __len__(self):
        return len(self.data)

    def __getstate__(self, idx):
        return self.input_ids[idx], self.output_ids[idx]
