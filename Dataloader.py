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

        # include the last chunk which may be smaller than max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i: i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.output_ids.append(torch.tensor(target_chunk))


    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int):
        return self.input_ids[idx], self.output_ids[idx]

def create_dataloader_v1(data, batch_size = 4, max_length = 512, stride=128, shuffle=True, drop_last=True,
        num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataloaderV1(data, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers, drop_last = drop_last)
    return dataloader

# data_loader = create_dataloader_v1(raw_data, batch_size=1, max_length=4, stride=1, shuffle=False)
# inputs, targets = next(iter(data_loader))
# print("inputs:", inputs)   # [1, 4]
# print("targets:", targets) # [1, 4]
# 验证右移关系：targets 等于 inputs 在时间维上向后对齐一位
# print(torch.all(inputs[:, 1:] == targets[:, :-1]))  # True
