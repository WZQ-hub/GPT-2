import pandas as pd
import torch
from torch.utils.data import Dataset
import tiktoken
from torch.utils.data import DataLoader


tokenizer = tiktoken.get_encoding("gpt2")

class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]
        # 计算最长text，以便后续padding
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            # 按照给定的max_length截断，有可能会丢失信息
            self.max_length = max_length
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]
        self.encoded_texts = [
            encoded_text + [pad_token_id]*(self.max_length - len(encoded_text)) for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
        torch.tensor(encoded, dtype=torch.long),
        torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)


    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            max_length = max(len(encoded_text), max_length)
        return max_length

train_dataset = SpamDataset(
csv_file="../data/sms_spam_collection/train.csv",
max_length=None,
tokenizer=tokenizer
)
val_dataset = SpamDataset(
csv_file="../data/sms_spam_collection/validation.csv",
max_length=train_dataset.max_length,
tokenizer=tokenizer
)
test_dataset = SpamDataset(
csv_file="../data/sms_spam_collection/test.csv",
max_length=train_dataset.max_length,
tokenizer=tokenizer
)
# print(f"训练集大小: {len(train_dataset)}")
# print(f"验证集大小: {len(val_dataset)}")
# print(f"测试集大小: {len(test_dataset)}")


num_workers = 0
batch_size = 8
torch.manual_seed(123)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False)

test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False)

# for input_batch, target_batch in train_loader:
#     pass
# print("Input batch dimensions:", input_batch.shape)
# print("Label batch dimensions", target_batch.shape)