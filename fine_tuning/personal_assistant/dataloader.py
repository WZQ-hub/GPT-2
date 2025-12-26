import json
import torch
from torch.utils.data import Dataset
from fine_tuning.utils.format import format_input
from functools import partial
from torch.utils.data import DataLoader
import tiktoken


file_path = "../instruction-data.json"
with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

train_portion = int(len(data) * 0.85)
test_portion = int(len(data) * 0.1)
val_portion = len(data) - train_portion - test_portion

train_data = data[:train_portion]
test_data = data[train_portion : train_portion + test_portion]
val_data = data[train_portion + test_portion :]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Training set length:", len(train_data))
# print("Validation set length:", len(val_data))
# print("Test set length:", len(test_data))

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.encoded_texts[idx]


def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100, allowed_max_length=None, device="cpu"):
    batch_max_length = max(len(item) + 1 for item in batch)
    input_1st, target_1st = [], []
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = (
            new_item + [pad_token_id] * (batch_max_length - len(new_item))
        )

        # 这就是为什么开局先加一个pad_token_id的原因
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        mask = targets == pad_token_id # eg: mask = [False, False, False, True, True]
        indices = torch.nonzero(mask).squeeze() # eg: indices = [3, 4]
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        input_1st.append(inputs)
        target_1st.append(targets)
    input_tensor = torch.stack(input_1st).to(device)
    target_tensor = torch.stack(target_1st).to(device)
    return input_tensor, target_tensor

customized_collate_fn = partial(
    custom_collate_fn,
    device = device,
    allowed_max_length=1024,
)

tokenizer = tiktoken.get_encoding("gpt2")
num_workers = 0
batch_size = 8
torch.manual_seed(123)

train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers,
)
val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=True,
    num_workers=num_workers,
)
test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=True,
    num_workers=num_workers,
)

# print("Train loader:")
# for inputs, targets in train_loader:
#     print(inputs.shape, targets.shape)









