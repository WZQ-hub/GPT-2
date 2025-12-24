import os

import torch
from model.GPT_2 import GPT2, GPT_CONFIG_124M
import torch.nn as nn
import tiktoken
from Dataloader import create_dataloader_v1
# model.eval()
# print(model)

# GPT_CONFIG_124M = {
# "vocab_size": 50257, # Vocabulary size
# "context_length": 1024, # Context length
# "emb_dim": 768, # Embedding dimension
# "n_heads": 12, # Number of attention heads
# "n_layers": 12, # Number of layers
# "drop_rate": 0.1, # Dropout rate
# "qkv_bias": False # Query-Key-Value bias
# }

def generate_text_simple(model, idx, max_new_tokens, context_length):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_length:] # 保持输入长度不超过context_length. idx_cond shape:(Batch, T)
        with torch.no_grad():
            logits = model(idx_cond)
            logits = logits[:, -1, :] # 取最后一个时间步的logits. logits shape:(Batch, T, Vocab_size)
        probs = torch.softmax(logits, dim=1)
        next_token = torch.multinomial(probs, 1)
        idx = torch.cat((idx, next_token), dim=1)
    return idx

def tokens_to_text(tokens, tokenizer):
    flat_tokens = tokens.squeeze(0)
    return tokenizer.decode(flat_tokens.tolist())

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # batch dim
    return encoded_tensor

# start_context = "Hello, my name is"
# tokenizer = tiktoken.get_encoding("gpt2")
# token_ids = generate_text_simple(
# model=model, idx=text_to_token_ids(start_context, tokenizer), max_new_tokens=10, context_length=GPT_CONFIG_124M["context_length"] )
# print("Output text:\n", tokens_to_text(token_ids, tokenizer))

# print("Train loader:")
# for x, y in train_loader:
#     print(x.shape, y.shape)
# print("\nValidation loader:")
# for x, y in val_loader:
#     print(x.shape, y.shape)

def calc_loss_batch(model, input_batch, target_batch, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch) # logits shape: (B, T, Vocab_size)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten(0, 1))
    return loss

def calc_loss_loader(model, data_loader, device, num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(len(data_loader), num_batches)
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(model, input_batch, target_batch, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(model, input_batch, target_batch, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch + 1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, " f"Val loss {val_loss:.3f}")
                torch.save(model.state_dict(), "model.pth")
                print("Model saved to model.pth")
        generate_and_print_sample(model, tokenizer, device, start_context)
    return train_losses, val_losses, track_tokens_seen



def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(model, train_loader, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(model, val_loader, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.position_embedding.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded, max_new_tokens=50, context_length=context_size
        )
    decoded_text = tokens_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()



# Generate text with temperature and top-k sampling
def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

# torch.manual_seed(123)
# token_ids = generate(
# model=model,
# idx=text_to_token_ids("Every effort moves you", tokenizer),
# max_new_tokens=15,
# context_size=GPT_CONFIG_124M["context_length"],
# top_k=25,
# temperature=1.4
# )
# print("Output text:\n", tokens_to_text(token_ids, tokenizer))

if __name__ == "__main__":
    # 1. 基础设置
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = tiktoken.get_encoding("gpt2")

    # 2. 初始化模型
    model = GPT2(GPT_CONFIG_124M)
    model.to(device)

    # 3. 加载数据 (使用绝对路径或确保运行路径正确)
    # 建议加上路径检查，防止 FileNotFoundError
    file_path = "data/the-verdict.txt"
    if not os.path.exists(file_path):
        # 尝试向上级找一级，适配在 model/ 目录下运行的情况
        file_path = os.path.join("..", "data", "the-verdict.txt")

    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = f.read()

    # 4. 数据预处理与 DataLoader
    train_rate = 0.90
    split_idx = int(len(raw_data) * train_rate)
    train_data = raw_data[:split_idx]
    val_data = raw_data[split_idx:]

    train_loader = create_dataloader_v1(
        train_data, batch_size=2, shuffle=True,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True, num_workers=0
    )
    val_loader = create_dataloader_v1(
        val_data, batch_size=2, shuffle=False,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False, num_workers=0
    )

    # 5. 训练配置
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    num_epochs = 10

    # 6. 开始训练
    print("Starting training...")
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context="Every effort moves you", tokenizer=tokenizer
    )