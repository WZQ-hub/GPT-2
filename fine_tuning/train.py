import torch
from fine_tuning_model import create_model_for_finetuning
import time
from dataloader import train_loader, val_loader, test_loader
# print(model)

def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches - min(len(data_loader), num_batches)
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]
            predict_labels = torch.argmax(logits, dim=-1)
            num_examples += predict_labels.shape[0]
            correct_predictions += (predict_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    model.eval()
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(len(data_loader), num_batches)

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter):
    best_val_loss = float("inf")
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, 0
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device ,eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch + 1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, " f"Val loss {val_loss:.3f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), "best_model.pth")
                    print(f">>> 检测到更低的 Val Loss，模型已更新至 best_model.pth")
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy * 100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_losses = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_losses = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
        model.train()
        return train_losses, val_losses
if __name__ == "__main__":
    model = create_model_for_finetuning()
    start_time = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    num_epochs = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_losses, val_losses, train_accs, val_accs = train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq=50, eval_iter=5)
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")