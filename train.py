import os
import torch
from tqdm import tqdm
from config import checkpoint_path


def train(model, num_epochs, train_loader, loss_fn, optimizer):
    """
    Runs the training loop with validation and best model checkpointing.

    Args:
        model        : ForestPersonCNN instance
        num_epochs   : number of epochs to train
        train_loader : DataLoader for training split
        loss_fn      : loss function (e.g. nn.CrossEntropyLoss())
        optimizer    : optimizer (e.g. torch.optim.Adam)

    Returns:
        history (dict): train loss/acc and val loss/acc per epoch
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    # Load val loader for checkpointing
    from dataset import get_dataloader
    val_loader = get_dataloader(split="val", shuffle=False)

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [],
                "val_loss":   [], "val_acc":   []}

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        # ── Train ──────────────────────────────────────────────────
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in tqdm(train_loader, desc="  Train"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * images.size(0)
            preds          = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += images.size(0)

        train_loss /= train_total
        train_acc   = train_correct / train_total

        # ── Validate ───────────────────────────────────────────────
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="  Val  "):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss    = loss_fn(outputs, labels)

                val_loss    += loss.item() * images.size(0)
                preds        = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += images.size(0)

        val_loss /= val_total
        val_acc   = val_correct / val_total

        # ── Log ────────────────────────────────────────────────────
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"  Train loss: {train_loss:.4f}  acc: {train_acc:.4f}")
        print(f"  Val   loss: {val_loss:.4f}  acc: {val_acc:.4f}")

        # ── Save best model ────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ Best model saved (val_acc={val_acc:.4f})")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    return history


# ── Sanity check ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import torch.nn as nn
    from model import ForestPersonCNN
    from dataset import get_dataloader, ForestPersonsDataset
    from config import learning_rate, dropout
    from torch.utils.data import DataLoader, Subset

    model     = ForestPersonCNN(dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn   = nn.CrossEntropyLoss()

    # Use a tiny subset for the sanity check
    tiny_train = Subset(ForestPersonsDataset(split="train"), range(20))
    train_loader = DataLoader(tiny_train, batch_size=4, shuffle=True)

    history = train(model, num_epochs=2,
                    train_loader=train_loader,
                    loss_fn=loss_fn,
                    optimizer=optimizer)
    print("\nhistory keys:", list(history.keys()))
    print("train.py looks good!")
