# _shared/train_util.py
import os, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def default_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

def build_model(num_classes=2):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def train_loop(
    train_dataset,
    val_dataset,
    batch_size=32,
    lr=1e-4,
    num_epochs=30,
    model_save_path="best_model.pth",
    patience=5,
):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    patience_cnt = 0

    for epoch in range(1, num_epochs+1):
        print(f"\n[Epoch {epoch}/{num_epochs}]")
        # Train
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for x, y in tqdm(train_loader, desc="Train", ncols=100):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            tr_loss += loss.item() * x.size(0)
            tr_correct += (out.argmax(1) == y).sum().item()
            tr_total += y.size(0)

        tr_loss /= tr_total
        tr_acc = tr_correct / tr_total

        # Val
        model.eval()
        va_loss, va_correct, va_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc="Val  ", ncols=100):
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                va_loss += loss.item() * x.size(0)
                va_correct += (out.argmax(1) == y).sum().item()
                va_total += y.size(0)

        va_loss /= va_total
        va_acc = va_correct / va_total

        print(f"=> Train L:{tr_loss:.4f} A:{tr_acc:.4f} | Val L:{va_loss:.4f} A:{va_acc:.4f}")

        if va_loss < best_val:
            torch.save(model.state_dict(), model_save_path)
            print("✅ Best model saved.")
            best_val = va_loss
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print("⛔ Early stopping.")
                break

    return model_save_path
