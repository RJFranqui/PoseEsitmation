import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from FrameDataset import SquatKneeFrameDataset

class SquatMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[32, 16], dropout=0.4):
        super(SquatMLP, self).__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_dims:
            linear = nn.Linear(last_dim, h)
            nn.init.xavier_normal_(linear.weight)
            nn.init.constant_(linear.bias, 0.1)

            layers.append(linear)
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            last_dim = h
        
        output_layer = nn.Linear(last_dim, 2)
        nn.init.xavier_normal_(output_layer.weight)
        nn.init.constant_(output_layer.bias, 0.1)
        layers.append(output_layer)
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train(model, train_loader, val_loader, epochs=50, lr=1e-4, patience=5, lr_decay=0.5, min_lr=1e-6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # ---- Training ----
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.squeeze(1).to(device)
            features = features + torch.randn_like(features) * 0.01  # Adding noise for robustness
            outputs = model(features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * features.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_train_loss = total_loss / total
        train_acc = correct / total
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)

        # ---- Validation ----
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.squeeze(1).to(device)

                outputs = model(features)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * features.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        avg_val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2%} | "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2%}")

        # ---- Early Stopping and LR Decay ----
        if avg_val_loss < best_val_loss - 1e-4:  # significant improvement
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save the trained model (state_dict version - recommended)
            torch.save(model.state_dict(), "models/mlp_model.pth")
            print("Model saved successfully!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                for param_group in optimizer.param_groups:
                    old_lr = param_group['lr']
                    new_lr = max(old_lr * lr_decay, min_lr)
                    param_group['lr'] = new_lr
                print(f"Reducing learning rate to {optimizer.param_groups[0]['lr']:.2e}")
                patience_counter = 0

                if optimizer.param_groups[0]['lr'] <= min_lr:
                    print("Minimum learning rate reached. Stopping training.")
                    break

    # ---- Plotting ----
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss", marker='o')
    plt.plot(range(1, len(val_losses)+1), val_losses, label="Validation Loss", marker='o')
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accuracies)+1), train_accuracies, label="Train Accuracy", marker='o')
    plt.plot(range(1, len(val_accuracies)+1), val_accuracies, label="Validation Accuracy", marker='o')
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()



def evaluate(model, dataloader, return_preds=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.squeeze(1).to(device)

            outputs = model(features)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['DOWN', 'UP'], digits=3))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['DOWN', 'UP'], yticklabels=['DOWN', 'UP'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    if return_preds:
        return np.array(all_preds), np.array(all_labels)


if __name__ == "__main__":
    dataset = SquatKneeFrameDataset("Squat_Train.csv", threshold_pct=50, sigma=2.0)
    train_dataset = SquatKneeFrameDataset("Squat_Train.csv", threshold_pct=50, sigma=2.0)
    test_dataset = SquatKneeFrameDataset("Squat_Test.csv", threshold_pct=50, sigma=2.0)
    dataset.analyze()

    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    input_dim = dataset[0][0].shape[0]
    model = SquatMLP(input_dim=input_dim)

    train(model, loader, test_loader, epochs=50, lr=0.001)
    model.load_state_dict(torch.load("models/mlp_model.pth"))
    evaluate(model, test_loader, return_preds=True)


