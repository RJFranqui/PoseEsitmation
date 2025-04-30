import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from DatasetAngle import SquatPhaseDataset
from collections import deque
from sklearn.preprocessing import StandardScaler
import joblib

# ====================== Configuration ======================
CONFIG = {
    'input_size': 5,  # Will be set automatically
    'hidden_size': 256,
    'num_layers': 2,
    'lr': 0.0001,
    'batch_size': 32,
    'epochs': 50,
    'threshold_pct': 50,
    'sigma': 1.0,
    'early_stopping_patience': 10,
    'class_weights': torch.tensor([1.0, 1.0])
}

# ====================== Weighted Focal Loss ======================
class WeightedFocalLoss(nn.Module):
    def __init__(self, class_weights, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        targets = targets.long()
        ce_loss = nn.CrossEntropyLoss(weight=self.class_weights)(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# ====================== Model Architecture ======================
class PhaseLSTM(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=CONFIG['hidden_size'],
            num_layers=CONFIG['num_layers'],
            batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(CONFIG['hidden_size'], 2)
        )
        
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.1)

    def forward(self, x):
        x = x.float()
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1])

# ====================== Feature Distribution Diagnostics ======================
def check_feature_distributions(dataset):
    features_up = []
    features_down = []

    for seq, label in dataset:
        if label == 0:
            features_up.append(seq.numpy())
        elif label == 1:
            features_down.append(seq.numpy())

    features_up = np.concatenate(features_up)
    features_down = np.concatenate(features_down)

    print("\n=== Feature Statistics ===")
    print(f"UP samples: {len(features_up)}")
    print(f"DOWN samples: {len(features_down)}")

    plt.figure(figsize=(15, 8))
    num_features = min(5, features_up.shape[1])
    for i in range(num_features):
        plt.subplot(2, 3, i+1)
        plt.hist(features_up[:, i].ravel(), bins=50, alpha=0.5, label='UP')
        plt.hist(features_down[:, i].ravel(), bins=50, alpha=0.5, label='DOWN')
        plt.title(f'Feature {i} Distribution')
        plt.legend()
    plt.tight_layout()
    plt.show()

# ====================== Data Preparation ======================
def prepare_loaders():
    class SquatPhaseDatasetWithDelta(SquatPhaseDataset):
        def _create_sequences(self, features, labels):
            sequences = []
            seq_labels = []
            for i in range(0, len(features) - self.seq_length + 1, self.seq_length // 3):
                window = features[i:i + self.seq_length]
                window_labels = labels[i:i + self.seq_length]
                if len(window_labels) < self.seq_length:
                    continue

                delta = np.diff(window, axis=0, prepend=window[0:1])
                full_features = np.concatenate([window, delta], axis=1)  # Shape (seq_length, 8)

                majority = np.bincount(window_labels).argmax()
                sequences.append(full_features)
                seq_labels.append(majority)

            return np.array(sequences), np.array(seq_labels)

    train_set = SquatPhaseDatasetWithDelta("Squat_Train.csv", 
                                           seq_length=30,
                                           threshold_pct=CONFIG['threshold_pct'],
                                           sigma=CONFIG['sigma'])
    val_set = SquatPhaseDatasetWithDelta("Squat_Test.csv",
                                         seq_length=30,
                                         threshold_pct=CONFIG['threshold_pct'],
                                         sigma=CONFIG['sigma'])

    print("=== Training Set Diagnostics ===")
    check_feature_distributions(train_set)

    CONFIG['input_size'] = train_set.data[0].shape[1]

    train_loader = DataLoader(
        train_set,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        collate_fn=train_set.collate_fn
    )

    val_loader = DataLoader(
        val_set,
        batch_size=CONFIG['batch_size'],
        collate_fn=val_set.collate_fn
    )

    return train_loader, val_loader

# ====================== Training with Monitoring ======================
def train_model(train_loader, val_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PhaseLSTM(CONFIG['input_size']).to(device)

    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    criterion = WeightedFocalLoss(class_weights=CONFIG['class_weights'])

    grad_norms = []
    train_losses = []
    val_losses = []
    best_f1 = 0
    patience_counter = 0
    lr_counter = 0

    for epoch in range(CONFIG['epochs']):
        model.train()
        train_loss = 0
        current_grad_norms = []

        for seq, labels in train_loader:
            seq, labels = seq.to(device), labels.long().to(device)

            optimizer.zero_grad()
            outputs = model(seq)
            loss = criterion(outputs, labels)
            loss.backward()

            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            grad_norm = total_norm ** 0.5
            current_grad_norms.append(grad_norm)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        grad_norms.extend(current_grad_norms)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for seq, labels in val_loader:
                seq, labels = seq.to(device), labels.long().to(device)
                outputs = model(seq)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        val_metrics = evaluate(model, val_loader, device)

        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"  Val F1: {val_metrics['f1']:.4f}")
        print(f"  Avg Grad Norm: {np.mean(current_grad_norms):.4f}")
        print(f"  Confusion Matrix:\n{val_metrics['confusion_matrix']}")

        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            patience_counter = 0
            torch.save(model.state_dict(), "models/lstm_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['early_stopping_patience']:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
            if (lr_counter >= 5):
                lr_counter = 0
                for g in optimizer.param_groups:
                    g['lr'] *= 0.5
                print(f"Reducing learning rate to {optimizer.param_groups[0]['lr']:.2e}")
            else:
                lr_counter += 1

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(grad_norms)
    plt.title("Gradient Norms During Training")
    plt.xlabel("Iteration")
    plt.ylabel("Gradient Norm")

    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title("Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return model

# ====================== Evaluation ======================
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for seq, labels in loader:
            seq = seq.to(device)
            outputs = model(seq).cpu()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    return {
        'accuracy': (all_preds == all_labels).mean(),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': confusion_matrix(all_labels, all_preds)
    }

# ====================== Main Execution ======================
if __name__ == "__main__":
    train_loader, val_loader = prepare_loaders()

    # ===== Save LSTM Scaler =====
    # Stack all sequences into one big array
    all_sequences = []
    for seq, label in train_loader.dataset:
        all_sequences.append(seq.numpy())

    X_full = np.vstack(all_sequences)  # Shape (total_frames, 8 features)

    # Fit scaler on all features (angles + deltas)
    lstm_scaler = StandardScaler().fit(X_full)
    joblib.dump(lstm_scaler, "models/lstm_scaler.pkl")
    print("\nLSTM scaler saved successfully as models/lstm_scaler.pkl!")


    print("\n=== Starting Training ===")
    model = train_model(train_loader, val_loader)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    final_metrics = evaluate(model, val_loader, device)

    print("\n=== Final Evaluation ===")
    print(f"Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Precision: {final_metrics['precision']:.4f}")
    print(f"Recall: {final_metrics['recall']:.4f}")
    print(f"F1 Score: {final_metrics['f1']:.4f}")
    print("Confusion Matrix:")
    print(final_metrics['confusion_matrix'])
    print("\n=== Training Complete ===")
