import pandas as pd
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os

#config
CONFIG = {
    'sample_rate': 16000,
    'duration': 7,
    'n_mels': 128,
    'fmax': 8000,
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

CLASS_NAMES = [
    'Communication',
    'Gunshot',
    'Footsteps',
    'Shelling',
    'Vehicle',
    'Helicopter',
    'Fighter Jet'
]

#Log-Mel Spectrogram extraction
def extract_mel_spectrogram(file_path, sr=16000, duration=7, n_mels=128, fmax=8000):
    """Extract mel-spectrogram from audio file"""
    try:
        audio, _ = librosa.load(file_path, sr=sr, duration=duration)
        target_length = sr * duration

        if len(audio) < target_length:
            # Apply fade out before padding
            fade_samples = int(sr * 0.05)  # 50ms fade
            if len(audio) > fade_samples:
                fade_curve = np.linspace(1, 0, fade_samples)
                audio[-fade_samples:] *= fade_curve
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]

        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=n_mels,
            fmax=fmax,
            hop_length=512,
            n_fft=2048
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
        return mel_spec_norm

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


#Dataset extraction
class AudioDataset(Dataset):
    """Custom Dataset for audio classification"""
    def __init__(self, csv_path, base_path=''):
        self.df = pd.read_csv(csv_path)
        self.base_path = base_path
        self.features, self.labels = [], []

        print(f"Loading data from {csv_path}...")
        for idx, row in self.df.iterrows():
            if idx % 100 == 0:
                print(f"Processed {idx}/{len(self.df)} files...")
            file_path = os.path.join(self.base_path, row['path'])
            mel_spec = extract_mel_spectrogram(file_path)
            if mel_spec is not None:
                self.features.append(mel_spec)
                self.labels.append(row['label'])

        self.features = np.array(self.features)
        self.labels = np.array(self.labels)
        print(f"Data loading complete. Loaded {len(self.labels)} samples.")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.FloatTensor(self.features[idx]).unsqueeze(0)
        label = torch.LongTensor([self.labels[idx]])[0]
        return feature, label


#CNN Model
class AudioCNN(nn.Module):
    """CNN model for audio classification"""
    def __init__(self, num_classes=7):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )

        # Flattened size approximation
        self.flatten_size = 128 * 16 * 27

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)
        return x


#Training and Testing
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(dataloader), 100. * correct / total


def test(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for (inputs, labels) in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return running_loss / len(dataloader), 100. * correct / total, all_preds, all_labels


#Training Loop
def train_model(train_csv, test_csv, base_path=''):
    device = CONFIG['device']

    print("\n=== Loading Data ===")
    train_dataset = AudioDataset(train_csv, base_path)
    test_dataset = AudioDataset(test_csv, base_path)
    print(f"\nTraining samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")

    # Class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(train_dataset.labels), y=train_dataset.labels)
    class_weights = torch.FloatTensor(class_weights).to(device)
    print("Class weights:", class_weights.cpu().numpy())

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2)

    print("\n=== Building Model ===")
    model = AudioCNN(num_classes=len(CLASS_NAMES)).to(device)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    best_test_acc = 0.0
    patience, patience_counter = 15, 0

    print("\n=== Training Model ===")
    for epoch in range(CONFIG['epochs']):
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, _, _ = test(model, test_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")

        # Save best model by test accuracy
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Best model saved (test Acc: {test_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs.")
            break

    model.load_state_dict(torch.load('best_model.pth'))

    print("\n=== Final Evaluation ===")
    test_loss, test_acc, test_preds, test_labels = test(model, test_loader, criterion, device)
    print(f"Final Test Accuracy: {test_acc:.2f}% | Loss: {test_loss:.4f}")
    print("\nClassification Report:\n", classification_report(test_labels, test_preds, target_names=CLASS_NAMES))
    print("Confusion Matrix:\n", confusion_matrix(test_labels, test_preds))

    plot_training_history(history)
    torch.save(model.state_dict(), 'audio_classifier_final.pth')
    print("\nModel saved as 'audio_classifier_final.pth'")
    return model, history


#Plot Training History
def plot_training_history(history):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history['train_acc'], label='Train Acc')
    ax1.plot(epochs, history['test_acc'], label='Test Acc')
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, history['train_loss'], label='Train Loss')
    ax2.plot(epochs, history['test_loss'], label='Test Loss')
    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    print("Training history saved as 'training_history.png'")

if __name__ == "__main__":
    #CHANGE THESE PATHS ON EACH NEW MACHINE
    TRAIN_CSV = '/home/elton/NREIP/training.csv'
    TEST_CSV = '/home/elton/NREIP/test.csv'
    BASE_PATH = '/home/elton/NREIP'
    train_model(TRAIN_CSV, TEST_CSV, BASE_PATH)
