import sys
sys.path.insert(0, '/home/elton/NREIP/my_pytorch_env/lib/python3.12/site-packages')

import numpy as np
import torch
import torch.nn as nn
import collections
from gnuradio import gr

#dfsa
import librosa


class AudioCNN(nn.Module):
    """CNN model for audio classification"""
    def __init__(self, num_classes=7):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )

        # Flattened size (same as training)
        self.flatten_size = 128 * 16 * 15

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)
        return x


# ============== CONFIG (Same as your training code) ==============
CONFIG = {
    'sample_rate': 16000,
    'duration': 4,
    'n_mels': 128,
    'fmax': 8000,
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


# ============== FEATURE EXTRACTION (Same as your training code) ==============
def extract_mel_spectrogram(audio, sr=16000, duration=4, n_mels=128, fmax=8000):
    """Extract mel-spectrogram from audio array (same as training)"""
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
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-6)
    
    return mel_spec_norm


# ============== GNU RADIO BLOCK ==============
class blk(gr.sync_block):
    """Real-time audio classifier block"""

    def __init__(self):
        gr.sync_block.__init__(
            self,
            name='Audio Classifier',
            in_sig=[np.float32],
            out_sig=[np.float32, np.float32]
        )
        
        # Model path
        model_path = "/home/elton/NREIP/audio_classifier_final.pth"
        
        # Audio buffer (4 seconds at 16kHz)
        self.sample_rate = CONFIG['sample_rate']
        self.duration = CONFIG['duration']
        self.buffer_size = self.sample_rate * self.duration  # 112000 samples
        
        self.buffer = collections.deque(maxlen=self.buffer_size)
        self.samples_count = 0
        self.update_interval = self.sample_rate // 2  # Update every 0.5 seconds
        
        # Prediction smoothing
        self.probs = np.ones(len(CLASS_NAMES)) / len(CLASS_NAMES)
        self.smoothing = 0.3
        self.current_class = 0
        self.confidence = 0.0
        
        # Pre-fill buffer with silence
        for _ in range(self.buffer_size):
            self.buffer.append(0.0)
        
        # Load model
        self.device = torch.device('cpu')
        self.model = AudioCNN(num_classes=len(CLASS_NAMES))
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print(f"\n{'='*50}")
            print(f"AUDIO CLASSIFIER READY")
            print(f"Model: {model_path}")
            print(f"Device: {self.device}")
            print(f"Buffer: {self.duration}s ({self.buffer_size} samples)")
            print(f"Update: every {self.update_interval/self.sample_rate}s")
            print(f"{'='*50}\n")
            self.model_loaded = True
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            self.model_loaded = False

    def work(self, input_items, output_items):
        inp = input_items[0]
        out_audio = output_items[0]
        out_class = output_items[1]
        
        # Pass audio through
        out_audio[:] = inp
        
        # Output current class
        out_class[:] = float(self.current_class)
        
        # Skip if model not loaded
        if not self.model_loaded:
            return len(inp)
        
        # Add samples to buffer
        for s in inp:
            self.buffer.append(float(s))
            self.samples_count += 1
        
        # Classify every update_interval samples
        if self.samples_count >= self.update_interval:
            try:
                # Get audio from buffer
                audio = np.array(self.buffer, dtype=np.float32)
                
                # Extract features (same as training)
                mel_spec = extract_mel_spectrogram(
                    audio,
                    sr=CONFIG['sample_rate'],
                    duration=CONFIG['duration'],
                    n_mels=CONFIG['n_mels'],
                    fmax=CONFIG['fmax']
                )
                
                # Prepare tensor (same as training)
                x = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0)
                x = x.to(self.device)
                
                # Run inference
                with torch.no_grad():
                    outputs = self.model(x)
                    raw_probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                
                # Smooth predictions
                self.probs = self.smoothing * raw_probs + (1 - self.smoothing) * self.probs
                
                # Get prediction
                self.current_class = int(np.argmax(self.probs))
                self.confidence = float(self.probs[self.current_class])
                
                # Display result
                class_name = CLASS_NAMES[self.current_class]
                print(f"[{class_name:15} {self.confidence:.1%}")
                
            except Exception as e:
                print(f"[ERROR] Classification failed: {e}")
            
            self.samples_count = 0
        
        return len(inp)