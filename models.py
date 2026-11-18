import re
import torch
import torch.nn as nn
from torchvision import transforms

# ---------- TEXT ----------
TEXT_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]  # dair-ai/emotion order

def tokenize(text):
    return re.findall(r"[a-z']+", text.lower())

def encode(text, vocab, max_len=40):
    ids = [vocab.get(t, 1) for t in tokenize(text)][:max_len]  # 0=<pad>, 1=<unk>
    return ids if ids else [1]

class TextLSTM(nn.Module):
    def __init__(self, vocab_size, emb=128, hid=128, n_cls=6, drop=0.4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hid, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(hid * 2, n_cls)

    def forward(self, x, lengths):
        e = self.drop(self.emb(x))
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        h = torch.cat([h[-2], h[-1]], dim=1)
        return self.fc(self.drop(h))

# ---------- FACE ----------
FACE_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]  # ImageFolder alphabetical

face_eval_tf = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((48, 48)),
    transforms.RandomAutocontrast(p=1.0),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

def conv_block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
        nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
        nn.MaxPool2d(2), nn.Dropout(0.25),
    )

class FaceCNN(nn.Module):
    def __init__(self, n_cls=7):
        super().__init__()
        self.features = nn.Sequential(conv_block(1, 32), conv_block(32, 64), conv_block(64, 128))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 6 * 6, 256), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(256, n_cls))

    def forward(self, x):
        return self.classifier(self.features(x))