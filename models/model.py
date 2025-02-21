import torch
import torch.nn as nn
import torch.nn.functional as F


class FastSpeech(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256, num_layers=4):
        super(FastSpeech, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.encoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.variance_adaptor = VarianceAdaptor(hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.mel_linear = nn.Linear(hidden_dim * 2, 80)  # Mel-spectrogram output

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.encoder(x)
        x = self.variance_adaptor(x)
        x, _ = self.decoder(x)
        x = self.mel_linear(x)
        return x


class VarianceAdaptor(nn.Module):
    def __init__(self, hidden_dim):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = VariancePredictor(hidden_dim)
        self.pitch_predictor = VariancePredictor(hidden_dim)
        self.energy_predictor = VariancePredictor(hidden_dim)

    def forward(self, x):
        duration = self.duration_predictor(x)
        pitch = self.pitch_predictor(x)
        energy = self.energy_predictor(x)
        return x + duration + pitch + energy  # Simple adaptation


class VariancePredictor(nn.Module):
    def __init__(self, hidden_dim):
        super(VariancePredictor, self).__init__()
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.transpose(1, 2)
        x = self.linear(x)
        return x

