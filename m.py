# In model.py

import torch
import torch.nn as nn


class FastSpeech2(nn.Module):
    def __init__(self, config):
        super(FastSpeech2, self).__init__()
        self.config = config

        # Encoder
        self.encoder = nn.Sequential(
            nn.Embedding(config.vocab_size, config.model_dim),
            nn.Linear(config.model_dim, config.model_dim)
        )

        # Variance Adaptor
        self.variance_adaptor = nn.ModuleList([
            nn.Linear(config.model_dim, config.model_dim),
            nn.Linear(config.model_dim, config.model_dim),
            nn.Linear(config.model_dim, config.model_dim)
        ])

        # Duration/Pitch/Energy Predictor
        self.duration_predictor = nn.Sequential(
            nn.Conv1d(config.model_dim, config.model_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.LayerNorm(config.model_dim),  # Updated Line
            nn.Dropout(0.1),
            nn.Conv1d(config.model_dim, 1, kernel_size=3, padding=1)
        )
        # Similar structure for pitch and energy predictors

        # Waveform Decoder
        self.waveform_decoder = nn.Sequential(
            nn.Conv1d(config.model_dim, config.model_dim, kernel_size=1),
            nn.ReLU(),
            nn.LayerNorm(config.model_dim),
            nn.Dropout(0.1),
            nn.Conv1d(config.model_dim, config.model_dim, kernel_size=5, padding=2, dilation=1),
            nn.GELU(),
            nn.ConvTranspose1d(config.model_dim, config.model_dim, kernel_size=5, stride=2, padding=2,
                               output_padding=1),
            nn.ReLU()
        )

        # Mel-spectrogram Decoder
        self.mel_decoder = nn.Linear(config.model_dim, config.n_mels)

    def forward(self, x):
        # Placeholder for forward pass
        encoded = self.encoder(x)
        adapted = self.variance_adaptor[0](encoded)
        duration = self.duration_predictor(adapted.transpose(1, 2)).transpose(1, 2)
        # Add logic for pitch and energy prediction
        decoded = self.waveform_decoder(adapted)
        mel_spec = self.mel_decoder(decoded)
        return mel_spec