import torch
from model import FastSpeech

# Parametrlar
vocab_size = 100  # Matnli lug‘at hajmi
sequence_length = 50  # Kiritiladigan tokenlar soni

# Modelni yaratish
model = FastSpeech(vocab_size)

# Sinov uchun dummy kirish
input_text = torch.randint(0, vocab_size, (1, sequence_length))  # (batch_size, seq_length)

# Modelni ishga tushirish
output_mel = model(input_text)

print("Mel-spectrogram shakli:", output_mel.shape)  # (batch_size, seq_length, 80)
