# main.py

from m import FastSpeech2
from config import Config
import torch


def main():
    # Initialize the configuration
    config = Config()

    # Instantiate the model
    model = FastSpeech2(config)

    # Example of how you might use the model for inference
    # Here, we're creating a dummy input tensor
    batch_size = config.batch_size
    sequence_length = 100  # Example length, adjust as needed
    text_input = torch.randint(0, config.vocab_size, (batch_size, sequence_length))

    # Forward pass to get mel-spectrogram
    with torch.no_grad():
        mel_spec = model(text_input)

    print(f"Output shape: {mel_spec.shape}")

    # If you want to simulate training or evaluation, you could add something like this:
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # criterion = nn.MSELoss()  # Example loss, adjust based on your needs

    # Training loop (pseudo-code, you would need to define your actual training process)
    # for epoch in range(config.num_epochs):
    #     for batch in dataloader:  # You'd need to define a dataloader
    #         optimizer.zero_grad()
    #         outputs = model(batch['text'])
    #         loss = criterion(outputs, batch['target'])  # Assuming you have target mel-spectrograms
    #         loss.backward()
    #         optimizer.step()
    #     print(f"Epoch {epoch+1}/{config.num_epochs}, Loss: {loss.item()}")


if __name__ == "__main__":
    main()