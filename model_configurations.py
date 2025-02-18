# model_configurations.py

from m import FastSpeech2
from config import Config

def get_base_model():
    config = Config()
    return FastSpeech2(config)

def get_advanced_model():
    config = Config()
    config.model_dim = 768  # Larger model dimension
    config.num_epochs = 200  # More training epochs
    return FastSpeech2(config)

# Example usage in main.py or another script
from model_configurations import get_base_model, get_advanced_model

def main():
    base_model = get_base_model()
    advanced_model = get_advanced_model()
    # Use models as needed

if __name__ == "__main__":
    main()