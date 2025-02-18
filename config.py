class Config:
    def __init__(self):
        self.learning_rate = 0.001
        self.batch_size = 32
        self.num_epochs = 100
        self.model_dim = 384
        self.n_mels = 80
        self.vocab_size = 1000  # Add this line, adjust the number according to your vocabulary size