import torch
from torch import nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=3):
        super().__init__()

        # encode
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.hidden_to_latent = nn.Linear(hidden_dim, latent_dim)

        # latent space
        self.latent_to_hidden = nn.Linear(latent_dim, input_dim)
        self.latent_to_input = nn.Linear(latent_dim, input_dim)

        # decoder
        self.decoder = nn.LSTM(input_dim, input_dim, num_layers, batch_first=True)

    def forward(self, x):
        # encode
        _, (hidden_state, _) = self.encoder(x)
        hidden_state = hidden_state[-1]
        latent = self.hidden_to_latent(hidden_state)

        # decode
        hidden = self.latent_to_hidden(latent).unsqueeze(0)
        hidden = hidden.repeat(self.decoder.num_layers, 1, 1)
        c_0 = torch.zeros_like(hidden)

        latent_seq = self.latent_to_input(latent).unsqueeze(1).repeat(1, x.size(1), 1)
        reconstructed, _ = self.decoder(latent_seq, (hidden, c_0))

        return latent, reconstructed


class AE(nn.Module):
    def __init__(self, hidden_layers, dropout_rate: float = 0.2):
        super().__init__()

        # encoder layers
        encoder_layers = []
        for i in range(0, len(hidden_layers) - 1):
            encoder_layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            encoder_layers.append(nn.LeakyReLU())
            encoder_layers.append(nn.Dropout(dropout_rate))

        # decoder layers
        decoder_layers = []
        hidden_layers = hidden_layers[::-1]
        for i in range(0, len(hidden_layers) - 1):
            decoder_layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            decoder_layers.append(nn.LeakyReLU())

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
