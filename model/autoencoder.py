import torch
from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_layers, dropout_rate: float = 0.2):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = Encoder(input_size, hidden_layers, dropout_rate)
        self.decoder = Decoder(self.encoder)
        self.hidden_layers = hidden_layers

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class Encoder(nn.Module):
    def __init__(self, input_size: int, hidden_layers, dropout_rate: float = 0.2, activation=nn.LeakyReLU()):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.input_size = input_size
        self.dropout_rate = dropout_rate

        self.input_layer = torch.nn.Linear(self.input_size, self.hidden_layers[0])

        self.n_layers = 0
        for i in range(0, len(hidden_layers) - 1):
            setattr(self, f"dense{i}", torch.nn.Linear(self.hidden_layers[i], hidden_layers[i + 1]))
            self.n_layers += 1

        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        x = self.activation(self.input_layer(x))

        for i in range(0, self.n_layers - 1):
            dense = getattr(self, f"dense{i}")
            x = self.activation(dense(x))
            x = self.dropout(x)

        output_layer = getattr(self, f"dense{self.n_layers - 1}")
        return output_layer(x)


class Decoder(nn.Module):
    def __init__(self, encoder, activation=nn.LeakyReLU()):
        super().__init__()

        self.encoder = encoder
        self.hidden_layers = self.encoder.hidden_layers[::-1]

        for i in range(0, self.encoder.n_layers):
            setattr(self, f"dense{i}", torch.nn.Linear(self.hidden_layers[i], self.hidden_layers[i + 1]))

        self.output_layer = torch.nn.Linear(self.hidden_layers[-1], self.encoder.input_size)
        self.activation = activation
        self.dropout = nn.Dropout(self.encoder.dropout_rate)

    def forward(self, x):
        for i in range(0, self.encoder.n_layers):
            dense = getattr(self, f"dense{i}")
            x = dense(x)
            x = self.activation(x)
            x = self.dropout(x)

        return self.output_layer(x)