import torch
import torch.nn as nn

class ObservationEncoder(nn.Module):
    def __init__(self, input_dim=2, embedding_dim=32):
        """
        Initializes the Observation Encoder.

        :param input_dim: Dimensionality of the raw observation.
        :param embedding_dim: Dimensionality of the encoded embedding.
        """
        super(ObservationEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim),
            nn.ReLU()
        )

    def forward(self, observation):
        """
        Encodes the raw observation into an embedding.

        :param observation: Tensor of shape (batch_size, input_dim).
        :return: Tensor of shape (batch_size, embedding_dim).
        """
        return self.encoder(observation)