import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as in the original Transformer paper.
    Adds temporal position information to the sequence embeddings.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TemporalEncoder(nn.Module):
    """
    Projects raw node features through a linear layer, adds positional encoding,
    then runs a standard Transformer Encoder to produce a temporal embedding.

    Input:  x  of shape (N, T, F)   N=stocks, T=seq_len, F=feature_dim
    Output: out of shape (N, d_model)  → CLS-token pooled representation
    """

    def __init__(
        self,
        input_dim: int = 8,       # F: number of input features per time step
        d_model: int = 64,        # Transformer hidden dimension
        nhead: int = 4,           # Attention heads (d_model must be divisible)
        num_layers: int = 2,      # Number of TransformerEncoder layers
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        seq_len: int = 30,
    ):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.input_projection = nn.Linear(input_dim, d_model)

        # Learnable [CLS] token prepended to every sequence
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len + 1, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,   # (batch, seq, features)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, T, F)  — N stocks, T time steps, F features
        returns: (N, d_model)
        """
        N = x.size(0)

        # Linear projection: (N, T, F) → (N, T, d_model)
        x = self.input_projection(x)

        # Prepend [CLS] token
        cls = self.cls_token.expand(N, -1, -1)  # (N, 1, d_model)
        x   = torch.cat([cls, x], dim=1)        # (N, T+1, d_model)

        # Add positional encoding
        x = self.pos_enc(x)

        # Transformer: (N, T+1, d_model)
        x = self.transformer(x)
        x = self.norm(x)

        # Return [CLS] token output as the summary embedding
        return x[:, 0, :]  # (N, d_model)