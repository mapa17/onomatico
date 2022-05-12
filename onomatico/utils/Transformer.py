import math
import inspect
from typing import Dict

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)


class PositionalEncoding(nn.Module):
    """Add absolute position encodings to embedding vectors.

    For each embedding token add an (additative) position dependent term that
    is calculated as an superposition of sinus and cosines functions that is
    unique for each position in a sequence.

    For a simple visual instruction watch: https://youtu.be/dichIcUZfOw?t=318

    Copied from
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html#load-and-batch-data
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    kwargs: Dict[str, None]

    def __init__(
        self,
        ntokens: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.2,
    ):
        """Transformer based generative model learning to reproduce token sequences.

        Args:
            ntokens (int): Size of the vocab used for training and generation
            d_model (int): Embedding size of the input and output of the Feed Forward block of a single Transformer Layer
            nhead (int): number of self attention heads
            d_hid (int): Internal Embedding size of the Feed Forward block (emb sizes: d_model x d_hid x d_model)
            nlayers (int): Number of Transformer Layers
            dropout (float, optional): Dropout rate used for the Multi-headed attention and Feed Forward block. Defaults to 0.5.
        """
        super().__init__()

        # Store the kwargs as a dict, so they can be saved with the model
        # and reused when loading the model.
        s = inspect.signature(self.__init__)
        l = locals()
        self.kwargs = {k: l[k] for k in s.parameters.keys()}

        self.model_type = "Transformer"
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntokens, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntokens)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Use the defined Model components and perform the actual computation.

        As input we expect an tensor that contains multiple input sequences (all of 
        the same length) and a mask tensor that indicates for each position in
        the sequence what other positions can be used  in the self attention mechanism. 

        The output are raw, unnormalized scores (logits) for each token position
        that have to be fed to an activation function (e.g softmax) in order to
        be interpreted as a probability distribution of the vocabulary.

        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output
