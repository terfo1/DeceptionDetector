"""Neural sequence model definitions for eye-tracking sequences."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None


if nn is not None:

    class RecurrentSequenceClassifier(nn.Module):
        """LSTM or GRU binary classifier for fixed-length sequence tensors."""

        def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            dropout: float,
            model_type: str,
        ):
            super().__init__()
            if model_type not in {"lstm", "gru"}:
                raise ValueError("model_type must be 'lstm' or 'gru'.")

            self.model_type = model_type
            recurrent_dropout = dropout if num_layers > 1 else 0.0
            encoder_class = nn.LSTM if model_type == "lstm" else nn.GRU
            self.encoder = encoder_class(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=recurrent_dropout,
            )
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(hidden_size, 1)

        def forward(self, x, mask=None):
            """Return one logit per sequence."""
            outputs, _ = self.encoder(x)
            if mask is None:
                final_representation = outputs[:, -1, :]
            else:
                lengths = mask.sum(dim=1).long()
                safe_lengths = torch.clamp(lengths, min=1)
                last_indices = safe_lengths - 1
                batch_indices = torch.arange(outputs.size(0), device=outputs.device)
                final_representation = outputs[batch_indices, last_indices, :]

            logits = self.classifier(self.dropout(final_representation)).squeeze(-1)
            return logits

else:

    class RecurrentSequenceClassifier:  # type: ignore[no-redef]
        """Fallback class that raises a clean error when torch is unavailable."""

        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                "PyTorch is required for sequence models. Install torch and rerun."
            )
