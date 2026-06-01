"""Causal Temporal Convolutional Network for sequence classification."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None


if nn is not None:

    class Chomp1d(nn.Module):
        """Remove right-side padding so Conv1d output remains causal."""

        def __init__(self, chomp_size: int):
            super().__init__()
            self.chomp_size = chomp_size

        def forward(self, x):
            if self.chomp_size == 0:
                return x
            return x[:, :, : -self.chomp_size].contiguous()


    class TemporalBlock(nn.Module):
        """Two-layer residual causal temporal convolution block."""

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            dilation: int,
            padding: int,
            dropout: float,
        ):
            super().__init__()
            self.conv1 = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
            self.chomp1 = Chomp1d(padding)
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(dropout)

            self.conv2 = nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
            self.chomp2 = Chomp1d(padding)
            self.relu2 = nn.ReLU()
            self.dropout2 = nn.Dropout(dropout)

            self.downsample = (
                nn.Conv1d(in_channels, out_channels, kernel_size=1)
                if in_channels != out_channels
                else None
            )
            self.final_relu = nn.ReLU()

        def forward(self, x):
            out = self.conv1(x)
            out = self.chomp1(out)
            out = self.relu1(out)
            out = self.dropout1(out)
            out = self.conv2(out)
            out = self.chomp2(out)
            out = self.relu2(out)
            out = self.dropout2(out)

            residual = x if self.downsample is None else self.downsample(x)
            return self.final_relu(out + residual)


    class CausalTCNClassifier(nn.Module):
        """Causal TCN classifier returning one binary logit per sequence."""

        def __init__(
            self,
            input_size: int,
            num_channels: list[int],
            kernel_size: int,
            dropout: float,
        ):
            super().__init__()
            if not num_channels:
                raise ValueError("num_channels must contain at least one channel size.")

            layers = []
            for level, out_channels in enumerate(num_channels):
                dilation = 2 ** level
                in_channels = input_size if level == 0 else num_channels[level - 1]
                padding = (kernel_size - 1) * dilation
                layers.append(
                    TemporalBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        dilation=dilation,
                        padding=padding,
                        dropout=dropout,
                    )
                )

            self.network = nn.Sequential(*layers)
            self.classifier = nn.Linear(num_channels[-1], 1)

        def forward(self, x, mask=None):
            """Return logits with shape [batch]."""
            tcn_input = x.transpose(1, 2)
            output = self.network(tcn_input).transpose(1, 2)

            if mask is None:
                final_representation = output[:, -1, :]
            else:
                lengths = mask.sum(dim=1).long()
                safe_lengths = torch.clamp(lengths, min=1)
                last_indices = safe_lengths - 1
                batch_indices = torch.arange(output.size(0), device=output.device)
                final_representation = output[batch_indices, last_indices, :]

            return self.classifier(final_representation).squeeze(-1)

else:

    class Chomp1d:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("PyTorch is required for the Causal TCN model.")


    class TemporalBlock:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("PyTorch is required for the Causal TCN model.")


    class CausalTCNClassifier:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("PyTorch is required for the Causal TCN model.")
