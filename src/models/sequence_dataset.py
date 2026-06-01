"""PyTorch dataset wrapper for fixed-length eye-tracking sequence arrays."""

from __future__ import annotations

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:
    torch = None

    class Dataset:  # type: ignore[no-redef]
        """Fallback base class used only to provide a clean import error."""

        pass


def load_sequence_npz(npz_path: str) -> dict:
    """Load a sequence .npz file into a plain dictionary."""
    with np.load(npz_path, allow_pickle=True) as data:
        required_keys = [
            "X",
            "y",
            "masks",
            "window_ids",
            "participant_ids",
            "trial_ids",
            "valid_ratios",
            "original_lengths",
        ]
        missing = [key for key in required_keys if key not in data.files]
        if missing:
            raise ValueError(f"{npz_path} is missing arrays: {', '.join(missing)}")
        return {key: data[key] for key in required_keys}


class EyeTrackingSequenceDataset(Dataset):
    """Dataset for one split of fixed-length eye-tracking sequences."""

    def __init__(self, npz_path: str):
        if torch is None:
            raise ModuleNotFoundError(
                "PyTorch is required for sequence models. Install torch and rerun."
            )
        arrays = load_sequence_npz(npz_path)
        self.X = arrays["X"].astype(np.float32)
        self.y = arrays["y"].astype(np.float32)
        self.masks = arrays["masks"].astype(np.float32)
        self.window_ids = arrays["window_ids"].astype(str)
        self.participant_ids = arrays["participant_ids"].astype(str)
        self.trial_ids = arrays["trial_ids"].astype(str)
        self.valid_ratios = arrays["valid_ratios"].astype(np.float32)
        self.original_lengths = arrays["original_lengths"].astype(int)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int) -> dict:
        return {
            "X": torch.as_tensor(self.X[index], dtype=torch.float32),
            "y": torch.as_tensor(self.y[index], dtype=torch.float32),
            "mask": torch.as_tensor(self.masks[index], dtype=torch.float32),
            "window_id": str(self.window_ids[index]),
            "participant_id": str(self.participant_ids[index]),
            "trial_id": str(self.trial_ids[index]),
            "valid_ratio": float(self.valid_ratios[index]),
            "original_length": int(self.original_lengths[index]),
        }
