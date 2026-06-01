import numpy as np

train = np.load("data/processed/sequences/train_sequences.npz", allow_pickle=True)

print("X shape:", train["X"].shape)
print("y shape:", train["y"].shape)
print("masks shape:", train["masks"].shape)

print("First labels:", train["y"][:10])
print("First window IDs:", train["window_ids"][:5])
print("First participant IDs:", train["participant_ids"][:5])