"""Configuration for LSTM and GRU sequence models."""

SEQUENCE_DATA_DIR = "data/processed/sequences"
MODEL_OUTPUT_DIR = "models/sequences"
REPORT_OUTPUT_DIR = "reports/sequences"

TRAIN_FILE = "train_sequences.npz"
VALIDATION_FILE = "validation_sequences.npz"
TEST_FILE = "test_sequences.npz"

RANDOM_SEED = 42

BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT = 0.2

EARLY_STOPPING_PATIENCE = 5

THRESHOLD = 0.5

DEVICE = "auto"
