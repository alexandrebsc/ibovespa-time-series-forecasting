"""Script for shared config values."""

from utils.lstm_model import TrainingConfig

OPTIMAL_TRAINING_CONIFG = TrainingConfig(
    sequence_length=10, neurons=16, epochs=150, batch_size=16
)
OPTIMAL_TEST_DAYS = 90
