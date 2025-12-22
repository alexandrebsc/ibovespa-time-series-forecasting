"""Script for cross validating the model."""

import logging

from utils.config import OPTIMAL_TEST_DAYS, OPTIMAL_TRAINING_CONIFG
from utils.cross_validator import CrossValidator


def main() -> None:
    """Execute main function."""
    logging.basicConfig(level=logging.DEBUG)

    runner = CrossValidator()
    runner.run(OPTIMAL_TRAINING_CONIFG, OPTIMAL_TEST_DAYS)


if __name__ == "__main__":
    main()
