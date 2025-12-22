"""Script for shared constant."""

from pathlib import Path

RANDOM_SEED = 1293812

DATA_FOLDER_PATH = f"{Path(__file__).parent.parent.parent.resolve()}/data"
IBOV_HIST_CSV_PATH = f"{DATA_FOLDER_PATH}/ibovespa_history.csv"
IPCA_HIST_CSV_PATH = f"{DATA_FOLDER_PATH}/brazil_inflation_index_ipca.csv"


class Col:
    """Columns names from all df used."""

    # df bvsp
    date = "date"
    close = "close_price"
    open = "open_price"
    max = "max_price"
    min = "min_price"
    vol = "volume"
    var = "percentual_variation"
    # df ipca
    month = "month"
    inflation = "inflation"
    acc_inflation = "accumulated_inflation"
    # df train & test
    ds = "ds"
    y = "y"
