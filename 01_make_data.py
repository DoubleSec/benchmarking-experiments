from os import mkdir
import logging
import warnings

import pybaseball
import hydra
import polars as pl

from src.tasks import DEFINED_TASKS

pybaseball.cache.enable()

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="data")
def main(cfg):

    pl.set_random_seed(cfg.seed)

    # Get initial data
    try:
        mkdir(cfg.destination)
        logger.info(f"Creating directory {cfg.destination}")
    except FileExistsError:
        logger.info(f"Directory {cfg.destination} already exists, skipping creation.")

    logger.info("Downloading pitch data")
    try:
        cache_file = f"{cfg.destination}/{cfg.raw_cache_file}"

        pitch_data = pl.read_parquet(cache_file)
        logger.info("Read data from cached data.")

    except FileNotFoundError:
        logger.info(f"{cache_file} not found, loading data from source")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            pitch_data = pybaseball.statcast(cfg.min_date, cfg.max_date)

        pitch_data = pl.from_pandas(pitch_data)
        pitch_data.write_parquet(cache_file)
        logger.info(f"Wrote cached data to {cache_file}")

    assert cfg.task in DEFINED_TASKS

    pitch_data = DEFINED_TASKS[cfg.task](pitch_data, cfg)

    # Write the files
    pitch_data.filter(pl.col("segment") == "train").write_parquet(
        f"{cfg.destination}/{cfg.task}_train.parquet"
    )
    pitch_data.filter(pl.col("segment") == "test").write_parquet(
        f"{cfg.destination}/{cfg.task}_test.parquet"
    )


# Is this what you're supposed to do?
if __name__ == "__main__":
    main()
