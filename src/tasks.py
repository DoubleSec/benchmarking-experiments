import logging

import polars as pl

logger = logging.getLogger(__name__)


def make_bs(df: pl.DataFrame, cfg) -> pl.DataFrame:
    """Construct data for the balls and strikes task"""

    logger.info("Filtering and segmenting data for balls and strikes (bs) task.")

    return (
        df.filter(pl.col("type").is_in(["B", "S"]))
        .sort(pl.col("game_pk"), pl.col("at_bat_number"), pl.col("pitch_number"))
        .with_row_index("index")
        .with_columns(
            segment=pl.when(pl.col("index") / pl.col("index").count() < cfg.train_frac)
            .then(pl.lit("train"))
            .otherwise(pl.lit("test"))
            .shuffle(),
            target=pl.col("type") == "S",
        )
    )


def make_pt(df: pl.DataFrame, cfg) -> pl.DataFrame:
    """Construct data for the pitch type task"""

    logger.info("Filtering and segmenting data for pitch type (pt) task.")

    # Find the 8 most common pitch types
    pitch_counts = (
        df["pitch_name"].filter(df["pitch_name"].is_not_null()).value_counts(sort=True)
    )
    top_pitches = pitch_counts["pitch_name"].to_list()[:8]
    category_map = {pitch_name: i for i, pitch_name in enumerate(top_pitches)}
    print(category_map)

    return (
        df.filter(
            pl.col("pitch_name").is_not_null() & pl.col("pitch_name").is_in(top_pitches)
        )
        .sort(pl.col("game_pk"), pl.col("at_bat_number"), pl.col("pitch_number"))
        .with_row_index("index")
        .with_columns(
            segment=pl.when(pl.col("index") / pl.col("index").count() < cfg.train_frac)
            .then(pl.lit("train"))
            .otherwise(pl.lit("test"))
            .shuffle(),
            target=pl.col("pitch_name").replace(category_map, return_dtype=pl.Int64),
        )
    )


def make_ip(df: pl.DataFrame, cfg) -> pl.DataFrame:
    """Construct data for the in-play task"""

    logger.info("Filtering and segmenting data for in-play (ip) task.")

    # Find the 8 most common pitch types
    bb_types = df["bb_type"].filter(df["bb_type"].is_not_null()).value_counts(sort=True)
    bb_types = bb_types["bb_type"].to_list()
    category_map = {bb_type: i for i, bb_type in enumerate(bb_types)}
    print(category_map)

    return (
        df.filter(pl.col("bb_type").is_not_null())
        .sort(pl.col("game_pk"), pl.col("at_bat_number"), pl.col("pitch_number"))
        .with_row_index("index")
        .with_columns(
            segment=pl.when(pl.col("index") / pl.col("index").count() < cfg.train_frac)
            .then(pl.lit("train"))
            .otherwise(pl.lit("test"))
            .shuffle(),
            target=pl.col("bb_type").replace(category_map, return_dtype=pl.Int64),
        )
    )


DEFINED_TASKS = {
    "pt": make_pt,
    "bs": make_bs,
    "ip": make_ip,
}
