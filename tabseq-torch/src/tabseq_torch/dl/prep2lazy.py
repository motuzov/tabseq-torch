import polars as pl


def set_rand_index(g_col_name: str, lf: pl.LazyFrame, idx_start) -> pl.LazyFrame:
    g_hash_col_name = f"{g_col_name}_hash"
    lf.with_columns(pl.col(g_col_name).hash().alias(g_hash_col_name)).sort(
        by=g_hash_col_name
    ).with_columns(pl.col(g_hash_col_name).rank("dense") + idx_start)
