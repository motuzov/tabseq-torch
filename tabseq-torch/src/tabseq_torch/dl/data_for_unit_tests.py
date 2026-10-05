import polars as pl


class AppsForTestLazyLoader:
    def __init__(self, q_tx: pl.LazyFrame):
        self.tx_len_6: pl.LazyFrame = (
            q_tx.group_by("app_id")
            .agg(pl.len())
            .filter(pl.col("len") == 6)
            .select("app_id")
        )

    def _len_filter(self, lf_app_ids) -> list[int]:
        return (
            lf_app_ids.join(self.tx_len_6, on="app_id", how="inner")
            .head(3)
            .collect()["app_id"]
            .to_list()
        )

    def _flag_filte(self, q_targets: pl.LazyFrame, flag: int) -> pl.LazyFrame:
        return q_targets.filter(pl.col("flag") == flag).select("app_id")

    def get_app_ids_for_test(self, q_targets: pl.LazyFrame) -> list[int]:
        result = self._len_filter(self._flag_filte(q_targets, flag=1))
        result += self._len_filter(self._flag_filte(q_targets, flag=0))
        return result
