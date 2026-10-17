from tabseq_torch.dl import PdDataset

from .etl import dc, meta


def create_lzdataset(
    cat_columns: list[str], encodeed_cat_columns_setname: str
) -> PdDataset:
    return PdDataset(
        tbts_path=dc.num,
        tbts_groupby_column=meta.g_col_name,
        cat_columns=cat_columns,
        num_columns=meta.num_columns,
        targets_path=dc.targets,
        target_column=meta.labels_col_name,
    )
